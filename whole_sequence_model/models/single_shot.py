import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py

from models.properties import PROPERTY_TABLE, BACKGROUND_FREQ


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProteinDataset(Dataset):
    """
    Loads one perturbation level from the HDF5 file.
    Preloads all sequences into memory for speed.
    """

    def __init__(self, h5_path, dataset_key, protein_ids, aa_to_idx):
        self.data = []
        with h5py.File(h5_path, 'r') as f:
            for pid in protein_ids:
                p   = f[pid][dataset_key][()].astype(np.float32)   # (L, 20)
                seq = f[pid]['sequence'][()].decode()
                labels = np.array([aa_to_idx[aa] for aa in seq], dtype=np.int64)
                self.data.append((p, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p, labels = self.data[idx]
        return torch.from_numpy(p), torch.from_numpy(labels)


def collate_fn(batch):
    """Pad a batch of variable-length sequences."""
    probs_list, labels_list = zip(*batch)
    lengths = [p.shape[0] for p in probs_list]
    max_len = max(lengths)
    B       = len(batch)

    probs  = torch.zeros(B, max_len, 20)
    labels = torch.full((B, max_len), -1, dtype=torch.long)  # -1 ignored in loss

    for i, (p, l) in enumerate(zip(probs_list, labels_list)):
        L = p.shape[0]
        probs[i,  :L] = p
        labels[i, :L] = l

    return probs, labels


def make_loaders(h5_path, dataset_key, aa_to_idx, batch_size=32, val_frac=0.2, seed=42):
    with h5py.File(h5_path, 'r') as f:
        all_pids = list(f.keys())

    rng   = np.random.default_rng(seed)
    pids  = rng.permutation(all_pids).tolist()
    split = int(len(pids) * (1 - val_frac))
    train_pids, val_pids = pids[:split], pids[split:]

    train_ds = ProteinDataset(h5_path, dataset_key, train_pids, aa_to_idx)
    val_ds   = ProteinDataset(h5_path, dataset_key, val_pids,   aa_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, train_pids, val_pids


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SingleShotDenoiser(nn.Module):
    """
    Per-position MLP denoiser with physicochemical property features.

    For each position:
        1. Compute expected properties: e = p @ T     (L, k)
        2. Z-score normalise e
        3. Concatenate [p, e_norm]                    (L, 20+k)
        4. Apply shared MLP                           (L, 20) logits
    """

    def __init__(self, hidden_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        k = PROPERTY_TABLE.shape[1]

        # Fixed property table — not updated during training
        self.register_buffer('prop_table', torch.from_numpy(PROPERTY_TABLE))  # (20, k)

        # Z-score stats — set by fit_stats(), frozen during training
        self.register_buffer('prop_mean', torch.zeros(k))
        self.register_buffer('prop_std',  torch.ones(k))

        # Per-position MLP
        dims   = [20 + k] + [hidden_dim] * n_layers
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, 20))
        self.mlp = nn.Sequential(*layers)

    @torch.no_grad()
    def fit_stats(self, loader, device='cpu'):
        """Compute and store z-score statistics from a DataLoader."""
        all_props = []
        self.eval()
        for probs, _ in loader:
            probs = probs.to(device)
            e = probs @ self.prop_table          # (B, L, k)
            all_props.append(e.reshape(-1, e.shape[-1]).cpu())
        all_props = torch.cat(all_props, dim=0)
        self.prop_mean.copy_(all_props.mean(dim=0))
        self.prop_std.copy_(all_props.std(dim=0).clamp(min=1e-6))

    def forward(self, p):
        """
        p : (B, L, 20) noisy probability vectors
        returns: (B, L, 20) logits
        """
        e      = p @ self.prop_table                        # (B, L, k)
        e_norm = (e - self.prop_mean) / self.prop_std       # (B, L, k)
        x      = torch.cat([p, e_norm], dim=-1)             # (B, L, 20+k)
        return self.mlp(x)                                  # (B, L, 20)


# ---------------------------------------------------------------------------
# Variant A: background deviation input feature
# ---------------------------------------------------------------------------

class SingleShotDenoiserA(nn.Module):
    """
    Extends SingleShotDenoiser with a background deviation feature.

    Extra input: (p - bg_freq), the per-AA deviation from human proteome
    background frequencies. Positive = over-represented here, negative =
    under-represented. Input dim: 20 + 7 + 20 = 47.
    """

    def __init__(self, hidden_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        k = PROPERTY_TABLE.shape[1]

        self.register_buffer('prop_table', torch.from_numpy(PROPERTY_TABLE))   # (20, k)
        self.register_buffer('bg_freq',    torch.from_numpy(BACKGROUND_FREQ))  # (20,)

        self.register_buffer('prop_mean', torch.zeros(k))
        self.register_buffer('prop_std',  torch.ones(k))

        input_dim = 20 + k + 20   # p, e_norm, p - bg_freq
        dims   = [input_dim] + [hidden_dim] * n_layers
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, 20))
        self.mlp = nn.Sequential(*layers)

    @torch.no_grad()
    def fit_stats(self, loader, device='cpu'):
        all_props = []
        self.eval()
        for probs, _ in loader:
            probs = probs.to(device)
            e = probs @ self.prop_table
            all_props.append(e.reshape(-1, e.shape[-1]).cpu())
        all_props = torch.cat(all_props, dim=0)
        self.prop_mean.copy_(all_props.mean(dim=0))
        self.prop_std.copy_(all_props.std(dim=0).clamp(min=1e-6))

    def forward(self, p):
        e      = p @ self.prop_table
        e_norm = (e - self.prop_mean) / self.prop_std
        dev    = p - self.bg_freq                          # (B, L, 20)
        x      = torch.cat([p, e_norm, dev], dim=-1)      # (B, L, 47)
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Variant C: output logit bias initialised to log(background freq)
# ---------------------------------------------------------------------------

class SingleShotDenoiserC(nn.Module):
    """
    Extends SingleShotDenoiser with a trainable output bias initialised to
    log(bg_freq). When input features are uninformative the model's default
    prediction matches the proteome background distribution rather than
    uniform. Same input dim as the base model (27).
    """

    def __init__(self, hidden_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        k = PROPERTY_TABLE.shape[1]

        self.register_buffer('prop_table', torch.from_numpy(PROPERTY_TABLE))

        self.register_buffer('prop_mean', torch.zeros(k))
        self.register_buffer('prop_std',  torch.ones(k))

        dims   = [20 + k] + [hidden_dim] * n_layers
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, 20))
        self.mlp = nn.Sequential(*layers)

        log_bg = torch.log(torch.from_numpy(BACKGROUND_FREQ))
        self.output_bias = nn.Parameter(log_bg)            # trainable, (20,)

    @torch.no_grad()
    def fit_stats(self, loader, device='cpu'):
        all_props = []
        self.eval()
        for probs, _ in loader:
            probs = probs.to(device)
            e = probs @ self.prop_table
            all_props.append(e.reshape(-1, e.shape[-1]).cpu())
        all_props = torch.cat(all_props, dim=0)
        self.prop_mean.copy_(all_props.mean(dim=0))
        self.prop_std.copy_(all_props.std(dim=0).clamp(min=1e-6))

    def forward(self, p):
        e      = p @ self.prop_table
        e_norm = (e - self.prop_mean) / self.prop_std
        x      = torch.cat([p, e_norm], dim=-1)            # (B, L, 27)
        return self.mlp(x) + self.output_bias              # bias broadcast over (B, L)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for probs, labels in loader:
            probs, labels = probs.to(device), labels.to(device)
            logits = model(probs)                          # (B, L, 20)

            loss = F.cross_entropy(
                logits.reshape(-1, 20),
                labels.reshape(-1),
                ignore_index=-1,
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mask    = labels != -1
            preds   = logits.argmax(dim=-1)
            total_correct += (preds[mask] == labels[mask]).sum().item()
            total_tokens  += mask.sum().item()
            total_loss    += loss.item() * mask.sum().item()

    return total_loss / total_tokens, total_correct / total_tokens
