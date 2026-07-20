import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

from data.vibeseq.simulation.naive import simulate_vibeTag

N_AA = 20

PERTURBATION_CONDITIONS = [
    'ground_truth',
    'perturb_gamma50', 'perturb_gamma10', 'perturb_gamma3', 'perturb_gamma1',
    'decay_mild', 'decay_moderate', 'decay_strong', 'decay_severe',
    'lowconf_1pct', 'lowconf_5pct', 'lowconf_10pct', 'lowconf_20pct',
]

# (gamma_0, repetitive_yield r): gamma(i) = gamma_0 * r^i
DECAY_PARAMS = {
    'decay_mild':     (10.0, 0.98),
    'decay_moderate': (10.0, 0.95),
    'decay_strong':   (10.0, 0.90),
    'decay_severe':   (10.0, 0.80),
}


# ---------------------------------------------------------------------------
# Input smoothing / masking
# ---------------------------------------------------------------------------

def apply_mild_noise(p, eps=0.02):
    """Smooth probability vectors slightly toward uniform.
    Prevents exact one-hot inputs; applied identically at train and eval."""
    return (1.0 - eps) * p + eps / N_AA


def apply_bert_mask(probs, mask_rate=0.15):
    """Replace mask_rate fraction of positions with a uniform (1/20) vector.

    Applied only during training. The model must predict true labels at masked
    positions using context from neighbours, since the local signal is removed.
    Loss is computed on all positions so unmasked positions keep the denoising
    objective.

    Args:
        probs:     (B, L, 20) float tensor
        mask_rate: fraction of positions to mask, default 0.15 (BERT standard)

    Returns:
        probs_masked: (B, L, 20) with masked positions set to 1/20
        mask:         (B, L) bool tensor, True where positions were masked
    """
    B, L, V = probs.shape
    uniform = torch.full((V,), 1.0 / V, device=probs.device, dtype=probs.dtype)
    mask = torch.rand(B, L, device=probs.device) < mask_rate
    probs_masked = probs.clone()
    probs_masked[mask] = uniform
    return probs_masked, mask


# ---------------------------------------------------------------------------
# Perturbation generators
# ---------------------------------------------------------------------------

def make_ground_truth(sequence, aa_to_idx):
    true_idx = np.array([aa_to_idx[aa] for aa in sequence])
    gt = np.zeros((len(sequence), N_AA), dtype=np.float32)
    gt[np.arange(len(sequence)), true_idx] = 1.0
    return gt


def make_perturb_dirichlet(sequence, gamma, cm_frac, aa_to_idx, rng):
    """Sample p ~ Dirichlet(gamma * e_i + cm_frac[i,:]) for each position."""
    true_idx = np.array([aa_to_idx[aa] for aa in sequence])
    concentration = cm_frac[true_idx].copy()
    concentration[np.arange(len(sequence)), true_idx] += gamma
    samples = rng.gamma(shape=concentration, scale=1.0)
    return (samples / samples.sum(axis=1, keepdims=True)).astype(np.float32)


def trypsin_digest(sequence, max_len=50):
    """Cleave after K or R, except when the next residue is P.
    Each resulting peptide is truncated to max_len residues.
    Returns a list of peptide strings (empty peptides filtered out)."""
    peptides = []
    start = 0
    for i in range(len(sequence) - 1):
        if sequence[i] in ('K', 'R') and sequence[i + 1] != 'P':
            pep = sequence[start:i + 1]
            if pep:
                peptides.append(pep[:max_len])
            start = i + 1
    if start < len(sequence):
        peptides.append(sequence[start:][:max_len])
    return [p for p in peptides if p]


def make_decay(sequence, gamma_0, r, cm_frac, aa_to_idx, rng):
    """Edman degradation decay model: signal confidence falls with each cycle.

    gamma(i) = gamma_0 * r^i where r is the repetitive yield per cycle.
    At position 0 (N-terminus) concentration is gamma_0 above the confusion
    matrix baseline; by position 49 with r=0.95 it has dropped to ~8% of that.
    """
    L = len(sequence)
    true_idx = np.array([aa_to_idx[aa] for aa in sequence])
    gammas = gamma_0 * (r ** np.arange(L, dtype=np.float64))
    concentration = cm_frac[true_idx].copy()
    concentration[np.arange(L), true_idx] += gammas
    samples = rng.gamma(shape=concentration, scale=1.0)
    return (samples / samples.sum(axis=1, keepdims=True)).astype(np.float32)


def make_lowconf(base_sim, frac, rng):
    """Replace frac of positions with a Dirichlet(1,...,1) sample (near-uniform)."""
    out = base_sim.copy()
    L = out.shape[0]
    n = int(round(L * frac))
    if n > 0:
        positions = rng.choice(L, size=n, replace=False)
        samples = rng.gamma(shape=1.0, scale=1.0, size=(n, N_AA))
        out[positions] = (samples / samples.sum(axis=1, keepdims=True)).astype(np.float32)
    return out


def generate_condition(sequence, condition, cm_frac, classes, aa_to_idx, rng):
    """Generate the input array for a given perturbation condition."""
    if condition == 'ground_truth':
        return make_ground_truth(sequence, aa_to_idx)

    if condition.startswith('perturb_gamma'):
        gamma = float(condition[len('perturb_gamma'):])
        return make_perturb_dirichlet(sequence, gamma, cm_frac, aa_to_idx, rng)

    if condition in DECAY_PARAMS:
        gamma_0, r = DECAY_PARAMS[condition]
        return make_decay(sequence, gamma_0, r, cm_frac, aa_to_idx, rng)

    base = simulate_vibeTag(sequence, classes, cm_frac, rng)

    if condition.startswith('lowconf_'):
        frac = float(condition[len('lowconf_'):-len('pct')]) / 100.0
        return make_lowconf(base, frac, rng)

    raise ValueError(f'Unknown condition: {condition}')


# ---------------------------------------------------------------------------
# Training dataset  (on-the-fly simulation + random perturbation)
# ---------------------------------------------------------------------------

class UnifiedTrainDataset(Dataset):
    """
    Each __getitem__ call:
      1. Picks a random perturbation condition uniformly from PERTURBATION_CONDITIONS.
      2. Runs a fresh VibeTags simulation (different rng each call — equivalent to
         10x or more simulation augmentation per protein).
      3. Applies eps-smoothing to prevent exact one-hot inputs.

    This means every epoch the model sees genuinely different inputs for every protein.
    """

    def __init__(self, sequences, classes, cm_frac, aa_to_idx, mild_eps=0.02,
                 max_len=None, conditions=None):
        """
        sequences  : list of (pid, seq_str) tuples
        max_len    : optional int, discard sequences longer than this
        conditions : optional list of condition names to sample from;
                     defaults to all PERTURBATION_CONDITIONS
        """
        self.classes     = classes
        self.cm_frac     = cm_frac
        self.aa_to_idx   = aa_to_idx
        self.mild_eps    = mild_eps
        self.conditions  = conditions if conditions is not None else PERTURBATION_CONDITIONS

        if max_len is not None:
            sequences = [(pid, seq) for pid, seq in sequences if len(seq) <= max_len]

        self.sequences = sequences
        self.labels = [
            torch.tensor([aa_to_idx[aa] for aa in seq], dtype=torch.long)
            for _, seq in sequences
        ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        _, seq = self.sequences[idx]
        labels = self.labels[idx]

        rng  = np.random.default_rng()          # unseeded → different every call
        cond = self.conditions[rng.integers(len(self.conditions))]

        p = generate_condition(seq, cond, self.cm_frac, self.classes, self.aa_to_idx, rng)
        p = apply_mild_noise(p, eps=self.mild_eps)

        return torch.from_numpy(p), labels


# ---------------------------------------------------------------------------
# Evaluation dataset  (pre-computed from eval h5)
# ---------------------------------------------------------------------------

class UnifiedEvalDataset(Dataset):
    """
    Loads a specific perturbation condition from the pre-generated eval h5.
    Applies the same eps-smoothing as training so metrics are on the same scale.

    The h5 file is assumed to already contain only the val split (tryptic
    peptides keyed by '{pid}_p{i}'). All entries are loaded.
    """

    def __init__(self, h5_path, perturbation_key, aa_to_idx, mild_eps=0.02):
        self.mild_eps = mild_eps
        self.data = []
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                p   = f[key][perturbation_key][()].astype(np.float32)
                seq = f[key]['sequence'][()].decode()
                lbl = np.array([aa_to_idx[aa] for aa in seq], dtype=np.int64)
                p   = apply_mild_noise(p, eps=mild_eps)
                self.data.append((torch.from_numpy(p), torch.from_numpy(lbl)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ---------------------------------------------------------------------------
# Shared collate / loaders
# ---------------------------------------------------------------------------

def collate_fn(batch):
    probs_list, labels_list = zip(*batch)
    max_len = max(p.shape[0] for p in probs_list)
    B = len(batch)

    probs  = torch.zeros(B, max_len, N_AA)
    labels = torch.full((B, max_len), -1, dtype=torch.long)

    for i, (p, l) in enumerate(zip(probs_list, labels_list)):
        L = p.shape[0]
        probs[i,  :L] = p
        labels[i, :L] = l

    return probs, labels


def make_train_loader(sequences, classes, cm_frac, aa_to_idx,
                      batch_size=32, num_workers=4, mild_eps=0.02, max_len=1000,
                      conditions=None):
    ds = UnifiedTrainDataset(sequences, classes, cm_frac, aa_to_idx,
                             mild_eps=mild_eps, max_len=max_len, conditions=conditions)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      collate_fn=collate_fn, num_workers=num_workers,
                      persistent_workers=(num_workers > 0))


def make_eval_loader(h5_path, perturbation_key, aa_to_idx,
                     batch_size=64, mild_eps=0.02):
    ds = UnifiedEvalDataset(h5_path, perturbation_key, aa_to_idx, mild_eps)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
