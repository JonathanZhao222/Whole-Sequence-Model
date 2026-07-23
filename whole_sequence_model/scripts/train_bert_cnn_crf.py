#!/usr/bin/env python3
"""
Fine-tune ConvDenoiser with a fixed CRF transition matrix.

Training loss switches from per-position cross-entropy to CRF NLL
(forward algorithm). T is computed from proteome bigram log-odds and
held frozen throughout — only the CNN weights update.

Initialises from the existing decay checkpoint so convergence is fast.
Saves to models/checkpoints/unified_cnn_bert_crf.pt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import h5py
from tqdm import tqdm

from models.properties import AA_ORDER
from models.conv_denoiser import ConvDenoiser
from models.crf import FixedCRF, build_T_from_asset
from models.unified_dataset import (
    DECAY_PARAMS, apply_mild_noise, apply_bert_mask, trypsin_digest,
)
from data.vibeseq.simulation.naive import load_confusion_matrix

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
SRC_H5     = ROOT / 'data' / 'synthetic' / 'naive_full_seed42.h5'
EVAL_H5    = ROOT / 'data' / 'synthetic' / 'eval_tryptic_seed42.h5'
INIT_CKPT  = ROOT / 'models' / 'checkpoints' / 'unified_cnn_bert_decay.pt'
SAVE_CKPT  = ROOT / 'models' / 'checkpoints' / 'unified_cnn_bert_crf.pt'
BIGRAM_PATH = ROOT / 'assets' / 'aa_bigram_transitions.npy'
CM_PATH    = ROOT / 'assets' / 'rf_cm.npy'

# ── hyper-params ──────────────────────────────────────────────────────────────
SEED        = 42
EPOCHS      = 15          # fewer needed: initialising from trained checkpoint
BATCH       = 32
LR          = 3e-4        # lower LR for fine-tuning
MASK_RATE   = 0.15
MILD_EPS    = 0.02
N_TRAIN     = 50_000      # peptides per epoch
DECAY_CONDS = list(DECAY_PARAMS.keys())
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}
rng = np.random.default_rng(SEED)


# ── build training peptides (tryptic, from train split of source H5) ─────────
def load_train_peptides():
    classes, cm_frac = load_confusion_matrix(CM_PATH)
    all_pids = []
    sequences = {}
    with h5py.File(SRC_H5, 'r') as f:
        for pid in f.keys():
            seq = f[pid]['sequence'][()].decode()
            sequences[pid] = seq
            all_pids.append(pid)

    pids = np.random.default_rng(SEED).permutation(all_pids).tolist()
    train_pids = set(pids[:int(len(pids) * 0.8)])

    peptides = []
    for pid in train_pids:
        for pep in trypsin_digest(sequences[pid]):
            peptides.append((pid, pep))

    print(f'Train tryptic peptides: {len(peptides):,}')
    return peptides, classes, cm_frac


def make_decay_obs(seq, cond, cm_frac, rng_local):
    """Simulate one decay observation for a peptide sequence."""
    from models.unified_dataset import make_decay
    gamma_0, r = DECAY_PARAMS[cond]
    return make_decay(seq, gamma_0, r, cm_frac, aa_to_idx, rng_local).astype(np.float32)


# ── collate a batch ───────────────────────────────────────────────────────────
def make_batch(peptides, cm_frac, batch_size, rng_local):
    indices = rng_local.choice(len(peptides), size=batch_size, replace=True)
    max_len = max(len(peptides[i][1]) for i in indices)

    probs_batch  = np.zeros((batch_size, max_len, 20), dtype=np.float32)
    labels_batch = np.full((batch_size, max_len), -1, dtype=np.int64)

    for b, idx in enumerate(indices):
        _, seq = peptides[idx]
        L      = len(seq)
        cond   = rng_local.choice(DECAY_CONDS)
        obs    = make_decay_obs(seq, cond, cm_frac, rng_local)
        probs_batch[b, :L]  = obs
        labels_batch[b, :L] = [aa_to_idx[aa] for aa in seq]

    return probs_batch, labels_batch


# ── eval: per-position accuracy on each decay condition ──────────────────────
@torch.no_grad()
def eval_accuracy(model, eval_h5_path, n_peptides=2000):
    model.eval()
    results = {}
    with h5py.File(eval_h5_path, 'r') as f:
        keys = list(f.keys())
        rng.shuffle(keys)
        keys = keys[:n_peptides]
        for cond in DECAY_CONDS:
            correct = total = 0
            for key in keys:
                seq    = f[key]['sequence'][()].decode()
                labels = np.array([aa_to_idx[aa] for aa in seq])
                obs    = f[key][cond][()].astype(np.float32)
                obs_s  = apply_mild_noise(obs, eps=MILD_EPS)
                logits = model(torch.from_numpy(obs_s).unsqueeze(0).to(DEVICE))
                preds  = logits[0].argmax(-1).cpu().numpy()
                correct += (preds == labels).sum()
                total   += len(labels)
            results[cond] = correct / total
    model.train()
    return results


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)

    # Build fixed CRF from bigram log-odds
    T = build_T_from_asset(BIGRAM_PATH)
    crf = FixedCRF(T, device=DEVICE)
    print(f'T range: [{T.min():.3f}, {T.max():.3f}]  '
          f'(negative = biologically penalised)')
    # Show 3 most penalised bigrams
    flat = [(T[i,j], AA_ORDER[i], AA_ORDER[j])
            for i in range(20) for j in range(20)]
    flat.sort()
    print('Most penalised bigrams:', ', '.join(f'{a}→{b}({v:.2f})'
                                               for v, a, b in flat[:5]))
    print('Most favoured bigrams: ', ', '.join(f'{a}→{b}({v:.2f})'
                                               for v, a, b in flat[-5:]))

    # Load model from decay checkpoint
    model = ConvDenoiser(d=128, dropout=0.1).to(DEVICE)
    model.load_state_dict(torch.load(INIT_CKPT, map_location=DEVICE))
    print(f'Loaded checkpoint: {INIT_CKPT}')

    # Load training peptides
    peptides, classes, cm_frac = load_train_peptides()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    steps_per_epoch = N_TRAIN // BATCH
    best_avg_acc = 0.0
    rng_train = np.random.default_rng(SEED + 1)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in tqdm(range(steps_per_epoch), desc=f'Epoch {epoch}', leave=False):
            probs_np, labels_np = make_batch(peptides, cm_frac, BATCH, rng_train)

            # BERT masking on input only
            probs_t  = torch.from_numpy(probs_np).to(DEVICE)
            labels_t = torch.from_numpy(labels_np).to(DEVICE)
            probs_t, _ = apply_bert_mask(probs_t, MASK_RATE)

            emissions = model(probs_t)   # (B, Lmax, 20)

            loss = crf.batch_nll(emissions, labels_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / steps_per_epoch

        acc = eval_accuracy(model, EVAL_H5)
        avg_acc = np.mean(list(acc.values()))
        print(f'Epoch {epoch:2d}  loss={avg_loss:.4f}  '
              + '  '.join(f'{c.split("_")[1]}={v:.3%}' for c, v in acc.items())
              + f'  avg={avg_acc:.3%}')

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            torch.save(model.state_dict(), SAVE_CKPT)
            print(f'  ✓ saved → {SAVE_CKPT}')

    print(f'\nBest avg acc: {best_avg_acc:.3%}')


if __name__ == '__main__':
    main()
