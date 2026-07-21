#!/usr/bin/env python3
"""
Lift of per-position Bayes (decoder 2) over argmax (decoder 1).

Decoder 1: argmax(obs_i)           — ignores the noise channel entirely
Decoder 2: argmax(obs_i @ cm.T)    — inverts confusion matrix per-position,
                                      uniform prior over amino acids

Lift = D2_acc - D1_acc.

The confusion matrix cm[s, t] = P(output_call = t | true = s) makes D2 a
soft version of the Bayes-optimal single-position decoder: for each candidate
true amino acid s, the likelihood of the observed soft vector o is
  P(o | s) ≈ Σ_t o[t] · cm[s, t]  = (obs @ cm.T)[s]
and we take the argmax over s.

Run on the pre-generated eval_tryptic_seed42.h5 (249k val peptides).
Subsamples MAX_EVAL peptides for speed; set to None to use all.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h5py
from tqdm import tqdm

from data.vibeseq.simulation.naive import load_confusion_matrix

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

PERTURBATION_CONDITIONS = [
    'ground_truth',
    'perturb_gamma50', 'perturb_gamma10', 'perturb_gamma3', 'perturb_gamma1',
    'decay_mild', 'decay_moderate', 'decay_strong', 'decay_severe',
    'lowconf_1pct', 'lowconf_5pct', 'lowconf_10pct', 'lowconf_20pct',
]

DECAY_PARAMS = {
    'decay_mild':     (10.0, 0.98),
    'decay_moderate': (10.0, 0.95),
    'decay_strong':   (10.0, 0.90),
    'decay_severe':   (10.0, 0.80),
}

EVAL_H5  = Path(__file__).parent.parent / 'data' / 'synthetic' / 'eval_tryptic_seed42.h5'
CM_PATH  = Path(__file__).parent.parent / 'assets' / 'rf_cm.npy'
MAX_EVAL = 50_000   # set None to use all 249k peptides
MAX_POS  = 50       # tryptic cap

aa_to_idx  = {aa: i for i, aa in enumerate(AA_ORDER)}
decay_conds = list(DECAY_PARAMS.keys())


def main():
    _, cm_frac = load_confusion_matrix(CM_PATH)   # (20, 20), float64
    cm_frac    = cm_frac.astype(np.float32)

    with h5py.File(EVAL_H5, 'r') as f:
        all_keys = list(f.keys())

    rng  = np.random.default_rng(42)
    keys = (rng.choice(all_keys, size=min(MAX_EVAL, len(all_keys)), replace=False).tolist()
            if MAX_EVAL is not None else all_keys)
    print(f"Evaluating {len(keys):,} peptides across {len(PERTURBATION_CONDITIONS)} conditions\n")

    n = len(PERTURBATION_CONDITIONS)
    d1_correct = np.zeros(n, dtype=np.int64)
    d2_correct = np.zeros(n, dtype=np.int64)
    n_tokens   = np.zeros(n, dtype=np.int64)

    # per-position accumulators for decay conditions only
    decay_ci   = {c: PERTURBATION_CONDITIONS.index(c) for c in decay_conds}
    d1_pos = np.zeros((len(decay_conds), MAX_POS), dtype=np.int64)
    d2_pos = np.zeros((len(decay_conds), MAX_POS), dtype=np.int64)
    n_pos  = np.zeros((len(decay_conds), MAX_POS), dtype=np.int64)

    with h5py.File(EVAL_H5, 'r') as f:
        for key in tqdm(keys, desc='Evaluating'):
            seq    = f[key]['sequence'][()].decode()
            labels = np.array([aa_to_idx[aa] for aa in seq], dtype=np.int64)
            L      = len(labels)

            for ci, cond in enumerate(PERTURBATION_CONDITIONS):
                obs = f[key][cond][()].astype(np.float32)   # (L, 20)

                d1 = obs.argmax(-1)                          # (L,)
                d2 = (obs @ cm_frac.T).argmax(-1)            # (L,)

                d1_correct[ci] += (d1 == labels).sum()
                d2_correct[ci] += (d2 == labels).sum()
                n_tokens[ci]   += L

                if cond in decay_ci:
                    di = list(decay_conds).index(cond)
                    mask = np.arange(L)
                    d1_pos[di, mask] += (d1 == labels)
                    d2_pos[di, mask] += (d2 == labels)
                    n_pos[di, mask]  += 1

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'Condition':<25}  {'D1 argmax':>10}  {'D2 Bayes':>10}  {'Lift':>8}")
    print("─" * 60)
    for ci, cond in enumerate(PERTURBATION_CONDITIONS):
        d1a  = d1_correct[ci] / n_tokens[ci]
        d2a  = d2_correct[ci] / n_tokens[ci]
        lift = d2a - d1a
        flag = "  ◀" if cond in decay_ci else ""
        print(f"{cond:<25}  {d1a:>9.3%}  {d2a:>9.3%}  {lift:>+7.3%}{flag}")

    # ── Per-position accuracy for decay conditions ─────────────────────────
    print(f"\nPer-position accuracy (decay conditions):")
    header = f"{'pos':>3}  " + "  ".join(f"{c:>14}" for c in decay_conds)
    sub    = f"     " + "  ".join(f"{'D1':>6} {'D2':>6}" for _ in decay_conds)
    print(header)
    print(sub)
    print("─" * (5 + 16 * len(decay_conds)))

    for p in range(MAX_POS):
        if n_pos[0, p] == 0:
            break
        row = f"{p:>3}  "
        for di in range(len(decay_conds)):
            if n_pos[di, p] > 0:
                d1a = d1_pos[di, p] / n_pos[di, p]
                d2a = d2_pos[di, p] / n_pos[di, p]
                row += f"  {d1a:>5.1%} {d2a:>5.1%}"
            else:
                row += f"  {'—':>5} {'—':>5}"
        print(row)

    # ── Lift summary by regime ─────────────────────────────────────────────
    print(f"\nLift summary (D2 − D1), early (pos 0–4) vs late (pos 10–49) positions:")
    print(f"{'Condition':<20}  {'early':>8}  {'late':>8}")
    print("─" * 42)
    for di, cond in enumerate(decay_conds):
        early_mask = np.arange(5)
        late_mask  = np.arange(10, MAX_POS)
        valid_late = late_mask[n_pos[di, late_mask] > 0]

        def lift_over(mask):
            d1 = d1_pos[di, mask].sum()
            d2 = d2_pos[di, mask].sum()
            nt = n_pos[di, mask].sum()
            return (d2 - d1) / nt if nt > 0 else float('nan')

        print(f"{cond:<20}  {lift_over(early_mask):>+7.3%}  {lift_over(valid_late):>+7.3%}")


if __name__ == '__main__':
    main()
