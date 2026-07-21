#!/usr/bin/env python3
"""
Extract prediction data for a batch of peptides for the interactive viewer.
Outputs a JSON array to stdout.

Selects a mix of:
  - Peptides with errors in decay_severe (most interesting)
  - Peptides with errors in decay_strong too
  - Some perfect peptides (for contrast)
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import h5py

from data.vibeseq.simulation.naive import load_confusion_matrix
from models.conv_denoiser import ConvDenoiser
from models.unified_dataset import apply_mild_noise

EVAL_H5   = Path(__file__).parent.parent / 'data' / 'synthetic' / 'eval_tryptic_seed42.h5'
CM_PATH   = Path(__file__).parent.parent / 'assets' / 'rf_cm.npy'
CKPT      = Path(__file__).parent.parent / 'models' / 'checkpoints' / 'unified_cnn_bert_decay.pt'
MILD_EPS  = 0.02
SEED      = 99
MIN_LEN, MAX_LEN = 12, 40

AA_ORDER    = list("ACDEFGHIKLMNPQRSTVWY")
aa_to_idx   = {aa: i for i, aa in enumerate(AA_ORDER)}
DECAY_CONDS = ['decay_mild', 'decay_moderate', 'decay_strong', 'decay_severe']

N_WITH_ERRORS  = 55   # peptides that have ≥1 argmax error in decay_severe
N_PERFECT      = 20   # peptides that are 100% across all decay conditions


def process_peptide(key, seq, cond_obs, model):
    labels = np.array([aa_to_idx[aa] for aa in seq])
    conditions_out = []
    for cond in DECAY_CONDS:
        obs   = cond_obs[cond]
        obs_s = apply_mild_noise(obs, eps=MILD_EPS)
        am    = obs.argmax(-1)
        with torch.no_grad():
            logits = model(torch.from_numpy(obs_s).unsqueeze(0))
        ml = logits[0].argmax(-1).numpy()

        positions = [
            {
                'true':           seq[i],
                'argmax':         AA_ORDER[am[i]],
                'model':          AA_ORDER[ml[i]],
                'argmax_correct': bool(am[i] == labels[i]),
                'model_correct':  bool(ml[i]  == labels[i]),
                'top1_prob':      round(float(obs[i].max()), 4),
            }
            for i in range(len(seq))
        ]
        conditions_out.append({
            'condition':  cond,
            'argmax_acc': round(sum(p['argmax_correct'] for p in positions) / len(positions), 4),
            'model_acc':  round(sum(p['model_correct']  for p in positions) / len(positions), 4),
            'positions':  positions,
        })
    return {'key': key, 'sequence': seq, 'conditions': conditions_out}


def main():
    _, cm_frac = load_confusion_matrix(CM_PATH)

    model = ConvDenoiser(d=128, dropout=0.1)
    model.load_state_dict(torch.load(CKPT, map_location='cpu'))
    model.eval()

    rng = np.random.default_rng(SEED)

    with_errors, perfect = [], []

    with h5py.File(EVAL_H5, 'r') as f:
        keys = list(f.keys())
        rng.shuffle(keys)

        for key in keys:
            if len(with_errors) >= N_WITH_ERRORS and len(perfect) >= N_PERFECT:
                break
            seq = f[key]['sequence'][()].decode()
            if not (MIN_LEN <= len(seq) <= MAX_LEN):
                continue
            labels = np.array([aa_to_idx[aa] for aa in seq])
            obs_severe = f[key]['decay_severe'][()].astype(np.float32)
            n_err = (obs_severe.argmax(-1) != labels).sum()

            cond_obs = {c: f[key][c][()].astype(np.float32) for c in DECAY_CONDS}
            record = process_peptide(key, seq, cond_obs, model)

            if n_err >= 1 and len(with_errors) < N_WITH_ERRORS:
                with_errors.append(record)
                print(f"  [error]   {key}  L={len(seq)}  n_err={n_err}", file=sys.stderr)
            elif n_err == 0 and len(perfect) < N_PERFECT:
                perfect.append(record)
                print(f"  [perfect] {key}  L={len(seq)}", file=sys.stderr)

    batch = with_errors + perfect
    rng.shuffle(batch)
    print(json.dumps(batch))
    print(f"\nDone: {len(with_errors)} with-error + {len(perfect)} perfect = {len(batch)} total",
          file=sys.stderr)


if __name__ == '__main__':
    main()
