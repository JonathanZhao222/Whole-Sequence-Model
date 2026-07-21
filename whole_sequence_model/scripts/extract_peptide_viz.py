#!/usr/bin/env python3
"""
Extract prediction data for a single peptide across all 4 decay conditions.
Outputs JSON to stdout for use in the HTML visualization.
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

EVAL_H5    = Path(__file__).parent.parent / 'data' / 'synthetic' / 'eval_tryptic_seed42.h5'
CM_PATH    = Path(__file__).parent.parent / 'assets' / 'rf_cm.npy'
CKPT       = Path(__file__).parent.parent / 'models' / 'checkpoints' / 'unified_cnn_bert_decay.pt'
MILD_EPS   = 0.02
SEED       = 7          # change to pick a different peptide
TARGET_LEN = (18, 35)  # length range to search in

AA_ORDER   = list("ACDEFGHIKLMNPQRSTVWY")
aa_to_idx  = {aa: i for i, aa in enumerate(AA_ORDER)}
DECAY_CONDS = ['decay_mild', 'decay_moderate', 'decay_strong', 'decay_severe']


def main():
    _, cm_frac = load_confusion_matrix(CM_PATH)

    # Find a peptide: right length + a few errors in decay_severe
    rng = np.random.default_rng(SEED)
    with h5py.File(EVAL_H5, 'r') as f:
        keys = list(f.keys())
        rng.shuffle(keys)
        chosen_key = None
        for key in keys:
            seq = f[key]['sequence'][()].decode()
            if not (TARGET_LEN[0] <= len(seq) <= TARGET_LEN[1]):
                continue
            obs_severe = f[key]['decay_severe'][()].astype(np.float32)
            labels = np.array([aa_to_idx[aa] for aa in seq])
            n_err = (obs_severe.argmax(-1) != labels).sum()
            if 2 <= n_err <= len(seq) // 3:
                chosen_key = key
                break

        if chosen_key is None:
            print("No suitable peptide found — try a different SEED", file=sys.stderr)
            sys.exit(1)

        seq = f[chosen_key]['sequence'][()].decode()
        cond_obs = {c: f[chosen_key][c][()].astype(np.float32) for c in DECAY_CONDS}

    labels = np.array([aa_to_idx[aa] for aa in seq])

    # Load model
    model = ConvDenoiser(d=128, dropout=0.1)
    model.load_state_dict(torch.load(CKPT, map_location='cpu'))
    model.eval()

    # Build output data per condition
    conditions_out = []
    for cond in DECAY_CONDS:
        obs = cond_obs[cond]
        obs_s = apply_mild_noise(obs, eps=MILD_EPS)

        argmax_pred = obs.argmax(-1)

        with torch.no_grad():
            logits = model(torch.from_numpy(obs_s).unsqueeze(0))
        model_pred = logits[0].argmax(-1).numpy()

        positions = []
        for i in range(len(seq)):
            positions.append({
                'true':            seq[i],
                'argmax':          AA_ORDER[argmax_pred[i]],
                'model':           AA_ORDER[model_pred[i]],
                'argmax_correct':  bool(argmax_pred[i] == labels[i]),
                'model_correct':   bool(model_pred[i]  == labels[i]),
                'top1_prob':       float(obs[i].max()),
            })

        argmax_acc = sum(p['argmax_correct'] for p in positions) / len(positions)
        model_acc  = sum(p['model_correct']  for p in positions) / len(positions)
        conditions_out.append({
            'condition':   cond,
            'argmax_acc':  argmax_acc,
            'model_acc':   model_acc,
            'positions':   positions,
        })

    print(json.dumps({
        'key': chosen_key,
        'sequence': seq,
        'conditions': conditions_out,
    }))


if __name__ == '__main__':
    main()
