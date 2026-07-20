#!/usr/bin/env python3
"""
Generate the fixed evaluation dataset.

Reads all 20,562 proteins from naive_full_seed42.h5, takes the 20% val split
(same 80/20 seed=42 split used for training), and generates all 13 perturbation
conditions for each val protein. Output is a single HDF5 file used by all
subsequent evaluation runs.

Run once before training:
    python scripts/generate_eval_dataset.py
"""
import sys
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.vibeseq.simulation.naive import load_confusion_matrix, simulate_vibeTag
from models.properties import AA_ORDER
from models.unified_dataset import (
    make_ground_truth, make_perturb_dirichlet,
    make_scrambled, make_lowconf,
    PERTURBATION_CONDITIONS,
)

SEED     = 42
SRC_H5   = Path(__file__).parent.parent / 'data' / 'synthetic' / 'naive_full_seed42.h5'
OUT_H5   = Path(__file__).parent.parent / 'data' / 'synthetic' / 'eval_unified_seed42.h5'
CM_PATH  = Path(__file__).parent.parent / 'assets' / 'rf_cm.npy'
MAX_LEN  = 1000      # discard proteins longer than this

aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}


def main():
    classes, cm_frac = load_confusion_matrix(CM_PATH)

    with h5py.File(SRC_H5, 'r') as f:
        all_pids  = list(f.keys())
        sequences = {pid: f[pid]['sequence'][()].decode() for pid in all_pids}

    # Filter very long proteins (same filter applied in training)
    sequences = {pid: seq for pid, seq in sequences.items() if len(seq) <= MAX_LEN}
    all_pids  = list(sequences.keys())

    # 80/20 split — same seed as training
    rng_split = np.random.default_rng(SEED)
    pids      = rng_split.permutation(all_pids).tolist()
    val_pids  = pids[int(len(pids) * 0.8):]

    print(f'Total proteins (len ≤ {MAX_LEN}): {len(all_pids):,}')
    print(f'Val proteins:                      {len(val_pids):,}')
    print(f'Output → {OUT_H5}')

    rng = np.random.default_rng(SEED + 100)   # separate seed from any training RNG

    with h5py.File(OUT_H5, 'w') as f:
        f.attrs['classes']    = AA_ORDER
        f.attrs['seed']       = SEED
        f.attrs['conditions'] = PERTURBATION_CONDITIONS
        f.attrs['max_len']    = MAX_LEN

        for pid in tqdm(val_pids, desc='Generating eval'):
            seq = sequences[pid]
            grp = f.create_group(pid)
            grp.create_dataset('sequence', data=np.bytes_(seq))

            # Ground truth (one-hot — no simulation needed)
            grp.create_dataset('ground_truth',
                               data=make_ground_truth(seq, aa_to_idx),
                               compression='gzip')

            # Dirichlet perturbations (derived from true sequence, no simulation)
            for gamma in [50, 10, 3, 1]:
                grp.create_dataset(f'perturb_gamma{gamma}',
                                   data=make_perturb_dirichlet(seq, gamma, cm_frac, aa_to_idx, rng),
                                   compression='gzip')

            # Base simulation needed for scramble + lowconf
            base = simulate_vibeTag(seq, classes, cm_frac, rng)

            for pct in [1, 5, 10, 20]:
                frac = pct / 100.0
                grp.create_dataset(f'scramble_{pct}pct',
                                   data=make_scrambled(base, frac, rng),
                                   compression='gzip')
                grp.create_dataset(f'lowconf_{pct}pct',
                                   data=make_lowconf(base, frac, rng),
                                   compression='gzip')

    print('Done.')


if __name__ == '__main__':
    main()
