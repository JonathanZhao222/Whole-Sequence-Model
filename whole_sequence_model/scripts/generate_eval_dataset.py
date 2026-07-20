#!/usr/bin/env python3
"""
Generate the fixed evaluation dataset (tryptic peptides, decay conditions).

Reads all proteins from naive_full_seed42.h5, takes the 20% val split
(same 80/20 seed=42 split as training), applies trypsin digestion (cleave
after K/R not before P, truncate to 50 aa), and generates all 13 perturbation
conditions per peptide. Output is eval_tryptic_seed42.h5.

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
    PERTURBATION_CONDITIONS, DECAY_PARAMS,
    make_ground_truth, make_perturb_dirichlet, make_decay, make_lowconf,
    trypsin_digest,
)

SEED     = 42
MAX_PEP  = 50
SRC_H5   = Path(__file__).parent.parent / 'data' / 'synthetic' / 'naive_full_seed42.h5'
OUT_H5   = Path(__file__).parent.parent / 'data' / 'synthetic' / 'eval_tryptic_seed42.h5'
CM_PATH  = Path(__file__).parent.parent / 'assets' / 'rf_cm.npy'

aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}


def main():
    classes, cm_frac = load_confusion_matrix(CM_PATH)

    with h5py.File(SRC_H5, 'r') as f:
        all_pids  = list(f.keys())
        sequences = {pid: f[pid]['sequence'][()].decode() for pid in all_pids}

    # 80/20 split — identical seed to training
    rng_split = np.random.default_rng(SEED)
    pids      = rng_split.permutation(all_pids).tolist()
    val_pids  = pids[int(len(pids) * 0.8):]

    # Build flat list of (key, peptide_seq) for all val tryptic peptides
    val_peptides = []
    for pid in val_pids:
        for i, pep in enumerate(trypsin_digest(sequences[pid], max_len=MAX_PEP)):
            val_peptides.append((f'{pid}_p{i}', pep))

    print(f'Val proteins:  {len(val_pids):,}')
    print(f'Val peptides:  {len(val_peptides):,}  (avg {len(val_peptides)/len(val_pids):.1f} per protein)')
    print(f'Output → {OUT_H5}')

    rng = np.random.default_rng(SEED + 100)

    with h5py.File(OUT_H5, 'w') as f:
        f.attrs['classes']    = AA_ORDER
        f.attrs['seed']       = SEED
        f.attrs['conditions'] = PERTURBATION_CONDITIONS
        f.attrs['max_pep']    = MAX_PEP

        for key, seq in tqdm(val_peptides, desc='Generating eval'):
            grp = f.create_group(key)
            grp.create_dataset('sequence', data=np.bytes_(seq))

            grp.create_dataset('ground_truth',
                               data=make_ground_truth(seq, aa_to_idx),
                               compression='gzip')

            for gamma in [50, 10, 3, 1]:
                grp.create_dataset(f'perturb_gamma{gamma}',
                                   data=make_perturb_dirichlet(seq, gamma, cm_frac, aa_to_idx, rng),
                                   compression='gzip')

            for cond, (gamma_0, r) in DECAY_PARAMS.items():
                grp.create_dataset(cond,
                                   data=make_decay(seq, gamma_0, r, cm_frac, aa_to_idx, rng),
                                   compression='gzip')

            # lowconf needs base VibeTags simulation
            base = simulate_vibeTag(seq, classes, cm_frac, rng)
            for pct in [1, 5, 10, 20]:
                frac = pct / 100.0
                grp.create_dataset(f'lowconf_{pct}pct',
                                   data=make_lowconf(base, frac, rng),
                                   compression='gzip')

    print('Done.')


if __name__ == '__main__':
    main()
