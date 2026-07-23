#!/usr/bin/env python3
"""
Evaluate beam-search rank of the ground truth sequence.

For each (peptide, condition) pair, runs beam search with the CNN + fixed CRF
and records the 1-indexed rank of the ground truth sequence among the top-k
candidates. Reports per-condition statistics and plots a rank distribution.

Usage:
    python scripts/eval_crf_rank.py [--k 500] [--n 2000] [--ckpt <path>]
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

from models.properties import AA_ORDER
from models.conv_denoiser import ConvDenoiser
from models.crf import FixedCRF, build_T_from_asset
from models.unified_dataset import apply_mild_noise

ROOT        = Path(__file__).parent.parent
EVAL_H5     = ROOT / 'data'   / 'synthetic' / 'eval_tryptic_seed42.h5'
BIGRAM_PATH = ROOT / 'assets' / 'aa_bigram_transitions.npy'
CRF_CKPT    = ROOT / 'models' / 'checkpoints' / 'unified_cnn_bert_crf.pt'
BASE_CKPT   = ROOT / 'models' / 'checkpoints' / 'unified_cnn_bert_decay.pt'

DECAY_CONDS  = ['decay_mild', 'decay_moderate', 'decay_strong', 'decay_severe']
COND_LABELS  = {
    'decay_mild':     'Mild (r=0.98)',
    'decay_moderate': 'Moderate (r=0.95)',
    'decay_strong':   'Strong (r=0.90)',
    'decay_severe':   'Severe (r=0.80)',
}
MILD_EPS = 0.02
aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--k',    type=int, default=500,    help='beam width')
    p.add_argument('--n',    type=int, default=2000,   help='peptides per condition')
    p.add_argument('--ckpt', type=str, default=str(CRF_CKPT),
                   help='model checkpoint (default: CRF-trained)')
    p.add_argument('--compare-base', action='store_true',
                   help='also evaluate base model (no CRF) for comparison')
    return p.parse_args()


@torch.no_grad()
def rank_peptides(model, crf, eval_h5, cond, keys, k, device):
    """Return list of ranks (int or None) for each peptide under given condition."""
    ranks = []
    model.eval()
    with h5py.File(eval_h5, 'r') as f:
        for key in tqdm(keys, desc=cond, leave=False):
            seq    = f[key]['sequence'][()].decode()
            labels = torch.tensor([aa_to_idx[aa] for aa in seq])
            obs    = f[key][cond][()].astype(np.float32)
            obs_s  = apply_mild_noise(obs, eps=MILD_EPS)
            logits = model(torch.from_numpy(obs_s).unsqueeze(0).to(device))
            emissions = logits[0].cpu()          # (L, 20)
            rank = crf.ground_truth_rank(emissions, labels, k=k)
            ranks.append(rank)
    return ranks


def summarise(ranks, k, cond_label):
    found    = [r for r in ranks if r is not None]
    n_found  = len(found)
    n_total  = len(ranks)
    found_arr = np.array(found) if found else np.array([k + 1])

    print(f'\n{cond_label}')
    print(f'  Found in top-{k}: {n_found}/{n_total} ({n_found/n_total:.1%})')
    if found:
        print(f'  Rank  mean={found_arr.mean():.1f}  '
              f'median={np.median(found_arr):.0f}  '
              f'p90={np.percentile(found_arr, 90):.0f}')
        for threshold in [1, 5, 10, 50, 100]:
            pct = (found_arr <= threshold).mean()
            print(f'  Top-{threshold:<3d}: {pct:.1%}')
    return found_arr


def plot_rank_distributions(all_ranks, k, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flat
    bins = np.logspace(0, np.log10(k), 40)

    for ci, (cond, ranks_arr) in enumerate(all_ranks.items()):
        ax = axes[ci]
        ax.hist(ranks_arr, bins=bins, color='#4e9af1', edgecolor='white',
                linewidth=0.4, alpha=0.85)
        ax.axvline(np.median(ranks_arr), color='#e05c5c', lw=1.5,
                   linestyle='--', label=f'Median={np.median(ranks_arr):.0f}')
        ax.set_xscale('log')
        ax.set_xlabel('Rank of ground truth sequence')
        ax.set_ylabel('Count')
        ax.set_title(COND_LABELS[cond])
        ax.legend(fontsize=9)

    fig.suptitle(
        f'Ground-truth sequence rank in beam (k={k})\n'
        f'CNN + fixed CRF (proteome log-odds transitions)',
        fontsize=12,
    )
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'\nFigure saved → {save_path}')


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    k      = args.k

    # Load T and CRF
    T   = build_T_from_asset(BIGRAM_PATH)
    crf = FixedCRF(T, device='cpu')   # beam search runs on CPU (numpy)
    print(f'T: [{T.min():.3f}, {T.max():.3f}]  '
          f'most-penalised={AA_ORDER[np.unravel_index(T.argmin(),(20,20))[0]]}'
          f'→{AA_ORDER[np.unravel_index(T.argmin(),(20,20))[1]]} ({T.min():.3f})')

    # Load model
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f'CRF checkpoint not found at {ckpt_path}, falling back to {BASE_CKPT}')
        ckpt_path = BASE_CKPT
    model = ConvDenoiser(d=128, dropout=0.1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f'Loaded: {ckpt_path}')

    # Sample eval keys
    rng = np.random.default_rng(42)
    with h5py.File(EVAL_H5, 'r') as f:
        all_keys = list(f.keys())
    rng.shuffle(all_keys)
    keys = all_keys[:args.n]

    # Evaluate
    all_ranks = {}
    for cond in DECAY_CONDS:
        ranks      = rank_peptides(model, crf, EVAL_H5, cond, keys, k, device)
        # Replace None (not found) with k+1 for statistics
        ranks_arr  = np.array([r if r is not None else k + 1 for r in ranks])
        all_ranks[cond] = ranks_arr
        summarise(ranks, k, COND_LABELS[cond])

    # Plot
    save_path = ROOT / 'crf_rank_distribution.png'
    plot_rank_distributions(all_ranks, k, save_path)


if __name__ == '__main__':
    main()
