#!/usr/bin/env python3
"""
Compute a 20x20 amino acid bigram transition matrix from the full human proteome
and save it to assets/aa_bigram_transitions.npy.

Matrix[i, j] = log P(AA_j | AA_i), estimated from observed consecutive AA pairs
with pseudocount smoothing to avoid log(0).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from vibeseq.data import parse_fasta, filter_standard
from models.properties import AA_ORDER

FASTA    = Path(__file__).parent.parent / 'data' / 'raw' / 'UP000005640_9606.fasta.gz'
OUT_PATH = Path(__file__).parent.parent / 'assets' / 'aa_bigram_transitions.npy'
PSEUDOCOUNT = 1.0

aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}
N_AA = len(AA_ORDER)


def main():
    print(f'Loading proteome from {FASTA}...')
    proteins = filter_standard(parse_fasta(FASTA))
    print(f'  {len(proteins):,} proteins after filtering standard AAs')

    counts = np.zeros((N_AA, N_AA), dtype=np.float64)

    for seq in proteins.values():
        for a, b in zip(seq[:-1], seq[1:]):
            counts[aa_to_idx[a], aa_to_idx[b]] += 1

    total_pairs = counts.sum()
    print(f'  {total_pairs:,.0f} consecutive AA pairs counted')

    counts += PSEUDOCOUNT
    row_sums = counts.sum(axis=1, keepdims=True)
    bigram_freq = counts / row_sums           # P(AA_j | AA_i)
    log_bigram  = np.log(bigram_freq)         # log-space for CRF init

    np.save(OUT_PATH, {
        'classes':     AA_ORDER,
        'counts':      counts - PSEUDOCOUNT,  # raw counts without pseudocount
        'bigram_freq': bigram_freq,
        'log_bigram':  log_bigram,
        'pseudocount': PSEUDOCOUNT,
    })
    print(f'Saved → {OUT_PATH}')

    print('\nTop-5 most common transitions:')
    flat = [(counts[i, j] - PSEUDOCOUNT, AA_ORDER[i], AA_ORDER[j])
            for i in range(N_AA) for j in range(N_AA)]
    for cnt, a, b in sorted(flat, reverse=True)[:5]:
        print(f'  {a}→{b}: {cnt:,.0f}')

    print('\nDiagonal (self-transition) log-probs:')
    for i, aa in enumerate(AA_ORDER):
        print(f'  {aa}: {log_bigram[i, i]:.3f}')


if __name__ == '__main__':
    main()
