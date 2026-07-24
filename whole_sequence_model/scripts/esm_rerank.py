#!/usr/bin/env python3
"""
ESM2 re-ranking of CNN+CRF beam search candidates.

Samples peptides with >= MIN_ERRORS argmax errors so the beam actually has
something to resolve. Treats not-found cases as rank K+1.

score = CNN_score + β × ESM_pseudo_log_likelihood  (at variable positions only)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM

from models.properties import AA_ORDER
from models.conv_denoiser import ConvDenoiser
from models.crf import FixedCRF, build_T_from_asset
from models.unified_dataset import apply_mild_noise

ROOT        = Path(__file__).parent.parent
EVAL_H5     = ROOT / 'data'   / 'synthetic' / 'eval_tryptic_seed42.h5'
BIGRAM_PATH = ROOT / 'assets' / 'aa_bigram_transitions.npy'
CKPT        = ROOT / 'models' / 'checkpoints' / 'unified_cnn_bert_decay.pt'
OUT_PNG     = ROOT.parent / 'esm_rerank_results.png'

COND       = 'decay_severe'
K          = 100
MILD_EPS   = 0.02
N_EVAL     = 500
MIN_ERRORS = 4    # pre-filter: only peptides where argmax makes >= this many errors
SEED       = 42
BETA_GRID  = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}

# ── Load models ───────────────────────────────────────────────────────────────
print('Loading CNN + CRF ...')
T   = build_T_from_asset(BIGRAM_PATH)
crf = FixedCRF(T, device='cpu')
cnn = ConvDenoiser(d=128, dropout=0.1)
cnn.load_state_dict(torch.load(CKPT, map_location='cpu'))
cnn.eval()

print('Loading ESM2 (35M) ...')
esm_name  = 'facebook/esm2_t12_35M_UR50D'
tok       = AutoTokenizer.from_pretrained(esm_name)
esm       = EsmForMaskedLM.from_pretrained(esm_name).eval()
aa_to_tok = {aa: tok.convert_tokens_to_ids(aa) for aa in AA_ORDER}
print('Both models ready.\n')

# ── Collect eval peptides ─────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
with h5py.File(EVAL_H5, 'r') as f:
    all_keys = list(f.keys())
rng.shuffle(all_keys)

records = []

with h5py.File(EVAL_H5, 'r') as f:
    for key in tqdm(all_keys, desc='beam+ESM', unit='pep'):
        if len(records) >= N_EVAL:
            break

        seq    = f[key]['sequence'][()].decode()
        obs    = f[key][COND][()].astype(np.float32)
        labels = [aa_to_idx[aa] for aa in seq]

        # Pre-filter: skip easy peptides
        n_errors = int((obs.argmax(-1) != np.array(labels)).sum())
        if n_errors < MIN_ERRORS:
            continue

        # Beam search
        obs_s = apply_mild_noise(obs, eps=MILD_EPS)
        with torch.no_grad():
            em = cnn(torch.from_numpy(obs_s).unsqueeze(0))[0]
        beam = crf.beam_search(em, k=K)

        true_rank = next(
            (r for r, (_, s) in enumerate(beam, 1) if s == labels), None
        )

        # Variable positions: anywhere any beam candidate differs from truth
        var_pos = {i for _, s in beam for i in range(len(seq)) if s[i] != labels[i]}
        if not var_pos:
            continue

        # ESM: one forward pass with all variable positions masked
        ids = tok(seq, return_tensors='pt').input_ids.clone()
        for p in var_pos:
            ids[0, p + 1] = tok.mask_token_id

        with torch.no_grad():
            lp = F.log_softmax(esm(ids).logits, dim=-1)[0]

        cnn_scores = np.array([sc for sc, _ in beam])
        esm_scores = np.array([
            sum(lp[p + 1, aa_to_tok[AA_ORDER[s[p]]]].item() for p in var_pos)
            for _, s in beam
        ])

        records.append({
            'cnn':       cnn_scores,
            'esm':       esm_scores,
            'true_rank': true_rank,          # None if not in beam
            'true_idx':  (true_rank - 1) if true_rank is not None else None,
            'n_var':     len(var_pos),
            'n_errors':  n_errors,
        })

n_found    = sum(1 for r in records if r['true_rank'] is not None)
n_notfound = len(records) - n_found
print(f'\nCollected {len(records)} peptides  '
      f'(found in top-{K}: {n_found}  |  not found: {n_notfound})')
print(f'True rank = 1: {sum(1 for r in records if r["true_rank"] == 1)} '
      f'({sum(1 for r in records if r["true_rank"] == 1)/len(records):.1%})\n')

# ── Grid search over β ────────────────────────────────────────────────────────
# Not-found cases are treated as rank K+1 in the mean (they can't be re-ranked)
print(f"{'β':>7}  {'mean rank*':>11}  {'median':>8}  "
      f"{'top-1':>7}  {'top-5':>7}  {'top-10':>7}  (* not-found = {K+1})")

beta_to_ranks = {}
for beta in BETA_GRID:
    new_ranks = []
    for r in records:
        if r['true_idx'] is None:
            new_ranks.append(K + 1)
            continue
        combined = r['cnn'] + beta * r['esm']
        order    = np.argsort(-combined)
        new_rank = int(np.where(order == r['true_idx'])[0][0]) + 1
        new_ranks.append(new_rank)

    arr = np.array(new_ranks)
    beta_to_ranks[beta] = arr
    print(f"{beta:>7.2f}  {arr.mean():>11.2f}  {np.median(arr):>8.1f}  "
          f"{(arr==1).mean():>7.1%}  {(arr<=5).mean():>7.1%}  {(arr<=10).mean():>7.1%}")

best_beta = min(BETA_GRID, key=lambda b: beta_to_ranks[b].mean())
baseline  = beta_to_ranks[0].mean()
best_mean = beta_to_ranks[best_beta].mean()

print(f'\nBaseline (β=0):  mean rank = {baseline:.2f}')
print(f'Best  β={best_beta}:     mean rank = {best_mean:.2f}  '
      f'(Δ = {baseline - best_mean:+.2f})')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

mean_ranks = [beta_to_ranks[b].mean() for b in BETA_GRID]
axes[0].plot(BETA_GRID, mean_ranks, 'o-', color='#4e9af1', lw=2, ms=6)
axes[0].axhline(baseline, color='#888', lw=1.5, linestyle='--',
                label=f'Baseline β=0  (mean {baseline:.1f})')
axes[0].axvline(best_beta, color='#e05c5c', lw=1.5, linestyle=':',
                label=f'Best β={best_beta}  (mean {best_mean:.1f})')
axes[0].set_xlabel('β  (ESM weight)')
axes[0].set_ylabel('Mean ground-truth rank  (not-found = 101)')
axes[0].set_title(f'Re-rank performance vs β\n'
                  f'({len(records)} peptides ≥{MIN_ERRORS} errors, decay_severe, k={K})')
axes[0].legend(fontsize=9)
axes[0].set_xscale('symlog', linthresh=0.1)

# Rank distribution — only for peptides found in beam (comparable before/after)
found_before = [r['true_rank'] for r in records if r['true_rank'] is not None]
found_after  = [
    beta_to_ranks[best_beta][i]
    for i, r in enumerate(records) if r['true_rank'] is not None
]
bins = np.arange(0.5, K + 1.5, 2)
axes[1].hist(found_before, bins=bins, alpha=0.55, color='#888',
             density=True, label=f'CNN + CRF only  (β=0)')
axes[1].hist(found_after,  bins=bins, alpha=0.55, color='#4e9af1',
             density=True, label=f'+ ESM re-rank  (β={best_beta})')
axes[1].set_xlabel('Ground-truth rank in beam  (found-only)')
axes[1].set_ylabel('Density')
axes[1].set_title(f'Rank distribution  (n={len(found_before)} found in top-{K})')
axes[1].legend(fontsize=9)
axes[1].set_xlim(0.5, K + 0.5)

fig.suptitle(
    f'ESM2 re-ranking · decay_severe · ≥{MIN_ERRORS} argmax errors · k={K}',
    fontsize=12,
)
fig.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
print(f'\nFigure saved → {OUT_PNG}')
