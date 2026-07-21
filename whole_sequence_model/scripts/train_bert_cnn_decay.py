#!/usr/bin/env python3
"""
Train a BERT-CNN (ConvDenoiser) on decay conditions only.

Training data  : tryptic peptides (max 50 aa), sampling uniformly from
                 [decay_mild, decay_moderate, decay_strong, decay_severe].
Eval data      : eval_tryptic_seed42.h5 — all 13 conditions.
Checkpoint     : models/checkpoints/unified_cnn_bert_decay.pt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import h5py
from tqdm import tqdm

from data.vibeseq.simulation.naive import load_confusion_matrix
from models.properties import AA_ORDER
from models.conv_denoiser import ConvDenoiser
from models.unified_dataset import (
    PERTURBATION_CONDITIONS, DECAY_PARAMS,
    make_train_loader, make_eval_loader, apply_bert_mask, trypsin_digest,
)

SEED        = 42
EPOCHS      = 30
BATCH       = 32
LR          = 1e-3
MAX_PEP     = 50
MILD_EPS    = 0.02
MASK_RATE   = 0.15
MAX_PEPTIDES = 50_000   # random sample to keep CPU training tractable
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

SRC_H5    = Path(__file__).parent.parent / 'data' / 'synthetic' / 'naive_full_seed42.h5'
EVAL_H5   = Path(__file__).parent.parent / 'data' / 'synthetic' / 'eval_tryptic_seed42.h5'
CKPT_PATH = Path(__file__).parent.parent / 'models' / 'checkpoints' / 'unified_cnn_bert_decay.pt'
CM_PATH   = Path(__file__).parent.parent / 'assets' / 'rf_cm.npy'

DECAY_CONDITIONS = list(DECAY_PARAMS.keys())  # decay_mild/moderate/strong/severe

aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}


def run_epoch(model, loader, optimizer=None, device='cpu', mask_rate=0.0):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = total_correct = total_tokens = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for probs, labels in loader:
            probs, labels = probs.to(device), labels.to(device)
            if training and mask_rate > 0:
                probs, _ = apply_bert_mask(probs, mask_rate)
            logits = model(probs)
            loss = F.cross_entropy(
                logits.reshape(-1, 20), labels.reshape(-1), ignore_index=-1)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            mask = labels != -1
            total_correct += (logits.argmax(-1)[mask] == labels[mask]).sum().item()
            total_tokens  += mask.sum().item()
            total_loss    += loss.item() * mask.sum().item()
    return total_loss / total_tokens, total_correct / total_tokens


def main():
    print(f'Device:     {DEVICE}')
    print(f'Conditions: {DECAY_CONDITIONS}')
    print(f'Mask rate:  {MASK_RATE:.0%}')

    classes, cm_frac = load_confusion_matrix(CM_PATH)

    # Load sequences and apply trypsin digestion
    with h5py.File(SRC_H5, 'r') as f:
        all_pids  = list(f.keys())
        sequences = {pid: f[pid]['sequence'][()].decode() for pid in all_pids}

    rng_split = np.random.default_rng(SEED)
    pids      = rng_split.permutation(all_pids).tolist()
    train_pids = pids[:int(len(pids) * 0.8)]

    # Build tryptic peptide list for training, then subsample
    train_peptides = []
    for pid in train_pids:
        for i, pep in enumerate(trypsin_digest(sequences[pid], max_len=MAX_PEP)):
            train_peptides.append((f'{pid}_p{i}', pep))

    rng_sample = np.random.default_rng(SEED + 1)
    if len(train_peptides) > MAX_PEPTIDES:
        idx = rng_sample.choice(len(train_peptides), size=MAX_PEPTIDES, replace=False)
        train_peptides = [train_peptides[i] for i in idx]

    print(f'Train peptides: {len(train_peptides):,}  (capped at {MAX_PEPTIDES:,})')

    train_loader = make_train_loader(
        train_peptides, classes, cm_frac, aa_to_idx,
        batch_size=BATCH, num_workers=4, mild_eps=MILD_EPS,
        conditions=DECAY_CONDITIONS,
    )
    # Sample a fixed 10k-peptide eval subset (fast startup, still representative)
    MAX_EVAL = 10_000
    with h5py.File(EVAL_H5, 'r') as f:
        all_eval_keys = list(f.keys())
    rng_eval = np.random.default_rng(SEED + 2)
    eval_keys = rng_eval.choice(all_eval_keys, size=min(MAX_EVAL, len(all_eval_keys)),
                                replace=False).tolist()

    eval_loaders = {
        cond: make_eval_loader(EVAL_H5, cond, aa_to_idx,
                               batch_size=64, mild_eps=MILD_EPS, key_list=eval_keys)
        for cond in PERTURBATION_CONDITIONS
    }
    print(f'Eval peptides:  {len(eval_keys):,} per condition')
    print(f'Train batches/epoch: {len(train_loader):,}')

    model = ConvDenoiser(d=128, dropout=0.1).to(DEVICE)
    model.fit_stats(train_loader, device=DEVICE)
    print(f'Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_avg = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, DEVICE,
                                    mask_rate=MASK_RATE)
        scheduler.step()

        if epoch % 5 == 0 or epoch == EPOCHS:
            cond_accs = {c: run_epoch(model, ldr, device=DEVICE)[1]
                         for c, ldr in eval_loaders.items()}
            avg = np.mean(list(cond_accs.values()))
            if avg > best_avg:
                best_avg = avg
                torch.save(model.state_dict(), CKPT_PATH)
            decay_avg = np.mean([cond_accs[c] for c in DECAY_CONDITIONS])
            print(f'Epoch {epoch:3d}  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3%}  '
                  f'decay_avg={decay_avg:.3%}  '
                  f'mild={cond_accs["decay_mild"]:.3%}  '
                  f'severe={cond_accs["decay_severe"]:.3%}  '
                  f'overall_avg={avg:.3%}')
        else:
            print(f'Epoch {epoch:3d}  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3%}')

    print(f'\nBest overall avg val acc: {best_avg:.3%}  →  {CKPT_PATH}')


if __name__ == '__main__':
    main()
