#!/usr/bin/env python3
"""Generate synthetic VibeTags data from the human proteome using the naive model."""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from vibeseq.data import parse_fasta, filter_standard
from vibeseq.simulation import load_confusion_matrix, simulate_vibeTag

FASTA = Path(__file__).parent.parent / "data" / "raw" / "UP000005640_9606.fasta.gz"
CM_PATH = Path(__file__).parent.parent / "assets" / "rf_cm.npy"
OUT_DIR = Path(__file__).parent.parent / "data" / "synthetic"


def main(args):
    rng = np.random.default_rng(args.seed)
    classes, cm_frac = load_confusion_matrix(CM_PATH)

    print(f"Loading proteome from {FASTA}...")
    proteins = filter_standard(parse_fasta(FASTA))
    print(f"  {len(proteins):,} proteins after filtering")

    if args.n_proteins is not None:
        protein_ids = list(proteins)[:args.n_proteins]
        proteins = {pid: proteins[pid] for pid in protein_ids}
        print(f"  Using subset of {len(proteins):,} proteins")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"n{len(proteins)}_seed{args.seed}" if args.n_proteins else f"full_seed{args.seed}"
    out_path = OUT_DIR / f"naive_{tag}.h5"

    with h5py.File(out_path, "w") as f:
        f.attrs["model"] = "naive"
        f.attrs["classes"] = classes
        f.attrs["seed"] = args.seed

        for protein_id, sequence in tqdm(proteins.items(), desc="Simulating"):
            vibeTag_output = simulate_vibeTag(sequence, classes, cm_frac, rng)
            grp = f.create_group(protein_id)
            grp.create_dataset("sequence", data=np.bytes_(sequence))
            grp.create_dataset("vibeTag_output", data=vibeTag_output, compression="gzip")

    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_proteins", type=int, default=None,
                        help="Limit to first N proteins (default: all)")
    main(parser.parse_args())
