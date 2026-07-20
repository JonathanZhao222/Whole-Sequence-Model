#!/usr/bin/env python3
"""Download the UniProt human reference proteome (UP000005640) to data/raw/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vibeseq.data import download_proteome, HUMAN_PROTEOME_URL

DEST = Path(__file__).parent.parent / "data" / "raw" / "UP000005640_9606.fasta.gz"


def main():
    if DEST.exists():
        print(f"Already exists: {DEST}")
        return
    print(f"Downloading human proteome → {DEST}")
    download_proteome(HUMAN_PROTEOME_URL, DEST)
    print("Done.")


if __name__ == "__main__":
    main()
