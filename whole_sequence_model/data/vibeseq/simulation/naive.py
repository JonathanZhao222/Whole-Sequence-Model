import numpy as np
from pathlib import Path


def load_confusion_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True).item()
    return data["classes"], data["cm_frac"].astype(np.float64)


def simulate_vibeTag(
    sequence: str,
    classes: list[str],
    cm_frac: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Simulate VibeTags output for a single protein sequence.

    For each position i with true amino acid X:
      1. Sample detected amino acid Y ~ cm_frac[X_idx, :]
      2. Output vector for position i = cm_frac[Y_idx, :]

    Returns float32 array of shape (L, 20).
    """
    if rng is None:
        rng = np.random.default_rng()

    aa_to_idx = {aa: i for i, aa in enumerate(classes)}
    true_indices = np.array([aa_to_idx[aa] for aa in sequence], dtype=np.int32)

    cdf = cm_frac.cumsum(axis=1)
    u = rng.random(len(sequence))
    detected_indices = np.clip(
        (u[:, None] > cdf[true_indices]).sum(axis=1),
        0,
        len(classes) - 1,
    )

    return cm_frac[detected_indices].astype(np.float32)
