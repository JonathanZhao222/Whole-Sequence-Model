import numpy as np
import torch
import torch.nn.functional as F


class FixedCRF:
    """
    Linear-chain CRF with a fixed (non-learned) transition matrix.

    T[i, j] = log-odds score for amino acid j following amino acid i,
    computed as log P(j|i) - log P(j) from proteome bigram statistics.

    Negative entries penalise biologically rare bigrams; near-zero entries
    are neutral. T is never updated during training.
    """

    def __init__(self, T: np.ndarray, device: str = 'cpu'):
        # T: (20, 20) log-odds matrix  (numpy, converted to tensor for GPU use)
        self._T_np = T.astype(np.float32)
        self.T = torch.from_numpy(self._T_np).to(device)

    def to(self, device):
        self.T = self.T.to(device)
        return self

    # ------------------------------------------------------------------
    # Training loss  (uses torch; gradients flow through emissions only)
    # ------------------------------------------------------------------

    def nll(self, emissions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        CRF negative log-likelihood for a single sequence.

        emissions : (L, 20)  raw logits from the CNN
        labels    : (L,)     ground-truth amino acid indices (no padding)
        """
        L = emissions.shape[0]

        # Gold score: Σ emission[t, y_t] + Σ T[y_{t-1}, y_t]
        gold = emissions[torch.arange(L), labels].sum()
        if L > 1:
            gold = gold + self.T[labels[:-1], labels[1:]].sum()

        # Partition function via forward algorithm (log-space)
        log_alpha = emissions[0]                                  # (20,)
        for t in range(1, L):
            # log_alpha[j] = logsumexp_i(log_alpha[i] + T[i,j]) + emissions[t,j]
            log_alpha = (
                torch.logsumexp(log_alpha.unsqueeze(1) + self.T, dim=0)
                + emissions[t]
            )
        log_Z = torch.logsumexp(log_alpha, dim=0)

        return log_Z - gold   # NLL ≥ 0

    def batch_nll(
        self,
        emissions: torch.Tensor,  # (B, Lmax, 20)
        labels:    torch.Tensor,  # (B, Lmax)  padded with -1
    ) -> torch.Tensor:
        """Mean CRF NLL over a batch with variable-length sequences."""
        losses = []
        for b in range(emissions.shape[0]):
            valid = labels[b] != -1
            losses.append(self.nll(emissions[b, valid], labels[b, valid]))
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Decoding  (numpy; no gradient needed)
    # ------------------------------------------------------------------

    def viterbi(self, emissions: torch.Tensor) -> list[int]:
        """Standard Viterbi: single best sequence."""
        em = F.log_softmax(emissions, dim=-1).detach().cpu().numpy()
        T  = self._T_np
        L  = em.shape[0]

        scores      = em[0].copy()                     # (20,)
        backpointer = np.zeros((L, 20), dtype=np.int32)

        for t in range(1, L):
            cand        = scores[:, None] + T           # (20 prev, 20 next)
            best_prev   = cand.argmax(axis=0)           # (20,)
            scores      = cand[best_prev, np.arange(20)] + em[t]
            backpointer[t] = best_prev

        seq = [int(scores.argmax())]
        for t in range(L - 1, 0, -1):
            seq.append(int(backpointer[t][seq[-1]]))
        return list(reversed(seq))

    def beam_search(
        self,
        emissions: torch.Tensor,
        k: int = 500,
    ) -> list[tuple[float, list[int]]]:
        """
        Beam search returning the top-k sequences sorted best-first.

        Uses vectorised numpy operations: O(k × L × 20) per call.
        Returns list of (score, [aa_index, ...]).
        """
        em = F.log_softmax(emissions, dim=-1).detach().cpu().numpy()  # (L, 20)
        T  = self._T_np                                                 # (20, 20)
        L  = em.shape[0]

        # Initialise from position 0  (at most 20 beams)
        b0          = min(k, 20)
        top_init    = np.argsort(-em[0])[:b0]
        beam_scores = em[0, top_init]               # (b0,)
        beam_seqs   = top_init[:, None]             # (b0, 1)

        for t in range(1, L):
            b = len(beam_scores)
            last_aas = beam_seqs[:, -1]             # (b,)
            trans    = T[last_aas]                  # (b, 20)  T[prev, :]
            # new_scores[i, j] = beam_scores[i] + T[last_aas[i], j] + em[t, j]
            new_scores = beam_scores[:, None] + trans + em[t][None, :]  # (b, 20)

            flat = new_scores.ravel()               # (b*20,)
            n_keep = min(k, len(flat))
            top_k  = np.argpartition(-flat, n_keep - 1)[:n_keep]
            top_k  = top_k[np.argsort(-flat[top_k])]

            prev_idx    = top_k // 20
            next_aa     = top_k % 20
            beam_scores = flat[top_k]
            beam_seqs   = np.hstack([beam_seqs[prev_idx], next_aa[:, None]])

        return [(float(beam_scores[i]), beam_seqs[i].tolist())
                for i in range(len(beam_scores))]

    def ground_truth_rank(
        self,
        emissions: torch.Tensor,
        labels:    torch.Tensor,
        k:         int = 500,
    ) -> int | None:
        """
        1-indexed rank of the ground truth in the top-k beam.
        Returns None if ground truth not found within k.
        """
        beam     = self.beam_search(emissions, k=k)
        true_seq = labels.tolist()
        for rank, (_, seq) in enumerate(beam, 1):
            if seq == true_seq:
                return rank
        return None


# ------------------------------------------------------------------
# Helper: build T from the precomputed bigram asset
# ------------------------------------------------------------------

def build_T_from_asset(asset_path) -> np.ndarray:
    """
    Load the proteome bigram asset and convert to log-odds:
        T[i, j] = log P(j|i) - log P(j)

    Negative entries = rarer than background (biological penalties).
    Near-zero = neutral.
    """
    data       = np.load(asset_path, allow_pickle=True).item()
    log_p_j_i  = data['log_bigram']          # (20, 20)  log P(j|i)
    bigram_freq = data['bigram_freq']         # (20, 20)  P(j|i)

    # Marginal P(j) from row-marginals: P(j) = Σ_i P(i) P(j|i)
    # Approximate with column means of bigram_freq (uniform prior over i)
    p_j    = bigram_freq.mean(axis=0)        # (20,)
    log_p_j = np.log(p_j)

    T = log_p_j_i - log_p_j[None, :]        # (20, 20) log-odds
    return T.astype(np.float32)
