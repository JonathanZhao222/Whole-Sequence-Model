import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.properties import PROPERTY_TABLE


class ResBlock(nn.Module):
    """
    Dilated residual conv block operating on (B, d, L).

    Two Conv1d(k=3) layers with the same dilation, GroupNorm(1, d) — equivalent
    to LayerNorm over the channel dimension — and a residual skip connection.
    Padding = dilation preserves sequence length for k=3.
    """

    def __init__(self, d, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(d, d, kernel_size=3, dilation=dilation, padding=dilation)
        self.norm1 = nn.GroupNorm(1, d)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, dilation=dilation, padding=dilation)
        self.norm2 = nn.GroupNorm(1, d)

    def forward(self, x):           # x: (B, d, L)
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(h + x)        # residual skip


class ConvDenoiser(nn.Module):
    """
    Dilated residual 1D conv denoiser.

    Receptive field with dilations [1, 2, 4, 8, 16, 32] and k=3: 127 positions.

    Per-position input features (27-dim):
        p       (20)  noisy probability vector
        e_norm  ( 7)  z-score normalised expected physicochemical properties

    Forward pass:
        1. Compute and normalise physicochemical features
        2. Project input to d channels
        3. Transpose to (B, d, L) for Conv1d
        4. Pass through 6 dilated residual blocks
        5. Transpose back to (B, L, d)
        6. Project to 20 output logits
    """

    DILATIONS = [1, 2, 4, 8, 16, 32]

    def __init__(self, d=128, dropout=0.1):
        super().__init__()
        k = PROPERTY_TABLE.shape[1]

        self.register_buffer('prop_table', torch.from_numpy(PROPERTY_TABLE))  # (20, k)
        self.register_buffer('prop_mean', torch.zeros(k))
        self.register_buffer('prop_std',  torch.ones(k))

        self.input_proj = nn.Sequential(
            nn.Linear(20 + k, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            ResBlock(d, dil) for dil in self.DILATIONS
        ])

        self.dropout     = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d, 20)

    @torch.no_grad()
    def fit_stats(self, loader, device='cpu'):
        """Compute and store z-score statistics from a DataLoader."""
        all_props = []
        self.eval()
        for probs, _ in loader:
            probs = probs.to(device)
            e = probs @ self.prop_table
            all_props.append(e.reshape(-1, e.shape[-1]).cpu())
        all_props = torch.cat(all_props, dim=0)
        self.prop_mean.copy_(all_props.mean(dim=0))
        self.prop_std.copy_(all_props.std(dim=0).clamp(min=1e-6))

    def forward(self, p):                               # p: (B, L, 20)
        e      = p @ self.prop_table                    # (B, L, k)
        e_norm = (e - self.prop_mean) / self.prop_std   # (B, L, k)
        x      = torch.cat([p, e_norm], dim=-1)         # (B, L, 27)

        x = self.input_proj(x)                          # (B, L, d)
        x = x.transpose(1, 2)                           # (B, d, L)

        for block in self.blocks:
            x = block(x)

        x = self.dropout(x)
        x = x.transpose(1, 2)                           # (B, L, d)
        return self.output_proj(x)                      # (B, L, 20)
