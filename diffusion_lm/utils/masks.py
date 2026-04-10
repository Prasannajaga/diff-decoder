"""Attention mask utilities for diffusion language models.

Provides two masking modes:
 • Bidirectional   — full attention (MDLM)
 • Block-causal    — causal across blocks, bidirectional within (BD3LM)
"""

from __future__ import annotations

import torch


def create_bidirectional_mask(seq_len: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Full-attention mask: every token attends to every other token.

    Returns:
        Bool tensor of shape (1, 1, seq_len, seq_len), True = attend.
    """
    return torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=device)


def create_block_causal_mask(
    seq_len: int,
    block_size: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Block-causal mask for BD3LM.

    Within each block  → bidirectional (full attention).
    Across blocks      → strictly causal (attend only to *previous* blocks).

    Args:
        seq_len:    Total sequence length.
        block_size: Size of each diffusion block.
        device:     Target device.

    Returns:
        Bool tensor of shape (1, 1, seq_len, seq_len), True = attend.
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    num_blocks = (seq_len + block_size - 1) // block_size

    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, seq_len)

        # Current block: bidirectional (attend to self)
        mask[start:end, start:end] = True

        # Attend to all *previous* blocks (causal)
        mask[start:end, :start] = True

    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)
