"""BD3LM — Block Discrete Denoising Diffusion Language Model.

The sequence is split into fixed-size blocks:
    x = (x^1, x^2, ..., x^B)

The model factorises:
    p(x) = ∏_b  p(x^b | x^{<b})

Each block is generated via diffusion conditioned on the *clean*
previous blocks.  The attention mask is block-causal: bidirectional
within the current block, causal across blocks.

Loss:
    L = Σ_b  E_{t,x} [ (1/t) Σ_{i ∈ mask(B_b, t)} -log p_θ(x_i | x^b_t, x^{<b}) ]

Integration:
    forward(input_ids) returns {"loss": scalar} so the existing
    Trainer._compute_loss detects the dict and uses it directly.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_lm.constants import MASK_TOKEN_ID
from diffusion_lm.diffusion.noise import (
    apply_mask_noise,
    get_schedule,
    sample_timesteps,
)
from diffusion_lm.models.transformer import DiffusionTransformer
from diffusion_lm.utils.masks import create_block_causal_mask


class BD3LM(nn.Module):
    """Block Diffusion Language Model.

    Wraps a DiffusionTransformer with the BD3LM training objective:
    block-causal attention, per-block masking noise, and weighted
    cross-entropy over masked positions within each block.

    Compatible with the project's Trainer: ``forward(input_ids)``
    returns ``{"loss": scalar_loss}``.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        block_size: int = 64,
        mask_token_id: int = MASK_TOKEN_ID,
        schedule: str = "linear",
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.schedule_fn = get_schedule(schedule)

        self.transformer = DiffusionTransformer(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

    def _prepare_block_input(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        block_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a training input where previous blocks are clean and the
        current block is noised.

        Args:
            x_0:       Clean token ids, (B, L).
            t:         Per-sample timesteps, (B,).
            block_idx: Index of the current block to noise.

        Returns:
            x_t:        Mixed clean/noisy token ids, (B, L).
            mask_flags: Bool mask — True only at masked positions in the
                        current block, (B, L).
        """
        B, L = x_0.shape
        start = block_idx * self.block_size
        end = min(start + self.block_size, L)

        # Noise *only* the current block
        block_clean = x_0[:, start:end]
        block_noisy, block_mask = apply_mask_noise(
            block_clean, t, self.mask_token_id, self.schedule_fn,
        )

        # Assemble: previous blocks clean | current block noisy | future blocks clean
        x_t = x_0.clone()
        x_t[:, start:end] = block_noisy

        # Full-sequence mask flags (True only in current block's masked spots)
        mask_flags = torch.zeros(B, L, dtype=torch.bool, device=x_0.device)
        mask_flags[:, start:end] = block_mask

        return x_t, mask_flags

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute BD3LM diffusion loss for a batch.

        This signature is compatible with the project Trainer which calls
        ``model(inputs)`` and checks for a ``"loss"`` key in the output.

        Args:
            input_ids: Clean token ids, (B, L).

        Returns:
            Dict with "loss" (scalar).
        """
        x_0 = input_ids
        B, L = x_0.shape
        t = sample_timesteps(B, device=x_0.device)
        num_blocks = (L + self.block_size - 1) // self.block_size
        attention_mask = create_block_causal_mask(L, self.block_size, device=x_0.device)

        total_loss = torch.tensor(0.0, device=x_0.device)

        for b in range(num_blocks):
            x_t, mask_flags = self._prepare_block_input(x_0, t, b)

            logits, _ = self.transformer(x_t, t, attention_mask=attention_mask)

            log_probs = F.log_softmax(logits, dim=-1)
            target_log_probs = log_probs.gather(2, x_0.unsqueeze(-1)).squeeze(-1)
            nll = -target_log_probs

            nll = nll * mask_flags.float()

            masked_counts = mask_flags.float().sum(dim=1).clamp(min=1.0)
            per_sample_loss = nll.sum(dim=1) / masked_counts
            weighted_loss = per_sample_loss / t.clamp(min=1e-5)

            total_loss = total_loss + weighted_loss.mean()

        loss = total_loss / num_blocks
        return {"loss": loss}

    def predict(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run input through the transformer for sampling.

        Args:
            x_t: Token ids (mix of clean prefix + noisy current block), (B, L).
            t:   Timesteps, (B,).
            attention_mask: Block-causal mask.

        Returns:
            logits: (B, L, V)
        """
        if attention_mask is None:
            attention_mask = create_block_causal_mask(
                x_t.shape[1], self.block_size, device=x_t.device,
            )
        logits, _ = self.transformer(x_t, t, attention_mask=attention_mask)
        return logits
