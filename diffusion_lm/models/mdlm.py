"""MDLM — Masked Diffusion Language Model.

Training objective:
    L = E_{t, x} [ (1/t) * sum_{i in M_t} -log p_θ(x_i | x_t) ]

where M_t is the set of masked positions at timestep t, and x_t is
obtained by masking tokens in x_0 with probability schedule(t).

The model uses bidirectional attention so every token can attend to
every other token (masked or unmasked).

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
from diffusion_lm.utils.masks import create_bidirectional_mask


class MDLM(nn.Module):
    """Masked Diffusion Language Model.

    Wraps a DiffusionTransformer with the MDLM training objective:
    bidirectional attention, forward masking noise, and weighted
    cross-entropy loss over masked positions.

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
        mask_token_id: int = MASK_TOKEN_ID,
        schedule: str = "linear",
    ) -> None:
        super().__init__()
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

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute MDLM diffusion loss for a batch.

        This signature is compatible with the project Trainer which calls
        ``model(inputs)`` and checks for a ``"loss"`` key in the output.

        Args:
            input_ids: Clean token ids, (B, L).

        Returns:
            Dict with "loss" (scalar) and "logits" (B, L, V).
        """
        x_0 = input_ids
        t = sample_timesteps(x_0.shape[0], device=x_0.device)
        x_t, mask_flags = apply_mask_noise(x_0, t, self.mask_token_id, self.schedule_fn)

        attention_mask = create_bidirectional_mask(x_t.shape[1], device=x_t.device)
        logits, _ = self.transformer(x_t, t, attention_mask=attention_mask)

        # Cross-entropy per token
        log_probs = F.log_softmax(logits, dim=-1)                          # (B, L, V)
        target_log_probs = log_probs.gather(2, x_0.unsqueeze(-1)).squeeze(-1)  # (B, L)
        nll = -target_log_probs                                             # (B, L)

        # Zero out non-masked positions
        nll = nll * mask_flags.float()                                      # (B, L)

        # Per-sample: mean NLL over masked tokens, weighted by 1/t
        masked_counts = mask_flags.float().sum(dim=1).clamp(min=1.0)        # (B,)
        per_sample_loss = nll.sum(dim=1) / masked_counts                    # (B,)
        weighted_loss = per_sample_loss / t.clamp(min=1e-5)                 # (B,)

        loss = weighted_loss.mean()
        return {"loss": loss, "logits": logits}

    def predict(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run noisy input through the transformer for sampling.

        Args:
            x_t: Noisy token ids, (B, L).
            t:   Timesteps, (B,).
            attention_mask: Optional mask override.

        Returns:
            logits: (B, L, V)
        """
        if attention_mask is None:
            attention_mask = create_bidirectional_mask(x_t.shape[1], device=x_t.device)
        logits, _ = self.transformer(x_t, t, attention_mask=attention_mask)
        return logits
