"""Diffusion samplers for MDLM and BD3LM text generation.

MDLM sampling:
    Start from all-MASK → iteratively unmask by model confidence.

BD3LM sampling:
    Generate block-by-block, each block denoised via diffusion
    conditioned on previous clean blocks.

Models are accepted as any callable with signature:
    (x_t: Tensor, t: Tensor, attention_mask: Tensor) → logits: Tensor
This is compatible with both the raw DiffusionTransformer.forward()
and MDLM/BD3LM.predict().
"""

from __future__ import annotations

from typing import Callable, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_lm.constants import MASK_TOKEN_ID
from diffusion_lm.utils.masks import create_bidirectional_mask, create_block_causal_mask


# ── Helpers ──────────────────────────────────────────────────────────

def _sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    greedy: bool = False,
) -> torch.Tensor:
    """Sample token ids from logits.

    Args:
        logits:      (B, L, V)
        temperature: Sampling temperature (ignored if greedy).
        greedy:      If True, take argmax.

    Returns:
        Token ids (B, L).
    """
    if greedy:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits / temperature, dim=-1)
    B, L, V = probs.shape
    flat = probs.view(B * L, V)
    sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return sampled.view(B, L)


# ── MDLM Sampler ────────────────────────────────────────────────────

class MDLMSampler:
    """Iterative denoising sampler for MDLM.

    Algorithm:
        1. Start with x = [MASK] * seq_len
        2. For each step k in {num_steps, ..., 1}:
           a. t = k / num_steps
           b. Predict logits via predict_fn(x, t, mask)
           c. Sample candidate tokens from logits
           d. Compute confidence (max prob) for each masked position
           e. Unmask the top-confidence positions
        3. Return fully denoised sequence
    """

    def __init__(
        self,
        predict_fn: Callable,
        num_steps: int = 64,
        temperature: float = 1.0,
        greedy: bool = False,
        mask_token_id: int = MASK_TOKEN_ID,
    ) -> None:
        self.predict_fn = predict_fn
        self.num_steps = num_steps
        self.temperature = temperature
        self.greedy = greedy
        self.mask_token_id = mask_token_id

    @torch.no_grad()
    def sample(
        self,
        seq_len: int,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Generate sequences via iterative denoising.

        Args:
            seq_len:    Length of sequence to generate.
            batch_size: Number of sequences.
            device:     Target device.

        Returns:
            Token ids (batch_size, seq_len).
        """
        x = torch.full(
            (batch_size, seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        attention_mask = create_bidirectional_mask(seq_len, device=device)

        for step in range(self.num_steps, 0, -1):
            t_val = step / self.num_steps
            t = torch.full((batch_size,), t_val, device=device)

            logits = self.predict_fn(x, t, attention_mask)
            candidates = _sample_from_logits(logits, self.temperature, self.greedy)

            probs = F.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values

            is_masked = (x == self.mask_token_id)

            num_masked = is_masked.sum(dim=1, keepdim=True).float()
            fraction = 1.0 / step
            num_to_unmask = (num_masked * fraction).ceil().long()

            confidence = confidence.masked_fill(~is_masked, -float("inf"))

            for b in range(batch_size):
                n = min(num_to_unmask[b].item(), is_masked[b].sum().item())
                if n <= 0:
                    continue
                _, top_idx = confidence[b].topk(n)
                x[b, top_idx] = candidates[b, top_idx]

        return x


# ── BD3LM Sampler ───────────────────────────────────────────────────

class BD3LMSampler:
    """Block-by-block diffusion sampler for BD3LM.

    Algorithm:
        For each block b:
            1. Initialize block positions with [MASK]
            2. Run num_steps denoising steps for this block
            3. Use clean previous blocks as context
            4. Append denoised block
    """

    def __init__(
        self,
        predict_fn: Callable,
        block_size: int = 64,
        num_steps: int = 64,
        temperature: float = 1.0,
        greedy: bool = False,
        mask_token_id: int = MASK_TOKEN_ID,
    ) -> None:
        self.predict_fn = predict_fn
        self.block_size = block_size
        self.num_steps = num_steps
        self.temperature = temperature
        self.greedy = greedy
        self.mask_token_id = mask_token_id

    @torch.no_grad()
    def sample(
        self,
        seq_len: int,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Generate sequences block-by-block.

        Args:
            seq_len:    Total length of sequence to generate.
            batch_size: Number of sequences.
            device:     Target device.

        Returns:
            Token ids (batch_size, seq_len).
        """
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        generated = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

        for b_idx in range(num_blocks):
            this_block_size = min(self.block_size, seq_len - b_idx * self.block_size)

            block = torch.full(
                (batch_size, this_block_size),
                self.mask_token_id,
                dtype=torch.long,
                device=device,
            )

            for step in range(self.num_steps, 0, -1):
                t_val = step / self.num_steps
                t = torch.full((batch_size,), t_val, device=device)

                full_seq = torch.cat([generated, block], dim=1)
                full_len = full_seq.shape[1]

                attention_mask = create_block_causal_mask(
                    full_len, self.block_size, device=device,
                )

                logits = self.predict_fn(full_seq, t, attention_mask)

                block_logits = logits[:, -this_block_size:, :]
                candidates = _sample_from_logits(
                    block_logits, self.temperature, self.greedy,
                )

                probs = F.softmax(block_logits, dim=-1)
                confidence = probs.max(dim=-1).values

                is_masked = (block == self.mask_token_id)

                num_masked = is_masked.sum(dim=1, keepdim=True).float()
                fraction = 1.0 / step
                num_to_unmask = (num_masked * fraction).ceil().long()

                confidence = confidence.masked_fill(~is_masked, -float("inf"))

                for bi in range(batch_size):
                    n = min(num_to_unmask[bi].item(), is_masked[bi].sum().item())
                    if n <= 0:
                        continue
                    _, top_idx = confidence[bi].topk(n)
                    block[bi, top_idx] = candidates[bi, top_idx]

            generated = torch.cat([generated, block], dim=1)

        return generated
