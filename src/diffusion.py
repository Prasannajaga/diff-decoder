from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    num_steps: int = 64
    mask_token_id: int = 257
    pad_token_id: int = 256


class DiscreteMaskDiffusion:
    """
    Simple discrete diffusion process for text using token masking.

    Forward process q(x_t | x_0): each token is independently replaced with a MASK token
    with probability p(t), where p(t) increases linearly with t.

    Reverse model p_theta(x_0 | x_t, t): a Transformer predicts clean tokens from x_t and t.
    """

    def __init__(self, cfg: DiffusionConfig) -> None:
        self.cfg = cfg

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        return torch.randint(1, self.cfg.num_steps + 1, (batch_size,), device=device, generator=generator)

    def mask_probability(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps.float() / float(self.cfg.num_steps)

    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x0: [batch, seq]
        probs = self.mask_probability(timesteps).unsqueeze(1)
        rand = torch.rand(x0.shape, device=x0.device, dtype=torch.float32, generator=generator)
        to_mask = rand < probs

        if attention_mask is not None:
            to_mask = to_mask & attention_mask.bool()

        # Never mask PAD tokens if present.
        to_mask = to_mask & (x0 != self.cfg.pad_token_id)

        x_t = x0.clone()
        x_t[to_mask] = self.cfg.mask_token_id
        return x_t, to_mask

    def training_loss(
        self,
        model,
        x0: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = x0.size(0)
        t = timesteps if timesteps is not None else self.sample_timesteps(batch_size, x0.device, generator=generator)
        # this will return masked input and the positions that were masked
        x_t, masked_positions = self.q_sample(x0, t, attention_mask=attention_mask, generator=generator)

        logits = model(x_t, t, attention_mask=attention_mask, causal=False)
        vocab_size = logits.size(-1)

        if attention_mask is None:
            valid = torch.ones_like(masked_positions, dtype=torch.bool)
        else:
            valid = attention_mask.bool()

        loss_positions = masked_positions & valid

        # Guard against rare empty-mask edge cases on tiny sequences.
        if not loss_positions.any():
            loss_positions = valid

        loss = F.cross_entropy(
            logits[loss_positions].reshape(-1, vocab_size),
            x0[loss_positions].reshape(-1),
        )

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = (pred[loss_positions] == x0[loss_positions]).float().mean()
            mask_ratio = masked_positions.float().mean()

        return {
            "loss": loss,
            "acc": acc,
            "mask_ratio": mask_ratio,
        }

    @torch.no_grad()
    def sample(
        self,
        model,
        prefix_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        stream_callback: Optional[Callable[[dict], None]] = None,
    ) -> torch.Tensor:
        """
        Iteratively unmask target positions from high to low noise.

        prefix_ids: [batch, prefix_len]
        returns: generated token ids [batch, max_new_tokens]
        """
        if max_new_tokens <= 0:
            return prefix_ids.new_zeros((prefix_ids.size(0), 0))

        batch_size, prefix_len = prefix_ids.shape
        total_len = prefix_len + max_new_tokens

        x = torch.full(
            (batch_size, total_len),
            fill_value=self.cfg.mask_token_id,
            device=prefix_ids.device,
            dtype=prefix_ids.dtype,
        )
        x[:, :prefix_len] = prefix_ids

        known = torch.zeros((batch_size, total_len), device=prefix_ids.device, dtype=torch.bool)
        known[:, :prefix_len] = True
        gen_slice = slice(prefix_len, None)
        neg_inf = torch.finfo(torch.float32).min

        def _emit(phase: str, step_value: int) -> None:
            if stream_callback is None:
                return
            stream_callback(
                {
                    "phase": phase,
                    "step": step_value,
                    "generated_ids": x[:, gen_slice].detach().clone(),
                    "known_mask": known[:, gen_slice].detach().clone(),
                }
            )

        def _target_logits(curr_x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            logits = model(curr_x, t, attention_mask=None, causal=False)[:, gen_slice, :]
            if 0 <= self.cfg.mask_token_id < logits.size(-1):
                logits = logits.clone()
                logits[..., self.cfg.mask_token_id] = torch.finfo(logits.dtype).min
            return logits

        _emit("init", self.cfg.num_steps)

        for step in range(self.cfg.num_steps, 0, -1):
            t = torch.full((batch_size,), step, device=prefix_ids.device, dtype=torch.long)
            target_logits = _target_logits(x, t)
            probs = F.softmax(target_logits.float() / max(temperature, 1e-8), dim=-1)
            conf = probs.amax(dim=-1)

            if temperature <= 0:
                pred_ids = probs.argmax(dim=-1)
            else:
                pred_ids = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).reshape(batch_size, -1)

            gen_known = known[:, gen_slice]
            gen_unknown = ~gen_known

            if not gen_unknown.any():
                break

            # Reveal a shrinking fraction each step (MaskGIT-style schedule), vectorized.
            unknown_counts = gen_unknown.sum(dim=-1)
            reveal_counts = torch.clamp(torch.ceil(unknown_counts.float() / float(step)).to(torch.long), min=1)
            reveal_counts = torch.minimum(reveal_counts, unknown_counts)

            conf_masked = conf.masked_fill(~gen_unknown, neg_inf)
            sorted_idx = conf_masked.argsort(dim=-1, descending=True)
            ranks = torch.empty_like(sorted_idx)
            rank_values = torch.arange(sorted_idx.size(1), device=sorted_idx.device).unsqueeze(0).expand_as(sorted_idx)
            ranks.scatter_(1, sorted_idx, rank_values)
            reveal_mask = gen_unknown & (ranks < reveal_counts.unsqueeze(1))

            x_gen = x[:, gen_slice]
            x_gen[reveal_mask] = pred_ids[reveal_mask]
            known[:, gen_slice] = known[:, gen_slice] | reveal_mask

            _emit("denoise", step)

        # Fill any leftover masked positions with final greedy decode.
        if (~known[:, gen_slice]).any():
            t = torch.ones((batch_size,), device=prefix_ids.device, dtype=torch.long)
            target_logits = _target_logits(x, t)
            final_pred = target_logits.argmax(dim=-1)
            remaining = ~known[:, gen_slice]
            x_gen = x[:, gen_slice]
            x_gen[remaining] = final_pred[remaining]
            known[:, gen_slice] = True

        _emit("final", 0)
        return x[:, gen_slice]
