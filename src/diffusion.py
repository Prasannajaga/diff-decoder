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

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(1, self.cfg.num_steps + 1, (batch_size,), device=device)

    def mask_probability(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps.float() / float(self.cfg.num_steps)

    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x0: [batch, seq]
        probs = self.mask_probability(timesteps).unsqueeze(1)
        rand = torch.rand_like(x0, dtype=torch.float32)
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
    ) -> dict[str, torch.Tensor]:
        batch_size = x0.size(0)
        t = self.sample_timesteps(batch_size, x0.device)
        x_t, masked_positions = self.q_sample(x0, t, attention_mask=attention_mask)

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

        if stream_callback is not None:
            stream_callback(
                {
                    "phase": "init",
                    "step": self.cfg.num_steps,
                    "generated_ids": x[:, prefix_len:].detach().clone(),
                    "known_mask": known[:, prefix_len:].detach().clone(),
                }
            )

        for step in range(self.cfg.num_steps, 0, -1):
            t = torch.full((batch_size,), step, device=prefix_ids.device, dtype=torch.long)
            logits = model(x, t, attention_mask=None, causal=False)
            target_logits = logits[:, prefix_len:, :]
            if 0 <= self.cfg.mask_token_id < target_logits.size(-1):
                target_logits = target_logits.clone()
                target_logits[..., self.cfg.mask_token_id] = torch.finfo(target_logits.dtype).min

            if temperature <= 0:
                pred_ids = target_logits.argmax(dim=-1)
                conf = F.softmax(target_logits.float(), dim=-1).amax(dim=-1)
            else:
                probs = F.softmax(target_logits.float() / temperature, dim=-1)
                conf, greedy = probs.max(dim=-1)
                draws = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, -1)
                pred_ids = draws
                # Confidence is based on greedy prob, while token can still be sampled.
                _ = greedy

            gen_known = known[:, prefix_len:]
            gen_unknown = ~gen_known

            if not gen_unknown.any():
                break

            for b in range(batch_size):
                unknown_idx = torch.nonzero(gen_unknown[b], as_tuple=False).squeeze(-1)
                if unknown_idx.numel() == 0:
                    continue

                # Reveal a shrinking fraction each step (MaskGIT-style schedule).
                reveal_count = max(1, math.ceil(unknown_idx.numel() / step))
                scores = conf[b, unknown_idx]
                top_local = torch.topk(scores, k=min(reveal_count, unknown_idx.numel())).indices
                reveal_idx = unknown_idx[top_local]

                x[b, prefix_len + reveal_idx] = pred_ids[b, reveal_idx]
                known[b, prefix_len + reveal_idx] = True

            if stream_callback is not None:
                stream_callback(
                    {
                        "phase": "denoise",
                        "step": step,
                        "generated_ids": x[:, prefix_len:].detach().clone(),
                        "known_mask": known[:, prefix_len:].detach().clone(),
                    }
                )

        # Fill any leftover masked positions with final greedy decode.
        if (~known[:, prefix_len:]).any():
            t = torch.ones((batch_size,), device=prefix_ids.device, dtype=torch.long)
            logits = model(x, t, attention_mask=None, causal=False)
            target_logits = logits[:, prefix_len:, :]
            if 0 <= self.cfg.mask_token_id < target_logits.size(-1):
                target_logits = target_logits.clone()
                target_logits[..., self.cfg.mask_token_id] = torch.finfo(target_logits.dtype).min
            final_pred = target_logits.argmax(dim=-1)
            remaining = ~known[:, prefix_len:]
            x[:, prefix_len:][remaining] = final_pred[remaining]
            known[:, prefix_len:][remaining] = True

        if stream_callback is not None:
            stream_callback(
                {
                    "phase": "final",
                    "step": 0,
                    "generated_ids": x[:, prefix_len:].detach().clone(),
                    "known_mask": known[:, prefix_len:].detach().clone(),
                }
            )

        return x[:, prefix_len:]
