"""Forward noise process for masked diffusion.

Implements the corruption step:
    q(x_t | x_0) = (1 - t) · x_0  +  t · [MASK]

Tokens in x_0 are independently replaced with [MASK] with probability
derived from a noise schedule evaluated at timestep t.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from diffusion_lm.constants import EPSILON, MIN_TIMESTEP, MAX_TIMESTEP


# ── Noise Schedules ─────────────────────────────────────────────────

def linear_schedule(t: torch.Tensor) -> torch.Tensor:
    """Identity schedule: masking probability = t.

    Args:
        t: Timestep tensor in (0, 1), any shape.

    Returns:
        Masking probability, same shape as t.
    """
    return t


def cosine_schedule(t: torch.Tensor) -> torch.Tensor:
    """Cosine annealing schedule — slower noise at start, faster at end.

    prob(t) = 1 - cos(t · π/2)

    Args:
        t: Timestep tensor in (0, 1).

    Returns:
        Masking probability in [0, 1].
    """
    return 1.0 - torch.cos(t * (math.pi / 2.0))


SCHEDULE_REGISTRY: dict[str, callable] = {
    "linear": linear_schedule,
    "cosine": cosine_schedule,
}


def get_schedule(name: str) -> callable:
    """Look up a noise schedule by name.

    Raises:
        ValueError: If schedule name is not registered.
    """
    if name not in SCHEDULE_REGISTRY:
        raise ValueError(
            f"Unknown schedule '{name}'. "
            f"Available: {sorted(SCHEDULE_REGISTRY)}"
        )
    return SCHEDULE_REGISTRY[name]


# ── Forward Masking ──────────────────────────────────────────────────

def sample_timesteps(batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Sample uniform timesteps in (MIN_TIMESTEP, MAX_TIMESTEP).

    Args:
        batch_size: Number of timesteps to sample.
        device:     Target device.

    Returns:
        Tensor of shape (batch_size,).
    """
    return torch.rand(batch_size, device=device) * (MAX_TIMESTEP - MIN_TIMESTEP) + MIN_TIMESTEP


def apply_mask_noise(
    x_0: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int,
    schedule_fn: callable = linear_schedule,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply forward masking noise to a batch of token sequences.

    Each token in each sequence is independently replaced with
    mask_token_id with probability schedule_fn(t[b]).

    Args:
        x_0:           Clean token ids, shape (B, L).
        t:             Per-sample timesteps, shape (B,).
        mask_token_id: Token id to use as [MASK].
        schedule_fn:   Noise schedule mapping t → mask probability.

    Returns:
        x_t:        Noised token ids, shape (B, L).
        mask_flags: Bool tensor, True where masking was applied, shape (B, L).
    """
    prob = schedule_fn(t)                     # (B,)
    prob = prob.unsqueeze(1)                  # (B, 1)

    noise = torch.rand_like(x_0, dtype=torch.float32)  # (B, L)
    mask_flags = noise < prob                           # (B, L)

    x_t = x_0.clone()
    x_t[mask_flags] = mask_token_id

    return x_t, mask_flags
