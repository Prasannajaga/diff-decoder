"""Decoder-only transformer backbone for diffusion language models.

Components:
 • RotaryEmbedding  — Rotary positional encoding (RoPE)
 • MultiHeadAttention — Grouped query attention with RoPE and optional KV-cache
 • FeedForward       — SwiGLU feed-forward network
 • TransformerBlock  — Pre-norm residual block
 • DiffusionTransformer — Full model with timestep conditioning
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Rotary Positional Embedding ──────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Precomputes sin/cos tables up to max_seq_len and applies rotary
    transformations to query and key tensors.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)              # (S, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)             # (S, D)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for positions [offset, offset + seq_len).

        Args:
            x:      Any tensor whose second dim is the sequence length.
            offset: Starting position (for KV-cache continuation).

        Returns:
            (cos, sin) each of shape (1, seq_len, 1, dim).
        """
        seq_len = x.shape[1]
        end = offset + seq_len

        if end > self.cos_cached.shape[0]:
            self._build_cache(end)

        cos = self.cos_cached[offset:end].unsqueeze(0).unsqueeze(2)  # (1, S, 1, D)
        sin = self.sin_cached[offset:end].unsqueeze(0).unsqueeze(2)
        return cos.to(x.device), sin.to(x.device)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension: [x1, x2] → [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        q, k: (B, S, H, D)
        cos, sin: (1, S, 1, D)

    Returns:
        Rotated (q, k) of same shape.
    """
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# ── KV Cache ─────────────────────────────────────────────────────────

@dataclass
class KVCache:
    """Simple key-value cache for autoregressive / block generation."""

    keys: torch.Tensor    # (B, S_cached, H, D)
    values: torch.Tensor  # (B, S_cached, H, D)

    def update(
        self, new_k: torch.Tensor, new_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new keys/values and return the full sequence."""
        self.keys = torch.cat([self.keys, new_k], dim=1)
        self.values = torch.cat([self.values, new_v], dim=1)
        return self.keys, self.values


# ── Multi-Head Attention ─────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with RoPE and optional KV-cache."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Args:
            x:              (B, S, D)
            cos, sin:       (1, S, 1, head_dim)
            attention_mask: (1, 1, S_q, S_kv) bool — True = attend
            kv_cache:       Optional cache from previous steps.

        Returns:
            output: (B, S, D)
            updated kv_cache or None
        """
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # (B, H, S, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, S_q, S_kv)

        if attention_mask is not None:
            attn = attn.masked_fill(~attention_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)                    # (B, H, S_q, D)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(out), kv_cache


# ── Feed-Forward Network (SwiGLU) ────────────────────────────────────

class FeedForward(nn.Module):
    """SwiGLU feed-forward block.

    FFN(x) = W2 · (SiLU(W_gate · x) ⊙ W_up · x)
    """

    def __init__(self, hidden_size: int, expansion: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner = hidden_size * expansion
        self.gate_proj = nn.Linear(hidden_size, inner, bias=False)
        self.up_proj = nn.Linear(hidden_size, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ── Transformer Block ───────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → Attn → Res → LN → FFN → Res."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        residual = x
        x = self.attn_norm(x)
        x, kv_cache = self.attn(x, cos, sin, attention_mask, kv_cache)
        x = residual + x

        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)

        return x, kv_cache


# ── Timestep Embedding ──────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding → MLP projection.

    Maps scalar t ∈ (0, 1) to a vector of size hidden_size, then
    projects through a small MLP for expressivity.
    """

    def __init__(self, hidden_size: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) scalar timesteps.

        Returns:
            (B, hidden_size) embeddings.
        """
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)        # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, freq_dim)
        return self.mlp(emb)


# ── Full Diffusion Transformer ──────────────────────────────────────

class DiffusionTransformer(nn.Module):
    """Decoder-only transformer conditioned on diffusion timestep.

    Architecture:
        Token Embedding  +  Timestep Embedding (broadcast-added)
        N × TransformerBlock
        LayerNorm
        LM Head  (tied to embedding weights)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.timestep_emb = TimestepEmbedding(hidden_size)
        self.drop = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(hidden_size // num_heads, max_seq_len)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Small normal init for embeddings, Xavier for linear layers."""
        nn.init.normal_(self.token_emb.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[list[KVCache]] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[list[KVCache]]]:
        """
        Args:
            x:              Token ids, (B, S).
            t:              Timesteps, (B,).
            attention_mask: (1, 1, S, S) bool mask.
            kv_caches:      Per-layer KV caches (list of length num_layers).
            offset:         Position offset for RoPE (used with KV-cache).

        Returns:
            logits:    (B, S, vocab_size)
            kv_caches: Updated caches (or None).
        """
        h = self.token_emb(x)                              # (B, S, D)
        t_emb = self.timestep_emb(t)                       # (B, D)
        h = h + t_emb.unsqueeze(1)                         # broadcast add
        h = self.drop(h)

        cos, sin = self.rope(h, offset=offset)

        new_caches: list[KVCache] = []
        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if kv_caches is not None else None
            h, cache_i = block(h, cos, sin, attention_mask, cache_i)
            new_caches.append(cache_i)

        h = self.final_norm(h)
        logits = self.lm_head(h)                           # (B, S, V)

        out_caches = new_caches if kv_caches is not None else None
        return logits, out_caches
