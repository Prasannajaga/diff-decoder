from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionTransformerConfig:
    vocab_size: int
    max_seq_len: int = 256
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    ffn_mult: float = 4.0
    dropout: float = 0.1
    rope_theta: float = 10000.0
    init_std: float = 0.02


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * scale) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [batch, heads, seq, head_dim]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos

    out = torch.empty_like(x)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


class GQAAttention(nn.Module):
    def __init__(self, cfg: DiffusionTransformerConfig) -> None:
        super().__init__()
        if cfg.dim % cfg.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if cfg.n_heads % cfg.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA")

        self.dim = cfg.dim
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.kv_repeats = cfg.n_heads // cfg.n_kv_heads

        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

        self.dropout = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(self.head_dim, theta=cfg.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        q = self.q_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(seqlen, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = k.repeat_interleave(self.kv_repeats, dim=1)
        v = v.repeat_interleave(self.kv_repeats, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # attention_mask: [batch, seq], 1=valid, 0=padding
            pad_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~pad_mask, float("-inf"))

        if causal:
            causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        attn = F.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.out_proj(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: DiffusionTransformerConfig) -> None:
        super().__init__()
        hidden_dim = int(cfg.ffn_mult * cfg.dim)
        self.gate_proj = nn.Linear(cfg.dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(cfg.dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: DiffusionTransformerConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.attn = GQAAttention(cfg)
        self.ffn = SwiGLUFFN(cfg)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask, causal=causal)
        x = x + self.ffn(self.ffn_norm(x))
        return x


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    # timesteps: [batch] integers
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device) / max(half, 1))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class DiffusionTransformer(nn.Module):
    def __init__(self, cfg: DiffusionTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim * 4),
            nn.SiLU(),
            nn.Linear(cfg.dim * 4, cfg.dim),
        )
        self.in_dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Small-std init keeps early logits near zero, which stabilizes initial CE.
        self.apply(self._init_weights)
        scale = 1.0 / math.sqrt(2.0 * self.cfg.n_layers)
        for block in self.blocks:
            block.attn.out_proj.weight.data.mul_(scale)
            block.ffn.down_proj.weight.data.mul_(scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        if seqlen > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seqlen} exceeds max_seq_len {self.cfg.max_seq_len}")

        tok = self.token_embed(input_ids)
        t_emb = sinusoidal_timestep_embedding(timesteps, self.cfg.dim)
        t_emb = self.time_mlp(t_emb).unsqueeze(1)

        x = self.in_dropout(tok + t_emb)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask, causal=causal)

        x = self.final_norm(x)
        return self.lm_head(x)
