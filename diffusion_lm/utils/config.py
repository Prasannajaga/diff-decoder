"""Application configuration — strongly typed, validated, immutable.

All configurable values originate here.  Configuration is:
 • Strongly typed   (dataclass fields)
 • Validated         (fail-fast at init)
 • Immutable         (frozen dataclasses)
 • Environment-aware (development / testing / production)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from diffusion_lm.constants import (
    MODEL_TYPES,
    SCHEDULE_LINEAR,
    SUPPORTED_ENVIRONMENTS,
    VALID_SCHEDULES,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _validate_positive(value: int | float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_in(value: str, valid: frozenset[str], name: str) -> None:
    if value not in valid:
        raise ValueError(f"{name} must be one of {sorted(valid)}, got '{value}'")


# ── Model Configuration ─────────────────────────────────────────────

@dataclass(frozen=True)
class ModelConfig:
    """Transformer architecture hyper-parameters."""

    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    vocab_size: int = 30522        # BERT default
    max_seq_len: int = 512
    dropout: float = 0.1

    def __post_init__(self) -> None:
        _validate_positive(self.hidden_size, "hidden_size")
        _validate_positive(self.num_layers, "num_layers")
        _validate_positive(self.num_heads, "num_heads")
        _validate_positive(self.vocab_size, "vocab_size")
        _validate_positive(self.max_seq_len, "max_seq_len")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible "
                f"by num_heads ({self.num_heads})"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


# ── Training Configuration ──────────────────────────────────────────

@dataclass(frozen=True)
class TrainingConfig:
    """Training loop hyper-parameters."""

    learning_rate: float = 3e-4
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_steps: int = 100_000
    warmup_steps: int = 1_000
    mixed_precision: bool = True
    checkpoint_dir: str = "checkpoints"
    log_every: int = 50
    save_every: int = 5_000
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        _validate_positive(self.learning_rate, "learning_rate")
        _validate_positive(self.batch_size, "batch_size")
        _validate_positive(self.grad_accum_steps, "grad_accum_steps")
        _validate_positive(self.max_steps, "max_steps")
        _validate_positive(self.log_every, "log_every")
        _validate_positive(self.save_every, "save_every")


# ── Diffusion Configuration ─────────────────────────────────────────

@dataclass(frozen=True)
class DiffusionConfig:
    """Diffusion process parameters."""

    num_steps: int = 64
    schedule: str = SCHEDULE_LINEAR
    temperature: float = 1.0
    block_size: int = 64          # Only used by BD3LM

    def __post_init__(self) -> None:
        _validate_positive(self.num_steps, "num_steps")
        _validate_positive(self.temperature, "temperature")
        _validate_positive(self.block_size, "block_size")
        _validate_in(self.schedule, VALID_SCHEDULES, "schedule")


# ── Data Configuration ───────────────────────────────────────────────

@dataclass(frozen=True)
class DataConfig:
    """Dataset and tokenisation settings."""

    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    tokenizer_name: str = "bert-base-uncased"
    max_length: int = 512
    streaming: bool = False

    def __post_init__(self) -> None:
        _validate_positive(self.max_length, "max_length")
        if not self.dataset_name:
            raise ValueError("dataset_name must not be empty")
        if not self.tokenizer_name:
            raise ValueError("tokenizer_name must not be empty")


# ── Top-Level Application Configuration ──────────────────────────────

@dataclass(frozen=True)
class AppConfig:
    """Root configuration object — composed of sub-configs.

    Exactly one instance should exist per process.
    """

    environment: str = "development"
    model_type: str = "mdlm"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self) -> None:
        _validate_in(self.environment, SUPPORTED_ENVIRONMENTS, "environment")
        _validate_in(self.model_type, MODEL_TYPES, "model_type")


# ── Factory ──────────────────────────────────────────────────────────

def load_config(
    environment: str = "development",
    model_type: str = "mdlm",
    *,
    model_overrides: Optional[dict] = None,
    training_overrides: Optional[dict] = None,
    diffusion_overrides: Optional[dict] = None,
    data_overrides: Optional[dict] = None,
) -> AppConfig:
    """Build and validate an immutable AppConfig.

    Args:
        environment: One of development / testing / production.
        model_type:  One of mdlm / bd3lm.
        *_overrides: Dicts merged on top of defaults. Invalid keys raise.

    Returns:
        Frozen, validated AppConfig.

    Raises:
        ValueError: On any invalid configuration value.
        TypeError:  On unexpected override keys.
    """
    model_cfg = ModelConfig(**(model_overrides or {}))
    training_cfg = TrainingConfig(**(training_overrides or {}))
    diffusion_cfg = DiffusionConfig(**(diffusion_overrides or {}))
    data_cfg = DataConfig(**(data_overrides or {}))

    config = AppConfig(
        environment=environment,
        model_type=model_type,
        model=model_cfg,
        training=training_cfg,
        diffusion=diffusion_cfg,
        data=data_cfg,
    )
    return config
