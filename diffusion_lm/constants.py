"""Fixed constants for the diffusion language model pipeline.

These are non-configurable system values: protocol-level identifiers
and fixed defaults that never change across environments.
"""

# ── Token identifiers ──────────────────────────────────────────────
MASK_TOKEN_ID: int = 103          # Default [MASK] id (BERT-style tokenizers)
PAD_TOKEN_ID: int = 0            # Default [PAD] id

# ── Model types ─────────────────────────────────────────────────────
MODEL_TYPES: frozenset[str] = frozenset({"mdlm", "bd3lm"})

# ── Noise schedule types ────────────────────────────────────────────
SCHEDULE_LINEAR: str = "linear"
SCHEDULE_COSINE: str = "cosine"
VALID_SCHEDULES: frozenset[str] = frozenset({SCHEDULE_LINEAR, SCHEDULE_COSINE})

# ── Sampling strategies ─────────────────────────────────────────────
SAMPLING_GREEDY: str = "greedy"
SAMPLING_TEMPERATURE: str = "temperature"
VALID_SAMPLING: frozenset[str] = frozenset({SAMPLING_GREEDY, SAMPLING_TEMPERATURE})

# ── Environment identifiers ─────────────────────────────────────────
SUPPORTED_ENVIRONMENTS: frozenset[str] = frozenset({
    "development",
    "testing",
    "production",
})

# ── Numerical stability ─────────────────────────────────────────────
EPSILON: float = 1e-8
MIN_TIMESTEP: float = 1e-5
MAX_TIMESTEP: float = 1.0 - 1e-5
