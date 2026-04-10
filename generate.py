#!/usr/bin/env python3
"""CLI entrypoint — generate text with trained MDLM or BD3LM models.

Usage:
    python generate.py --model mdlm --checkpoint checkpoints/step_10000.pt
    python generate.py --model bd3lm --checkpoint checkpoints/step_10000.pt --temperature 0.8
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from diffusion_lm.constants import MODEL_TYPES
from diffusion_lm.models.mdlm import MDLM
from diffusion_lm.models.bd3lm import BD3LM
from diffusion_lm.diffusion.sampler import MDLMSampler, BD3LMSampler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text with a diffusion language model")

    p.add_argument("--model", type=str, default="mdlm", choices=sorted(MODEL_TYPES))
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

    # Model architecture (must match training)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"])

    # Generation
    p.add_argument("--seq-len", type=int, default=128, help="Length of generated sequences")
    p.add_argument("--num-samples", type=int, default=4, help="Number of sequences to generate")
    p.add_argument("--num-steps", type=int, default=64, help="Diffusion denoising steps")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true")

    # Tokenizer
    p.add_argument("--tokenizer", type=str, default="bert-base-uncased")

    # Device
    p.add_argument("--device", type=str, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── Tokenizer ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    mask_token_id = tokenizer.mask_token_id or 103

    # ── Model ────────────────────────────────────────────────────────
    if args.model == "mdlm":
        model = MDLM(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_seq_len=args.max_seq_len,
            mask_token_id=mask_token_id,
            schedule=args.schedule,
        )
    else:
        model = BD3LM(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_seq_len=args.max_seq_len,
            block_size=args.block_size,
            mask_token_id=mask_token_id,
            schedule=args.schedule,
        )

    # ── Load checkpoint ──────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Handle both Trainer checkpoints (model_state_dict) and bare state dicts
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Loaded {args.model.upper()} from {args.checkpoint}")

    # ── Sampler ──────────────────────────────────────────────────────
    if args.model == "mdlm":
        sampler = MDLMSampler(
            predict_fn=model.predict,
            num_steps=args.num_steps,
            temperature=args.temperature,
            greedy=args.greedy,
            mask_token_id=mask_token_id,
        )
    else:
        sampler = BD3LMSampler(
            predict_fn=model.predict,
            block_size=args.block_size,
            num_steps=args.num_steps,
            temperature=args.temperature,
            greedy=args.greedy,
            mask_token_id=mask_token_id,
        )

    # ── Generate ─────────────────────────────────────────────────────
    print(f"\nGenerating {args.num_samples} sequences of length {args.seq_len}...\n")

    tokens = sampler.sample(
        seq_len=args.seq_len,
        batch_size=args.num_samples,
        device=device,
    )

    for i in range(args.num_samples):
        text = tokenizer.decode(tokens[i], skip_special_tokens=True)
        print(f"── Sample {i + 1} ──")
        print(text)
        print()


if __name__ == "__main__":
    main()
