#!/usr/bin/env python3
"""CLI entrypoint — train MDLM or BD3LM diffusion language models.

Uses the project's shared PackedDatasetBuilder and Trainer for
consistency across projects.

Usage:
    python train.py --model mdlm
    python train.py --model bd3lm --block-size 64
    python train.py --model mdlm --dataset wikitext --batch-size 16
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from diffusion_lm.constants import MODEL_TYPES
from diffusion_lm.models.mdlm import MDLM
from diffusion_lm.models.bd3lm import BD3LM
from diffusion_lm.data.packed_dataset_builder import PackedDatasetBuilder
from diffusion_lm.training.trainer import Trainer
from utils.config import TrainingConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a diffusion language model")

    # Model
    p.add_argument("--model", type=str, default="mdlm", choices=sorted(MODEL_TYPES))
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"])
    p.add_argument("--block-size", type=int, default=64, help="Block size for BD3LM")

    # Data
    p.add_argument("--dataset", type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    p.add_argument("--data-dir", type=str, default="data", help="Output dir for packed binary")
    p.add_argument("--skip-pack", action="store_true", help="Skip dataset packing (use existing binary)")

    # Training
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--warmup-steps", type=int, default=1_000)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Device
    p.add_argument("--device", type=str, default=None, help="Auto-detect if not set")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ── Tokenizer ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    mask_token_id = tokenizer.mask_token_id or 103

    # ── Dataset (PackedDatasetBuilder) ───────────────────────────────
    if not args.skip_pack:
        builder = PackedDatasetBuilder(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            block_size=args.max_seq_len,
            output_path=args.data_dir,
            subset=args.dataset_config if args.dataset_config != args.dataset else None,
            split="train",
        )
        builder.build()

    train_loader, val_loader = PackedDatasetBuilder.to_dataloader(
        bin_path=args.data_dir,
        block_size=args.max_seq_len,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        val_ratio=args.val_ratio,
    )

    # ── Model ────────────────────────────────────────────────────────
    if args.model == "mdlm":
        model = MDLM(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
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
            dropout=args.dropout,
            block_size=args.block_size,
            mask_token_id=mask_token_id,
            schedule=args.schedule,
        )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model.upper()}")
    print(f"Parameters: {param_count:,}")
    print(f"Device: {device}")

    # ── Training config ──────────────────────────────────────────────
    training_config = TrainingConfig(
        lr=args.lr,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        train_batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        use_amp=not args.no_amp,
        block_size=args.max_seq_len,
        ckpt_dir=args.ckpt_dir,
    )

    # ── Trainer ──────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        config=training_config,
        device=device,
        tokenizer=tokenizer,
        ckpt_dir=args.ckpt_dir,
    )

    if args.resume:
        print(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
    )

    trainer.save_final_checkpoint()
    trainer.save_results_json()
    print("Training complete.")


if __name__ == "__main__":
    main()
