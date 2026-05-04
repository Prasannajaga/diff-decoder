from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from diffusion import DiffusionConfig, DiscreteMaskDiffusion
from model import DiffusionTransformer, DiffusionTransformerConfig


class ByteTokenizer:
    PAD_ID = 256
    MASK_ID = 257
    BOS_ID = 258
    EOS_ID = 259
    VOCAB_SIZE = 260

    def encode(self, text: str, max_len: int) -> list[int]:
        token_bytes = list(text.encode("utf-8", errors="ignore"))
        tokens = [self.BOS_ID] + token_bytes + [self.EOS_ID]
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            if tokens[-1] != self.EOS_ID:
                tokens[-1] = self.EOS_ID
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        raw = [t for t in token_ids if 0 <= t <= 255]
        return bytes(raw).decode("utf-8", errors="ignore")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_collate_fn(tokenizer: ByteTokenizer, max_seq_len: int):
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []

        for row in batch:
            text = row.get("text", "")
            tokens = tokenizer.encode(text, max_seq_len)
            pad_len = max_seq_len - len(tokens)
            ids = tokens + [tokenizer.PAD_ID] * pad_len
            mask = [1] * len(tokens) + [0] * pad_len
            input_ids.append(ids)
            attention_mask.append(mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    return collate


def cosine_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * float(step + 1) / float(max(1, warmup_steps))

    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


@torch.no_grad()
def evaluate_loss(
    model: DiffusionTransformer,
    diffusion: DiscreteMaskDiffusion,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x0 = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        stats = diffusion.training_loss(model, x0, attention_mask=attn)
        total_loss += float(stats["loss"].item())
        total_acc += float(stats["acc"].item())
        count += 1

    model.train()
    if count == 0:
        return {"loss": float("nan"), "acc": float("nan")}
    return {"loss": total_loss / count, "acc": total_acc / count}


def build_dataloaders(args, tokenizer: ByteTokenizer):
    train_ds = load_dataset(args.dataset_name, split=args.train_split)

    if args.eval_split:
        eval_ds = load_dataset(args.dataset_name, split=args.eval_split)
    else:
        split = train_ds.train_test_split(test_size=args.eval_fraction, seed=args.seed)
        train_ds = split["train"]
        eval_ds = split["test"]

    if args.max_train_examples > 0:
        train_ds = train_ds.select(range(min(args.max_train_examples, len(train_ds))))
    if args.max_eval_examples > 0:
        eval_ds = eval_ds.select(range(min(args.max_eval_examples, len(eval_ds))))

    collate = make_collate_fn(tokenizer, args.max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    return train_loader, eval_loader


def save_checkpoint(
    out_dir: Path,
    step: int,
    model: DiffusionTransformer,
    optimizer: torch.optim.Optimizer,
    model_cfg: DiffusionTransformerConfig,
    diff_cfg: DiffusionConfig,
    args,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"checkpoint_step_{step:07d}.pt"
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": asdict(model_cfg),
        "diffusion_config": asdict(diff_cfg),
        "train_args": vars(args),
    }
    torch.save(payload, ckpt_path)

    latest_path = out_dir / "latest.pt"
    torch.save(payload, latest_path)
    return ckpt_path


def save_training_curves(
    out_dir: Path,
    train_steps: list[int],
    train_losses: list[float],
    train_accs: list[float],
    eval_steps: list[int],
    eval_losses: list[float],
    eval_accs: list[float],
) -> None:
    if not train_steps:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_steps, train_losses, label="train_loss", color="tab:blue", alpha=0.85)
    if eval_steps:
        axes[0].plot(eval_steps, eval_losses, label="eval_loss", color="tab:orange", marker="o")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(train_steps, train_accs, label="train_acc", color="tab:green", alpha=0.85)
    if eval_steps:
        axes[1].plot(eval_steps, eval_accs, label="eval_acc", color="tab:red", marker="o")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=180)
    plt.close(fig)

    history = {
        "train": {"steps": train_steps, "loss": train_losses, "acc": train_accs},
        "eval": {"steps": eval_steps, "loss": eval_losses, "acc": eval_accs},
    }
    with (out_dir / "training_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a custom diffusion language model on TinyStories.")

    p.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default="validation")
    p.add_argument("--eval_fraction", type=float, default=0.002)

    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_kv_heads", type=int, default=2)
    p.add_argument("--ffn_mult", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--num_diffusion_steps", type=int, default=64)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=1000)

    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_examples", type=int, default=0)
    p.add_argument("--max_eval_examples", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = ByteTokenizer()
    train_loader, eval_loader = build_dataloaders(args, tokenizer)

    model_cfg = DiffusionTransformerConfig(
        vocab_size=tokenizer.VOCAB_SIZE,
        max_seq_len=args.max_seq_len,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
    )
    diff_cfg = DiffusionConfig(
        num_steps=args.num_diffusion_steps,
        mask_token_id=tokenizer.MASK_ID,
        pad_token_id=tokenizer.PAD_ID,
    )

    model = DiffusionTransformer(model_cfg).to(device)
    diffusion = DiscreteMaskDiffusion(diff_cfg)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    global_step = 0
    running_loss = 0.0
    running_acc = 0.0
    log_count = 0
    start_time = time.time()
    train_steps: list[int] = []
    train_losses: list[float] = []
    train_accs: list[float] = []
    eval_steps: list[int] = []
    eval_losses: list[float] = []
    eval_accs: list[float] = []

    model.train()
    while global_step < args.max_steps:
        for batch in train_loader:
            x0 = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            stats = diffusion.training_loss(model, x0, attention_mask=attn)
            loss = stats["loss"] / args.grad_accum_steps
            loss.backward()

            running_loss += float(stats["loss"].item())
            running_acc += float(stats["acc"].item())
            log_count += 1

            if (global_step + 1) % args.grad_accum_steps == 0:
                lr = cosine_lr(
                    step=global_step,
                    total_steps=args.max_steps,
                    warmup_steps=args.warmup_steps,
                    max_lr=args.lr,
                    min_lr=args.min_lr,
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            train_steps.append(global_step)
            train_losses.append(float(stats["loss"].item()))
            train_accs.append(float(stats["acc"].item()))

            if global_step % args.log_every == 0:
                elapsed = time.time() - start_time
                avg_loss = running_loss / max(log_count, 1)
                avg_acc = running_acc / max(log_count, 1)
                print(
                    f"step={global_step:6d} loss={avg_loss:.4f} acc={avg_acc:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6f} time={elapsed:.1f}s"
                )
                running_loss = 0.0
                running_acc = 0.0
                log_count = 0
                start_time = time.time()

            if global_step % args.eval_every == 0:
                val = evaluate_loss(model, diffusion, eval_loader, device, max_batches=50)
                print(f"[eval] step={global_step:6d} val_loss={val['loss']:.4f} val_acc={val['acc']:.4f}")
                eval_steps.append(global_step)
                eval_losses.append(float(val["loss"]))
                eval_accs.append(float(val["acc"]))
                save_training_curves(
                    out_dir=out_dir,
                    train_steps=train_steps,
                    train_losses=train_losses,
                    train_accs=train_accs,
                    eval_steps=eval_steps,
                    eval_losses=eval_losses,
                    eval_accs=eval_accs,
                )

            if global_step % args.save_every == 0:
                ckpt_path = save_checkpoint(out_dir, global_step, model, optimizer, model_cfg, diff_cfg, args)
                print(f"Saved checkpoint: {ckpt_path}")

            if global_step >= args.max_steps:
                break

    ckpt_path = save_checkpoint(out_dir, global_step, model, optimizer, model_cfg, diff_cfg, args)
    save_training_curves(
        out_dir=out_dir,
        train_steps=train_steps,
        train_losses=train_losses,
        train_accs=train_accs,
        eval_steps=eval_steps,
        eval_losses=eval_losses,
        eval_accs=eval_accs,
    )
    print(f"Training complete. Final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
