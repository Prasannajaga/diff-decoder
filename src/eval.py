from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from diffusion import DiffusionConfig, DiscreteMaskDiffusion
from model import DiffusionTransformer, DiffusionTransformerConfig
from tokenizer_utils import HFTokenizerAdapter, load_tokenizer


def make_collate_fn(tokenizer: HFTokenizerAdapter, max_seq_len: int):
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


@torch.no_grad()
def evaluate(
    model: DiffusionTransformer,
    diffusion: DiscreteMaskDiffusion,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x0 = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        stats = diffusion.training_loss(model, x0, attention_mask=attn)
        total_loss += float(stats["loss"].item())
        total_acc += float(stats["acc"].item())
        n += 1

    if n == 0:
        return float("nan"), float("nan")
    return total_loss / n, total_acc / n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate and sample from custom diffusion LM checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    p.add_argument("--eval_split", type=str, default="validation")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_eval_batches", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--gen_tokens", type=int, default=128)
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="",
        help="Optional HF tokenizer path/name. By default, uses checkpoint directory.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_ref = args.tokenizer_name_or_path if args.tokenizer_name_or_path else str(ckpt_path.parent)
    tokenizer = load_tokenizer(tok_ref)

    payload = torch.load(ckpt_path, map_location=device)
    model_cfg = DiffusionTransformerConfig(**payload["model_config"])
    diff_cfg = payload["diffusion_config"]

    model = DiffusionTransformer(model_cfg).to(device)
    model.load_state_dict(payload["model_state"])
    diffusion = DiscreteMaskDiffusion(cfg=DiffusionConfig(**diff_cfg))

    ds = load_dataset(args.dataset_name, split=args.eval_split)
    collate = make_collate_fn(tokenizer, model_cfg.max_seq_len)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    loss, acc = evaluate(model, diffusion, loader, device, max_batches=args.max_eval_batches)
    print(f"tokenizer_vocab={tokenizer.VOCAB_SIZE}")
    print(f"eval_loss={loss:.4f} eval_acc={acc:.4f}")

    prompt_ids = tokenizer.encode(args.prompt, model_cfg.max_seq_len - args.gen_tokens)
    prompt_tensor = torch.tensor([prompt_ids] * args.num_samples, dtype=torch.long, device=device)

    generated = diffusion.sample(
        model=model,
        prefix_ids=prompt_tensor,
        max_new_tokens=args.gen_tokens,
        temperature=args.temperature,
    )

    print("\n=== Samples ===")
    for i in range(args.num_samples):
        full_ids = prompt_tensor[i].tolist() + generated[i].tolist()
        text = tokenizer.decode(full_ids)
        print(f"\n[{i + 1}] {text}")


if __name__ == "__main__":
    main()
