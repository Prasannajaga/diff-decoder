from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import torch

from diffusion import DiffusionConfig, DiscreteMaskDiffusion
from model import DiffusionTransformer, DiffusionTransformerConfig

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.text import Text

    _RICH_AVAILABLE = True
except Exception:
    _RICH_AVAILABLE = False


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


def token_to_visible_slot(token_id: int, tokenizer: ByteTokenizer) -> str:
    if token_id == tokenizer.MASK_ID:
        return "<MASK>"
    if token_id in (tokenizer.PAD_ID, tokenizer.BOS_ID, tokenizer.EOS_ID):
        return ""
    if 0 <= token_id <= 255:
        ch = chr(token_id)
        if ch == "\n":
            return "\\n"
        if ch == "\t":
            return "\\t"
        if ch == "\r":
            return "\\r"
        if ch.isprintable():
            return ch
    return "?"


class MaskStreamRenderer:
    def __init__(
        self,
        prompt: str,
        max_new_tokens: int,
        total_steps: int,
        stream_every: int,
        enabled: bool,
    ) -> None:
        self.prompt = prompt
        self.total_tokens = max_new_tokens
        self.total_steps = total_steps
        self.stream_every = max(1, stream_every)
        self.enabled = enabled
        self.slots = ["<MASK>"] * max_new_tokens
        self.known_count = 0
        self.current_step = total_steps
        self._started = False

        self.use_rich = (
            enabled
            and _RICH_AVAILABLE
            and sys.stdout.isatty()
            and os.environ.get("TERM", "") not in ("", "dumb")
        )

        self.console = None
        self.layout = None
        self.progress = None
        self.task_id = None
        self.live = None

        if self.use_rich:
            self.console = Console(force_terminal=True, color_system="truecolor")
            self.layout = Layout()
            self.layout.split_column(
                Layout(name="text", ratio=1),
                Layout(name="progress", size=3),
            )

            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Diffusion"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TextColumn("[cyan]Known: {task.fields[known]}"),
                TextColumn("•"),
                TextColumn("[magenta]{task.fields[pct]:>4s}"),
                TimeRemainingColumn(),
                expand=True,
            )
            self.task_id = self.progress.add_task(
                "Streaming",
                total=self.total_steps + 1,
                known=f"{self.known_count}/{self.total_tokens}",
                pct="0%",
            )
            self.live = Live(
                self.layout,
                console=self.console,
                refresh_per_second=20,
                transient=False,
            )

    def _render_plain(self) -> None:
        slot_str = " ".join(slot if slot else "_" for slot in self.slots)
        line = (
            f"step={self.current_step:03d} known={self.known_count:03d}/{self.total_tokens:03d} "
            f"| {self.prompt} {slot_str}"
        )
        sys.stdout.write("\r\033[2K" + line)
        sys.stdout.flush()

    def _render_rich(self) -> None:
        assert self.layout is not None and self.progress is not None and self.task_id is not None
        slots_text = " ".join(self.slots if self.slots else [""])
        body = Text(f"{self.prompt} {slots_text}")
        panel = Panel(
            body if body.plain else Text("[dim]no tokens[/dim]"),
            title="[bold]Mask Decoding",
            subtitle=f"[dim]Step {self.current_step}/{self.total_steps}[/dim]",
            border_style="cyan",
            padding=(1, 1),
        )
        self.layout["text"].update(panel)

        completed = (self.total_steps - self.current_step) + 1
        completed = max(0, min(self.total_steps + 1, completed))
        pct = int(100 * completed / max(1, self.total_steps + 1))
        self.progress.update(
            self.task_id,
            completed=completed,
            known=f"{self.known_count}/{self.total_tokens}",
            pct=f"{pct}%",
        )
        self.layout["progress"].update(Panel(self.progress))

    def start(self) -> None:
        if not self.enabled or self._started:
            return
        self._started = True
        if self.use_rich and self.live is not None:
            self.live.start()
            self._render_rich()
        else:
            self._render_plain()

    def should_render(self, phase: str, step: int) -> bool:
        if phase in ("init", "final"):
            return True
        if phase == "denoise":
            return step == 1 or step % self.stream_every == 0
        return False

    def update(self, phase: str, step: int, generated_ids: list[int], known_mask: list[bool]) -> None:
        if not self.enabled:
            return
        if not self.should_render(phase, step):
            return

        self.current_step = step
        total = min(self.total_tokens, len(known_mask), len(generated_ids))
        for i in range(total):
            if known_mask[i]:
                self.slots[i] = generated_ids[i]
        self.known_count = sum(1 for v in known_mask[:total] if v)

        if self.use_rich:
            self._render_rich()
        else:
            self._render_plain()

    def finish(self) -> None:
        if not self.enabled:
            return
        if self.use_rich and self.live is not None:
            self.live.stop()
        else:
            sys.stdout.write("\n")
            sys.stdout.flush()

def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file:
        file_path = Path(args.prompt_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines()]
        prompts = [line for line in lines if line]
        if not prompts:
            raise ValueError("Prompt file is empty after removing blank lines.")
        return prompts
    return [args.prompt]


@torch.no_grad()
def generate_for_prompt(
    model: DiffusionTransformer,
    diffusion: DiscreteMaskDiffusion,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_seq_len: int,
    max_new_tokens: int,
    num_samples: int,
    temperature: float,
    device: torch.device,
    stream: bool,
    stream_every: int,
) -> list[str]:
    max_prompt_len = max_seq_len - max_new_tokens
    # if max_prompt_len <= 2:
    #     raise ValueError(
    #         "max_new_tokens is too large for model max_seq_len; reduce max_new_tokens or train with larger max_seq_len."
    #     )
    if stream and num_samples != 1:
        raise ValueError("Live stream mode currently supports --num_samples 1 for proper in-place rendering.")

    prompt_ids = tokenizer.encode(prompt, max_prompt_len)
    prompt_tensor = torch.tensor([prompt_ids] * num_samples, dtype=torch.long, device=device)
    renderer = MaskStreamRenderer(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        total_steps=diffusion.cfg.num_steps,
        stream_every=stream_every,
        enabled=stream,
    )
    if stream:
        renderer.start()

    def stream_callback(event: dict) -> None:
        if not stream:
            return

        phase = event["phase"]
        step = int(event["step"])

        gen_ids = event["generated_ids"][0].detach().cpu().tolist()
        known = event["known_mask"][0].detach().cpu().tolist()
        slot_values = [token_to_visible_slot(tid, tokenizer) for tid in gen_ids]
        renderer.update(phase=phase, step=step, generated_ids=slot_values, known_mask=known)

    generated = diffusion.sample(
        model=model,
        prefix_ids=prompt_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stream_callback=stream_callback if stream else None,
    )
    if stream:
        renderer.finish()

    outputs: list[str] = []
    for i in range(num_samples):
        full_ids = prompt_tensor[i].tolist() + generated[i].tolist()
        outputs.append(tokenizer.decode(full_ids))
    return outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with a custom diffusion LM checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--prompt_file", type=str, default="", help="Optional file with one prompt per line")
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--stream", action="store_true", help="Live stream mask-first denoising updates")
    p.add_argument("--stream_every", type=int, default=1, help="Print every N denoising steps in stream mode")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    payload = torch.load(ckpt_path, map_location=device)
    model_cfg = DiffusionTransformerConfig(**payload["model_config"])
    # Hardcoded diffusion config for now (instead of loading from checkpoint metadata).
    diff_cfg = DiffusionConfig(
        num_steps=500,
        mask_token_id=257,
        pad_token_id=256,
    )

    model = DiffusionTransformer(model_cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    diffusion = DiscreteMaskDiffusion(diff_cfg)
    tokenizer = ByteTokenizer()

    prompts = load_prompts(args)

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Device: {device}")

    for p_idx, prompt in enumerate(prompts, start=1):
        outputs = generate_for_prompt(
            model=model,
            diffusion=diffusion,
            tokenizer=tokenizer,
            prompt=prompt,
            max_seq_len=model_cfg.max_seq_len,
            max_new_tokens=args.max_new_tokens,
            num_samples=args.num_samples,
            temperature=args.temperature,
            device=device,
            stream=args.stream,
            stream_every=args.stream_every,
        )

        print("\n" + "=" * 80)
        print(f"Prompt {p_idx}: {prompt}")
        print("=" * 80)
        for i, text in enumerate(outputs, start=1):
            print(f"\n[{i}] {text}")


if __name__ == "__main__":
    main()
