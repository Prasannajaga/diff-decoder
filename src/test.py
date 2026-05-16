#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import io
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    TextIteratorStreamer,
    set_seed,
)
from transformers.utils import logging as transformers_logging

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

# Ensure local vendored `dllm` package is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDORED_DLLM_ROOT = PROJECT_ROOT / "src" / "dllm"
if VENDORED_DLLM_ROOT.is_dir():
    sys.path.insert(0, str(VENDORED_DLLM_ROOT))

from dllm.core.samplers import BD3LMSampler, BD3LMSamplerConfig

DEFAULT_MODEL_PATH = "/media/prasanna/716F26140AED9B67/models/Qwen2.5-Coder-0.5B-Instruct"
DIFFUSION_NAME_HINTS = ("diffusion-bd3lm", "bd3lm", "diffusion")


@dataclass
class RunResult:
    model_source: str
    mode: str
    elapsed_sec: float
    output_tokens: int
    tps: float
    params: int
    prompt_tokens: int
    text: str


def token_to_visible_slot(token_id: int, tokenizer) -> str:
    if token_id == tokenizer.mask_token_id:
        return "<MASK>"
    if token_id in {tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id}:
        return ""
    text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if not text:
        text = tokenizer.convert_ids_to_tokens(token_id) or ""
    text = text.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
    return text if len(text) <= 16 else text[:13] + "..."


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

    def update(self, phase: str, step: int, generated_ids: list[str], known_mask: list[bool]) -> None:
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


def is_diffusion_name(path: Path) -> bool:
    name = path.name.lower()
    return any(hint in name for hint in DIFFUSION_NAME_HINTS)


def resolve_model_source(model_arg: str, prefer_diffusion: bool, verbose: bool = True) -> tuple[str, bool]:
    model_ref = Path(model_arg).expanduser()
    if not model_ref.exists():
        return model_arg, False

    if model_ref.is_dir() and not (model_ref / "config.json").is_file():
        candidate_dirs = sorted(
            d for d in model_ref.iterdir() if d.is_dir() and (d / "config.json").is_file()
        )
        if not candidate_dirs:
            raise ValueError(
                f"Provided path is a directory, but no model folders with config.json were found: {model_ref}"
            )

        preferred = [d for d in candidate_dirs if is_diffusion_name(d)]
        if not prefer_diffusion:
            preferred = [d for d in candidate_dirs if not is_diffusion_name(d)]

        if len(preferred) == 1:
            if verbose:
                print(f"[model] Auto-selected model directory: {preferred[0]}")
            return str(preferred[0]), True
        if len(candidate_dirs) == 1:
            if verbose:
                print(f"[model] Auto-selected model directory: {candidate_dirs[0]}")
            return str(candidate_dirs[0]), True

        sample = ", ".join(d.name for d in candidate_dirs[:8])
        mode_hint = "diffusion" if prefer_diffusion else "causal"
        raise ValueError(
            f"Provided path is a parent directory, not a model directory. "
            f"Found {len(candidate_dirs)} candidates ({sample}). "
            f"Pass a specific folder to --model for {mode_hint} mode."
        )

    return str(model_ref), True


def load_tokenizer(model_source: str, is_local_model: bool):
    kwargs = {
        "trust_remote_code": True,
        "local_files_only": is_local_model,
    }
    try:
        return AutoTokenizer.from_pretrained(model_source, **kwargs)
    except ValueError as exc:
        if "backend tokenizer" not in str(exc):
            raise
        print("Fast tokenizer init failed; retrying with slow tokenizer (`use_fast=False`).")
        return AutoTokenizer.from_pretrained(model_source, use_fast=False, **kwargs)


def load_model(
    model_source: str,
    is_local_model: bool,
    dtype: torch.dtype,
    device: torch.device,
    enable_diffusion: bool,
    verbose: bool = True,
):
    model_cls = AutoModelForMaskedLM if enable_diffusion else AutoModelForCausalLM
    mode = "diffusion" if enable_diffusion else "causal"
    common_kwargs = {
        "dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": is_local_model,
    }
    try:
        model = model_cls.from_pretrained(model_source, **common_kwargs).to(device)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {mode} model from {model_source}") from exc
    if enable_diffusion:
        patch_missing_attention_types(model, verbose=verbose)
    return model, mode


def patch_missing_attention_types(model, verbose: bool = True) -> None:
    """
    Some A2D/BD3LM Qwen remote-code checkpoints expect newer Transformers
    decoder layers to expose `attention_type`, but older/newer local Qwen2
    layer implementations may not set it. The diffusion forward pass indexes
    the prepared mask mapping with this field, so fill it from config when
    needed.
    """
    base_model = getattr(model, "model", None)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        return

    config = getattr(base_model, "config", getattr(model, "config", None))
    layer_types = getattr(config, "layer_types", None) or []

    patched = 0
    for idx, layer in enumerate(layers):
        if hasattr(layer, "attention_type"):
            continue
        attention_type = layer_types[idx] if idx < len(layer_types) else "full_attention"
        layer.attention_type = attention_type
        patched += 1

    if patched and verbose:
        print(f"[compat] Added missing attention_type to {patched} decoder layers.")


def build_prompt_text(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        return prompt


def normalize_tokenizer_for_generation(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


def ensure_diffusion_tokenizer(tokenizer) -> None:
    normalize_tokenizer_for_generation(tokenizer)
    if tokenizer.mask_token_id is None:
        raise ValueError(
            "Diffusion mode requires a tokenizer with mask_token_id. "
            "Use a diffusion/BD3LM model folder or run without --enable-diffusion."
        )
    if tokenizer.pad_token_id is None:
        raise ValueError("Diffusion mode requires pad_token_id or eos_token_id on the tokenizer.")


def run_diffusion_generation(
    model,
    tokenizer,
    prompt_ids: list[int],
    args: argparse.Namespace,
    prompt_for_stream: str | None = None,
) -> list[int]:
    ensure_diffusion_tokenizer(tokenizer)
    sampler = BD3LMSampler(model=model, tokenizer=tokenizer)
    sampler_config = BD3LMSamplerConfig(
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        block_size=args.block_size,
        temperature=args.temperature,
        remasking=args.remasking,
        cfg_scale=args.cfg_scale,
    )
    inputs = [torch.tensor(prompt_ids, dtype=torch.long, device=model.device)]

    renderer = MaskStreamRenderer(
        prompt=(prompt_for_stream or "").strip(),
        max_new_tokens=args.max_new_tokens,
        total_steps=args.steps,
        stream_every=args.stream_every,
        enabled=args.stream,
    )
    if args.stream:
        renderer.start()

        def stream_callback(event: dict) -> None:
            phase = event["phase"]
            step = int(event["step"])

            generated_ids = event["generated_ids"][0].detach().cpu().tolist()
            known_mask = event["known_mask"][0].detach().cpu().tolist()
            slot_values = [token_to_visible_slot(tid, tokenizer) for tid in generated_ids]
            renderer.update(phase=phase, step=step, generated_ids=slot_values, known_mask=known_mask)

        outputs = sampler.sample(
            inputs,
            sampler_config,
            return_dict=True,
            stream_callback=stream_callback,
            stream_every=args.stream_every,
        )
        renderer.finish()
    else:
        outputs = sampler.sample(inputs, sampler_config, return_dict=True)

    sequence = outputs.sequences[0].tolist()
    return sequence[-args.max_new_tokens :]


def build_causal_generate_kwargs(model, tokenizer, args: argparse.Namespace) -> dict:
    normalize_tokenizer_for_generation(tokenizer)
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    if args.temperature > 0:
        generation_config.do_sample = True
        generation_config.temperature = args.temperature
    else:
        generation_config.do_sample = False
        generation_config.temperature = 1.0
        generation_config.top_p = 1.0
        generation_config.top_k = 50

    return {"generation_config": generation_config}


def run_causal_generation(model, tokenizer, prompt_ids: list[int], args: argparse.Namespace) -> list[int]:
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    generate_kwargs = build_causal_generate_kwargs(model, tokenizer, args)

    if args.stream:
        print("[causal] Live stream (token chunks):")
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generate_kwargs["streamer"] = streamer
        holder: dict[str, torch.Tensor] = {}
        error_holder: dict[str, Exception] = {}

        def run_generate() -> None:
            previous_verbosity = transformers_logging.get_verbosity()
            transformers_logging.set_verbosity_error()
            try:
                holder["generated"] = model.generate(input_ids=input_tensor, **generate_kwargs)
            except Exception as exc:  # pragma: no cover
                error_holder["error"] = exc
            finally:
                transformers_logging.set_verbosity(previous_verbosity)

        thread = threading.Thread(target=run_generate, daemon=True)
        thread.start()
        sys.stdout.write("[causal] ")
        sys.stdout.flush()
        for chunk in streamer:
            sys.stdout.write(chunk)
            sys.stdout.flush()
        thread.join()
        sys.stdout.write("\n")
        sys.stdout.flush()
        if "error" in error_holder:
            raise error_holder["error"]
        generated = holder["generated"]
    else:
        generated = model.generate(input_ids=input_tensor, **generate_kwargs)

    sequence = generated[0].tolist()
    return sequence[len(prompt_ids) :]


@torch.no_grad()
def run_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> RunResult:
    verbose = not args.stream
    model_source, is_local_model = resolve_model_source(args.model, args.enable_diffusion, verbose=verbose)
    if verbose:
        print(f"Loading model: {model_source}")
        print(f"Mode: {'diffusion' if args.enable_diffusion else 'causal'}")

    if args.stream:
        load_log_sink = io.StringIO()
        with contextlib.redirect_stdout(load_log_sink), contextlib.redirect_stderr(load_log_sink):
            tokenizer = load_tokenizer(model_source, is_local_model)
            model, mode = load_model(
                model_source,
                is_local_model,
                dtype,
                device,
                args.enable_diffusion,
                verbose=verbose,
            )
    else:
        tokenizer = load_tokenizer(model_source, is_local_model)
        model, mode = load_model(
            model_source,
            is_local_model,
            dtype,
            device,
            args.enable_diffusion,
            verbose=verbose,
        )
    model.eval()
    params = sum(p.numel() for p in model.parameters())

    prompt_text = build_prompt_text(tokenizer, args.prompt)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if not prompt_ids:
        raise ValueError("Tokenizer produced an empty prompt.")

    start = time.perf_counter()
    if args.enable_diffusion:
        completion_ids = run_diffusion_generation(
            model,
            tokenizer,
            prompt_ids,
            args,
            prompt_for_stream=args.prompt,
        )
    else:
        completion_ids = run_causal_generation(model, tokenizer, prompt_ids, args)
    elapsed_sec = time.perf_counter() - start

    output_tokens = len(completion_ids)
    text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip() or "<empty>"
    tps = output_tokens / max(elapsed_sec, 1e-9)

    result = RunResult(
        model_source=model_source,
        mode=mode,
        elapsed_sec=elapsed_sec,
        output_tokens=output_tokens,
        tps=tps,
        params=params,
        prompt_tokens=len(prompt_ids),
        text=text,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def format_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    return str(n)


def short_model_name(model_source: str) -> str:
    if "/" in model_source:
        return Path(model_source).name
    return model_source


def print_summary(result: RunResult, device: torch.device, dtype: torch.dtype) -> None:
    print("\n=== Run Summary ===")
    print(f"Model: {short_model_name(result.model_source)}")
    print(f"Mode: {result.mode}")
    print(f"Device: {device}")
    print(f"DType: {dtype}")
    print(f"Params: {format_params(result.params)}")
    print(f"Prompt tokens: {result.prompt_tokens}")
    print(f"Output tokens: {result.output_tokens}")
    print(f"Time: {result.elapsed_sec:.3f}s")
    print(f"TPS: {result.tps:.2f}")


def print_output(prompt: str, result: RunResult) -> None:
    print("\n=== Prompt ===")
    print(prompt)
    print("\n=== Response ===")
    print(result.text)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def choose_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_by_name[dtype_arg]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU inference only supports --dtype float32 cleanly; use --dtype auto or float32.")
    return dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one local/HF model and report generation throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Model path or Hugging Face model id")
    parser.add_argument(
        "--enable-diffusion",
        action="store_true",
        help="Use BD3LM diffusion sampling instead of autoregressive generation",
    )
    parser.add_argument(
        "--prompt",
        default="Write a Python function that checks whether a string is a palindrome.",
        help="User prompt",
    )
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=128)
    parser.add_argument("--max_new_tokens", dest="max_new_tokens", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature; 0 means greedy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Model weight dtype",
    )
    parser.add_argument("--stream", action="store_true", help="Stream token chunks or diffusion step updates")

    diffusion = parser.add_argument_group("diffusion options")
    diffusion.add_argument("--steps", type=int, default=128, help="Total diffusion steps")
    diffusion.add_argument("--block-size", dest="block_size", type=int, default=32, help="BD3LM block size")
    diffusion.add_argument("--block_size", dest="block_size", type=int, help=argparse.SUPPRESS)
    diffusion.add_argument(
        "--stream-every",
        dest="stream_every",
        type=int,
        default=1,
        help="Render every N diffusion denoising steps",
    )
    diffusion.add_argument("--stream_every", dest="stream_every", type=int, help=argparse.SUPPRESS)
    diffusion.add_argument(
        "--remasking",
        default="low_confidence",
        choices=["low_confidence", "random"],
        help="Diffusion remasking strategy",
    )
    diffusion.add_argument("--cfg-scale", dest="cfg_scale", type=float, default=0.0, help="Classifier-free guidance scale")
    diffusion.add_argument("--cfg_scale", dest="cfg_scale", type=float, help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than 0.")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0.")
    if args.enable_diffusion:
        if args.steps <= 0:
            raise ValueError("--steps must be greater than 0 in diffusion mode.")
        if args.block_size <= 0:
            raise ValueError("--block-size must be greater than 0 in diffusion mode.")
        if args.stream_every <= 0:
            raise ValueError("--stream-every must be greater than 0 in diffusion mode.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)

    if not args.stream:
        print(f"Prompt: {args.prompt}")
        print(f"Max new tokens: {args.max_new_tokens}")
        if args.enable_diffusion:
            print(f"Diffusion steps/block size: {args.steps}/{args.block_size}")

    result = run_model(args, device, dtype)
    print_summary(result, device, dtype)
    print_output(args.prompt, result)


if __name__ == "__main__":
    main()
