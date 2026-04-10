"""
Production-quality training wrapper for a Transformer model. 
* Responsible for optimizer, amp, accumulation, clipping, checkpointing, logging, LR schedule
* Optimized for low VRAM (6GB) with gradient checkpointing, configurable AMP dtype, and memory management
""" 

import os
import gc
import time
import math
import re
import sys
import json
import tempfile
import traceback
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from typing import Optional, Any, Dict
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm import tqdm
from utils.config import TrainingConfig
from utils.config_io import save_config_json
from utils.eval_metrics import EvalMetrics
from utils.vertexExperiments import VertexExperiments

SUPPORTED_OPTIMIZERS = frozenset({"adamw", "adam", "sgd", "adafactor"})

class Trainer:

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        tokenizer: Optional[Any] = None,
        ckpt_dir: Optional[str] = None,
        prompts: Optional[list[str]] = None,
    ):
        self.model = model
        self.config = config
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.tokenizer = tokenizer
        self.prompts = list(prompts or [])
        self.enable_vertex_tracking = bool(getattr(self.config, "enable_vertex_tracking", True))
        self.enable_metrics = bool(getattr(self.config, "enable_metrics", False))
        self.show_progress_bar = os.environ.get("TRAIN_PROGRESS_BAR", "0") == "1"
        # checkpoint directory precedence: explicit arg > config.ckpt_dir
        self.ckpt_dir = ckpt_dir or config.ckpt_dir
        self.logs_dir: Optional[str] = None
        self.metrics_dir: Optional[str] = None
        self.log_file_path: Optional[str] = None
        self.step_metrics_path: Optional[str] = None
        self.vertex_event_path: Optional[str] = None
        self.tensorboard_dir: Optional[str] = None
        self.vertex_experiments: Optional[VertexExperiments] = None
        if self.ckpt_dir:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.logs_dir = os.path.join(self.ckpt_dir, "logs")
            os.makedirs(self.logs_dir, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self.log_file_path = os.path.join(self.logs_dir, f"trainer_{timestamp}.log")
            if self.enable_vertex_tracking:
                self.step_metrics_path = os.path.join(self.logs_dir, f"trainer_steps_{timestamp}.jsonl")
                self.vertex_event_path = os.path.join(self.logs_dir, f"vertex_events_{timestamp}.jsonl")
            if bool(getattr(self.config, "enable_tensorboard", True)):
                self.tensorboard_dir = os.path.join(self.logs_dir, "tensorboard")
                os.makedirs(self.tensorboard_dir, exist_ok=True)
            if self.enable_metrics or len(self.prompts) > 0:
                self.metrics_dir = os.path.join(self.ckpt_dir, "metrics")
                os.makedirs(self.metrics_dir, exist_ok=True)

        # device placement (defensive fallback for broken CUDA runtime)
        try:
            self.model.to(self.device)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if self.device.type == "cuda" and "cuda" in msg:
                self.device = torch.device("cpu")
                self.config.device = "cpu"
                print(f"[DEVICE] CUDA move failed ({exc}). Falling back to CPU.")
                self.model.to(self.device)
            else:
                raise
        self._setup_gradient_checkpointing()

        # AMP setup: disable on CPU automatically
        self.use_amp = bool(self.config.use_amp and self.device.type == "cuda")
        self.amp_dtype = self._resolve_amp_dtype()
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)


        # Optimizer with parameter groups (no weight decay on bias & LayerNorm)
        self.optimizer = self._create_optimizer()

        # lr schedule state (manual cosine + warmup)
        if self.config.total_steps is None:
            # total_steps unknown: leave as None; trainer.train will attempt to infer
            self.total_steps = None
        else:
            self.total_steps = int(self.config.total_steps)
        self.warmup_steps = int(self.config.warmup_steps or 0)
        # store base lrs to scale
        self._base_lrs = [g.get("lr", self.config.lr) for g in self.optimizer.param_groups]
        # ensure initial lr matches config.lr if not set in param groups
        for g in self.optimizer.param_groups:
            if "lr" not in g or g["lr"] is None:
                g["lr"] = self.config.lr

        # bookkeeping
        self.global_step = 0  # increments after each optimizer.step()
        self._accum_counter = 0  
        self._lr_schedule_anchor_step = 0

        # timer
        self._train_start_time = None
        self._last_step_time = None  

        # Loss tracking for metadata
        self._best_train_loss = float("inf")
        self._worst_train_loss = float("-inf")
        self._final_train_loss = None
        self._best_train_loss_step = 0
        self._best_val_loss = float("inf")
        self._final_val_loss = None
        self._best_val_loss_step = 0
        self._total_tokens_trained = 0
        self._total_training_time = 0.0

        # CUDA optimizations
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self._clear_cuda_cache()
        
        # Log initialization info
        if self.config.enable_logging:
            self._log_init_info()
        self._init_vertex_tracking()

    def _log(self, message: str):
        if self.config.enable_logging:
            tqdm.write(message, file=sys.stdout)
            if self.log_file_path:
                try:
                    with open(self.log_file_path, "a", encoding="utf-8") as fp:
                        fp.write(f"{datetime.now().isoformat()} | {message}\n")
                except Exception:
                    pass

    def _log_exception(self, prefix: str):
        self._log(prefix)
        self._log(traceback.format_exc().rstrip())
    
    def _log_init_info(self):
        params = self.parameter_counts()
        self._log(f"[Trainer] Model: {params['trainable_params_m']:.2f}M trainable / {params['total_params_m']:.2f}M total ({params['trainable_percent']:.1f}%)")
        self._log(f"[Trainer] Optimizer: {self.config.optimizer.upper()} | LR: {self.config.lr:.2e} | Weight Decay: {self.config.weight_decay}")
        if self.device.type == "cuda":
            mem = self.get_memory_summary()
            self._log(f"[Trainer] GPU Memory: {mem['allocated_gb']:.2f}GB allocated | Device: {self.device}")

    def _is_distributed(self) -> bool:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()

    def _is_main_process(self) -> bool:
        if not self._is_distributed():
            return True
        import torch.distributed as dist
        return dist.get_rank() == 0

    def _track_vertex_event(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self.vertex_experiments is None:
            return
        self.vertex_experiments.track_event(event, payload or {}, global_step=self.global_step)

    def _init_vertex_tracking(self) -> None:
        if not self._is_main_process():
            return
        enable_tensorboard = bool(getattr(self.config, "enable_tensorboard", True))
        if not self.enable_vertex_tracking and not enable_tensorboard:
            return
        self.vertex_experiments = VertexExperiments(
            enabled=self.enable_vertex_tracking,
            is_main_process=True,
            event_path=self.vertex_event_path,
            tensorboard_dir=self.tensorboard_dir,
            enable_tensorboard=enable_tensorboard,
            tensorboard_hist_interval_steps=int(
                getattr(self.config, "tensorboard_hist_interval_steps", 0) or 0
            ),
            run_config=asdict(self.config),
            log_interval_steps=int(getattr(self.config, "log_interval_steps", 0) or 0),
            logger=self._log,
            log_exception=self._log_exception,
        )
        self.vertex_experiments.start(
            global_step=self.global_step,
            device=str(self.device),
            world_size=self._get_world_size(),
            enable_metrics=bool(self.enable_metrics),
        )

    def _log_vertex_sdk_metrics(self, metrics: Dict[str, float], step: int, force: bool = False) -> None:
        if self.vertex_experiments is None:
            return
        self.vertex_experiments.log_metrics(metrics=metrics, step=step, force=force)

    def finalize_vertex_tracking(self, status: str = "completed", error: Optional[str] = None) -> None:
        if self.vertex_experiments is not None:
            self.vertex_experiments.finalize(
                status=status,
                error=error,
                global_step=self.global_step,
                best_train_loss=self._clean_metric(self._best_train_loss, invalid=float("inf")),
                best_val_loss=self._clean_metric(self._best_val_loss, invalid=float("inf")),
                final_train_loss=self._clean_metric(self._final_train_loss, invalid=float("inf")),
                final_val_loss=self._clean_metric(self._final_val_loss, invalid=float("inf")),
            )

    def _unwrap_model(self) -> nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _resolve_amp_dtype(self) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        dtype_str = self.config.amp_dtype.lower()
        if dtype_str not in dtype_map:
            self._log(f"[Trainer] Warning: Unknown amp_dtype '{self.config.amp_dtype}', defaulting to float16")
            return torch.float16
        
        requested_dtype = dtype_map[dtype_str]
        
        # Validate bfloat16 support
        if requested_dtype == torch.bfloat16 and self.device.type == "cuda":
            if not torch.cuda.is_bf16_supported():
                self._log("[Trainer] Warning: bfloat16 not supported, falling back to float16")
                return torch.float16
        
        return requested_dtype
    
    def _setup_gradient_checkpointing(self):
        if not self.config.enable_gradient_checkpointing:
            return
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            self._log("[Trainer] Enabled gradient checkpointing")
            return
        checkpointed_count = 0
        for module in self.model.modules():
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = True
                checkpointed_count += 1
        if checkpointed_count > 0:
            self._log(f"[Trainer] Enabled gradient checkpointing on {checkpointed_count} modules")
        else:
            self._log("[Trainer] Warning: Gradient checkpointing not supported by model")
    
    def _clear_cuda_cache(self):
        if self.device.type != "cuda":
            return
        gc.collect()
        torch.cuda.empty_cache()
    
    def reset_peak_memory_stats(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def get_memory_summary(self) -> Dict[str, float]:
        if self.device.type != "cuda":
            return {"device": "cpu", "allocated_gb": 0, "reserved_gb": 0, "peak_gb": 0}
        return {
            "device": str(self.device),
            "allocated_gb": round(torch.cuda.memory_allocated(self.device) / (1024 ** 3), 3),
            "reserved_gb": round(torch.cuda.memory_reserved(self.device) / (1024 ** 3), 3),
            "peak_gb": round(torch.cuda.max_memory_allocated(self.device) / (1024 ** 3), 3),
        }

    def _create_optimizer(self):
        optimizer_name = getattr(self.config, 'optimizer', 'adamw').lower()
        
        if optimizer_name not in SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer: '{self.config.optimizer}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_OPTIMIZERS))}"
            )
        
        # Exclude bias and LayerNorm/Norm weights from weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or "layernorm" in name.lower() or "layer_norm" in name.lower() or "ln_" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        # Create optimizer based on config
        if optimizer_name == "adamw":
            return AdamW(param_groups, lr=self.config.lr, betas=self.config.betas, eps=self.config.eps)
        elif optimizer_name == "adam":
            return torch.optim.Adam(param_groups, lr=self.config.lr, betas=self.config.betas, eps=self.config.eps)
        elif optimizer_name == "sgd":
            return torch.optim.SGD(param_groups, lr=self.config.lr, momentum=0.9)
        elif optimizer_name == "adafactor":
            try:
                from transformers.optimization import Adafactor
                return Adafactor(param_groups, lr=self.config.lr, relative_step=False, warmup_init=False)
            except ImportError:
                raise ImportError(
                    "Adafactor requires the 'transformers' library. "
                    "Install with: pip install transformers"
                )
 

    def _get_lr_factor(self, step: int) -> float: 
        anchor = max(0, int(getattr(self, "_lr_schedule_anchor_step", 0)))
        schedule_step = max(0, int(step) - anchor)
        if self.total_steps is None or self.total_steps <= 0:
            if self.warmup_steps > 0:
                return min(1.0, float(schedule_step) / float(max(1, self.warmup_steps)))
            return 1.0

        total_steps = int(self.total_steps)
        if anchor > 0:
            # When schedule is rebased at resume, decay only across the remaining window.
            total_steps = max(1, total_steps - anchor)

        schedule_step = min(schedule_step, total_steps)
        if schedule_step < self.warmup_steps and self.warmup_steps > 0:
            return float(schedule_step) / float(max(1, self.warmup_steps))
        
        if total_steps == self.warmup_steps:
            return 1.0
        progress = float(schedule_step - self.warmup_steps) / float(max(1, total_steps - self.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _update_lr(self): 
        factor = self._get_lr_factor(self.global_step)
        for base, group in zip(self._base_lrs, self.optimizer.param_groups):
            group["lr"] = base * factor

    def rebase_lr_schedule(self, anchor_step: Optional[int] = None) -> None:
        if anchor_step is None:
            anchor_step = self.global_step
        self._lr_schedule_anchor_step = max(0, int(anchor_step))
        self._update_lr()
        self._log(
            f"[Trainer] Rebased LR schedule at step {self._lr_schedule_anchor_step} "
            f"(total_steps={self.total_steps}, warmup_steps={self.warmup_steps})"
        )

    def _eval_interval_steps(self) -> int:
        value = int(getattr(self.config, "eval_interval_steps", 0) or 0)
        if value <= 0:
            value = int(getattr(self.config, "log_interval_steps", 0) or 0)
        return max(0, value)

    def _maybe_clear_cuda_cache(self) -> None:
        if self.device.type != "cuda":
            return
        interval = int(getattr(self.config, "clear_cache_interval", 0) or 0)
        if interval <= 0:
            return
        if self.global_step > 0 and (self.global_step % interval == 0):
            self._clear_cuda_cache()
 
    def parameter_counts(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        pct = (trainable / total * 100.0) if total > 0 else 0.0
        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_percent": pct,
            "total_params_m": total / 1e6,
            "trainable_params_m": trainable / 1e6,
        }

    def _track_step_metadata(self, loss: float, batch):
        self._final_train_loss = loss
        if loss < self._best_train_loss:
            self._best_train_loss = loss
            self._best_train_loss_step = self.global_step
        if loss > self._worst_train_loss:
            self._worst_train_loss = loss
        if isinstance(batch, (list, tuple)):
            batch_tensor = batch[0]
        elif isinstance(batch, dict):
            batch_tensor = self._get_first_present(
                batch, ["input_ids", "inputs", "input"]
            )
        else:
            batch_tensor = batch
        if batch_tensor is not None and hasattr(batch_tensor, 'size'):
            batch_size = batch_tensor.size(0)
            seq_length = batch_tensor.size(1) if batch_tensor.dim() > 1 else self.config.block_size
            self._total_tokens_trained += batch_size * seq_length

    @staticmethod
    def _get_first_present(batch: dict, keys: list[str]):
        for key in keys:
            if key in batch and batch[key] is not None:
                return batch[key]
        return None

    def _build_progress_postfix(
        self,
        avg_loss: float,
        lr: float,
        grad_norm: Optional[float] = None,
        val_loss: Optional[float] = None,
    ) -> Dict[str, str]:
        postfix = {
            "train_loss": f"{avg_loss:.4f}",
            "lr": f"{lr:.2e}",
        }
        if grad_norm is not None:
            postfix["gnorm"] = f"{grad_norm:.2f}"
        if val_loss is not None:
            postfix["val"] = f"{val_loss:.4f}"
        if self.config.log_memory_usage and self.device.type == "cuda":
            mem = self.get_memory_summary()
            postfix["mem"] = f"{mem['allocated_gb']:.1f}GB"
        return postfix

    def _should_log_step(self, step: int) -> bool:
        if not self.config.enable_logging:
            return False
        interval = int(getattr(self.config, "log_interval_steps", 0) or 0)
        return interval > 0 and step > 0 and step % interval == 0

    def _build_step_log_message(
        self,
        step: int,
        train_loss: float,
        lr: float,
        grad_norm: Optional[float],
        val_loss: Optional[float],
        iter_time_s: Optional[float],
    ) -> str:
        gnorm_str = f"{grad_norm:>6.2f}" if grad_norm is not None else "   n/a"
        val_str = f"{val_loss:>8.4f}" if val_loss is not None else "     n/a"
        iter_ms_str = f"{iter_time_s * 1000.0:>7.1f}" if iter_time_s is not None else "    n/a"
        return (
            f"[Trainer] step={step:>7d} | "
            f"train_loss={train_loss:>8.4f} | "
            f"val_loss={val_str} | "
            f"lr={lr:>9.2e} | "
            f"gnorm={gnorm_str} | "
            f"iter_ms={iter_ms_str}"
        )

    def _log_step_metrics(
        self,
        *,
        step: int,
        train_loss: float,
        lr: float,
        grad_norm: Optional[float],
        val_loss: Optional[float],
        iter_time_s: Optional[float],
    ) -> None:
        if not self.step_metrics_path:
            return
        payload = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "step": int(step),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "lr": float(lr),
            "grad_norm": float(grad_norm) if grad_norm is not None else None,
            "iter_time_ms": (float(iter_time_s) * 1000.0) if iter_time_s is not None else None,
        }
        try:
            with open(self.step_metrics_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload) + "\n")
        except Exception:
            pass
        self._track_vertex_event("step_metrics", payload)
        metrics_for_vertex = {
            "train_loss": float(train_loss),
            "lr": float(lr),
        }
        if grad_norm is not None:
            metrics_for_vertex["grad_norm"] = float(grad_norm)
        if val_loss is not None:
            metrics_for_vertex["val_loss"] = float(val_loss)
        self._log_vertex_sdk_metrics(metrics_for_vertex, step=step)
        if self.vertex_experiments is not None:
            memory_summary = self.get_memory_summary() if self.device.type == "cuda" else {}
            self.vertex_experiments.log_tensorboard_step(
                step=step,
                train_loss=train_loss,
                lr=lr,
                grad_norm=grad_norm,
                val_loss=val_loss,
                iter_time_s=iter_time_s,
                model=self._unwrap_model(),
                memory_summary=memory_summary,
            )

    def _generate_training_metadata(self, checkpoint_path: str) -> Dict[str, Any]:
        params = self.parameter_counts()
        mem = self.get_memory_summary()
        avg_tokens_per_sec = 0.0
        if self._total_training_time > 0 and self._total_tokens_trained > 0:
            avg_tokens_per_sec = self._total_tokens_trained / self._total_training_time

        return {
            "model": {
                "name": self._unwrap_model().__class__.__name__,
                "checkpoint_path": checkpoint_path,
                "timestamp": datetime.now().isoformat(),
                "total_params": params["total_params"],
                "trainable_params": params["trainable_params"],
            },
            "training": {
                "total_steps": self.global_step,
                "tokens_trained": self._total_tokens_trained,
                "time_seconds": round(self._total_training_time, 2),
                "tokens_per_second": round(avg_tokens_per_sec, 2),
            },
            "train_loss": {
                "best": self._best_train_loss if self._best_train_loss != float("inf") else None,
                "worst": self._worst_train_loss if self._worst_train_loss != float("-inf") else None,
                "final": self._final_train_loss,
                "best_step": self._best_train_loss_step,
            },
            "val_loss": {
                "best": self._best_val_loss if self._best_val_loss != float("inf") else None,
                "final": self._final_val_loss,
                "best_step": self._best_val_loss_step,
            },
            "system": {
                "device": str(self.device),
                "gpu_count": torch.cuda.device_count() if self.device.type == "cuda" else 0,
                "peak_memory_gb": mem.get("peak_gb", 0),
            },
            "optimizer": {
                "name": getattr(self.config, 'optimizer', 'adamw'),
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "grad_accum_steps": self.config.grad_accum_steps,
                "batch_size": self.config.train_batch_size,
                "effective_batch_size": self.config.train_batch_size * self.config.grad_accum_steps,
            },
        }

    def _save_metadata_atomically(self, metadata: Dict[str, Any], path: str):
        target_path = os.path.abspath(path)
        dir_path = os.path.dirname(target_path)
        os.makedirs(dir_path, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(suffix=".json", dir=dir_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            os.replace(temp_path, target_path)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _save_checkpoint_metadata(self, checkpoint_path: str, checkpoint_name: str):
        if not self.metrics_dir:
            return
        base_name = os.path.splitext(checkpoint_name)[0]
        metadata_path = os.path.join(self.metrics_dir, f"{base_name}.json")
        try:
            metadata = self._generate_training_metadata(checkpoint_path)
            self._save_metadata_atomically(metadata, metadata_path)
            self._log(f"[Trainer] Saved metadata: {metadata_path}")
        except Exception:
            self._log_exception("[Trainer] Warning: Failed to save metadata")

    @staticmethod
    def _clean_metric(value: Optional[float], *, invalid: float) -> Optional[float]:
        if value is None:
            return None
        if value == invalid:
            return None
        return float(value)

    def _get_world_size(self) -> int:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return int(dist.get_world_size())
        except Exception:
            pass
        return 1

    def build_results_payload(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = self.parameter_counts()
        mem = self.get_memory_summary()
        world_size = self._get_world_size()
        total_tokens_local = int(self._total_tokens_trained)
        total_tokens_global = int(total_tokens_local * world_size)
        train_time_s = float(self._total_training_time or 0.0)
        tokens_per_second = (total_tokens_global / train_time_s) if train_time_s > 0 else 0.0

        payload: Dict[str, Any] = {
            "schema_version": 1,
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "run": {
                "global_step": int(self.global_step),
                "target_total_steps": int(self.total_steps) if self.total_steps is not None else None,
                "world_size": int(world_size),
                "tokens_trained_local": total_tokens_local,
                "tokens_trained_global_estimate": total_tokens_global,
                "train_time_seconds": round(train_time_s, 3),
                "tokens_per_second_global_estimate": round(tokens_per_second, 3),
            },
            "loss": {
                "best_train_loss": self._clean_metric(self._best_train_loss, invalid=float("inf")),
                "best_train_loss_step": int(self._best_train_loss_step),
                "worst_train_loss": self._clean_metric(self._worst_train_loss, invalid=float("-inf")),
                "final_train_loss": self._clean_metric(self._final_train_loss, invalid=float("inf")),
                "best_val_loss": self._clean_metric(self._best_val_loss, invalid=float("inf")),
                "best_val_loss_step": int(self._best_val_loss_step),
                "final_val_loss": self._clean_metric(self._final_val_loss, invalid=float("inf")),
            },
            "model": {
                "name": self._unwrap_model().__class__.__name__,
                "n_layer": int(self.config.n_layer),
                "n_embd": int(self.config.n_embd),
                "n_head": int(self.config.n_head),
                "block_size": int(self.config.block_size),
                "total_params": int(params["total_params"]),
                "trainable_params": int(params["trainable_params"]),
                "trainable_percent": float(params["trainable_percent"]),
            },
            "optimization": {
                "optimizer": str(getattr(self.config, "optimizer", "adamw")),
                "lr": float(self.config.lr),
                "eps": float(self.config.eps),
                "weight_decay": float(self.config.weight_decay),
                "grad_clip_norm": float(self.config.grad_clip_norm),
                "grad_accum_steps": int(self.config.grad_accum_steps),
                "train_batch_size_per_process": int(self.config.train_batch_size),
                "eval_batch_size_per_process": int(self.config.eval_batch_size),
                "effective_batch_size_global": int(
                    self.config.train_batch_size * self.config.grad_accum_steps * world_size
                ),
                "use_amp": bool(self.use_amp),
                "amp_dtype": str(self.config.amp_dtype),
                "gradient_checkpointing": bool(self.config.enable_gradient_checkpointing),
            },
            "system": {
                "device": str(self.device),
                "gpu_count_visible": int(torch.cuda.device_count()) if self.device.type == "cuda" else 0,
                "peak_memory_gb": float(mem.get("peak_gb", 0.0)),
                "allocated_memory_gb": float(mem.get("allocated_gb", 0.0)),
                "reserved_memory_gb": float(mem.get("reserved_gb", 0.0)),
            },
        }

        if extra:
            payload.update(extra)
        return payload

    def save_results_json(
        self,
        path: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        if path is None:
            if self.ckpt_dir:
                path = os.path.join(self.ckpt_dir, "results.json")
            else:
                path = os.path.abspath("results.json")
        payload = self.build_results_payload(extra=extra)
        self._save_metadata_atomically(payload, path)
        self._log(f"[Trainer] Saved results: {path}")
        self._track_vertex_event(
            "results_saved",
            {
                "path": os.path.abspath(path),
                "schema_version": payload.get("schema_version"),
            },
        )
        return path

    def save_prompt_metrics(
        self,
        prompts: Optional[list[str]] = None,
        path: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Optional[str]:
        prompts = list(prompts or self.prompts)
        if len(prompts) == 0:
            self._log("[Trainer] Prompt metrics skipped: no prompts provided")
            return None
        if self.tokenizer is None:
            self._log("[Trainer] Prompt metrics skipped: tokenizer missing")
            return None

        if path is None:
            base_dir = self.metrics_dir or self.ckpt_dir or os.getcwd()
            os.makedirs(base_dir, exist_ok=True)
            path = os.path.join(base_dir, f"prompt_metrics_step_{self.global_step}.json")

        use_top_k = bool(getattr(self.config, "use_top_k", True))
        top_k = int(getattr(self.config, "top_k", 50)) if use_top_k else None
        use_repetition_penalty = bool(getattr(self.config, "use_repetition_penalty", True))
        repetition_penalty = (
            float(getattr(self.config, "repetition_penalty", 1.2))
            if use_repetition_penalty
            else 1.0
        )
        temperature = float(getattr(self.config, "temperature", 0.7))
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if max_new_tokens is None:
            max_new_tokens = int(getattr(self.config, "max_new_tokens", 128))

        evaluator = EvalMetrics(
            prompts=prompts,
            model=self._unwrap_model(),
            tokenizer=self.tokenizer,
            device=self.device,
            use_amp=self.use_amp,
            amp_dtype=self.amp_dtype,
            max_new_tokens=int(max_new_tokens),
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
        )
        payload = evaluator.run(global_step=self.global_step)
        successful = int(payload.get("summary", {}).get("success_count", 0))
        
        self._save_metadata_atomically(payload, path)
        self._log(f"[Trainer] Saved prompt metrics: {path}")
        self._track_vertex_event(
            "prompt_metrics_saved",
            {
                "path": os.path.abspath(path),
                "prompt_count": len(prompts),
                "success_count": successful,
            },
        )
        return path
  
    def save_checkpoint(
        self,
        step: Optional[int] = None,
        prefix: str = "model",
        include_optimizer_state: Optional[bool] = None,
        include_scaler_state: Optional[bool] = None,
        include_training_step: bool = True,
    ) -> str:
        step = self.global_step if step is None else int(step)
        if not self.ckpt_dir:
            raise ValueError("Checkpoint directory not configured.")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # Save config.json once (only if it doesn't exist)
        config_path = os.path.join(self.ckpt_dir, "config.json")
        if not os.path.exists(config_path):
            save_config_json(
                config_path=config_path,
                training_cfg=self.config,
                tokenizer=self.tokenizer,
                model=self._unwrap_model(),
            )
            self._log(f"[Trainer] Saved config: {config_path}")
        
        # Save checkpoint file in checkpoint root directory.
        fname = f"{prefix}_{step}.pt"
        path = os.path.join(self.ckpt_dir, fname)
        model_to_save = self._unwrap_model()
        payload = {"model_state_dict": model_to_save.state_dict()}
        
        if include_training_step:
            payload["training_step"] = step
            # Keep both keys for compatibility with state_dict()/load_state_dict() paths.
            payload["global_step"] = step
        payload["lr_schedule_anchor_step"] = int(self._lr_schedule_anchor_step)
        # Persist config in checkpoint payload so resume does not depend on sidecar files.
        payload["config"] = asdict(self.config)
        
        if include_optimizer_state is None:
            include_optimizer_state = bool(self.config.save_optimizer_state)
        
        if include_scaler_state is None:
            include_scaler_state = bool(include_optimizer_state)

        # Save tokenizer files once in checkpoint root.
        if self.tokenizer is not None:
            tokenizer_json_path = os.path.join(self.ckpt_dir, "tokenizer.json")
            if not os.path.exists(tokenizer_json_path):
                try:
                    self.tokenizer.save_pretrained(self.ckpt_dir)
                    self._log(f"[Trainer] Saved tokenizer files: {self.ckpt_dir}")
                except Exception:
                    self._log_exception("[Trainer] Warning: Failed to save tokenizer")

        if include_optimizer_state:
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
        
        if include_scaler_state and self.scaler is not None:
            try:
                payload["scaler_state_dict"] = self.scaler.state_dict()
            except Exception:
                payload["scaler_state_dict"] = None
        torch.save(payload, path)

        self._log(f"[Trainer] Saved checkpoint: {path}")
        self._track_vertex_event(
            "checkpoint_saved",
            {
                "path": os.path.abspath(path),
                "step": int(step),
                "include_optimizer_state": bool(include_optimizer_state),
            },
        )
        return path 

    def save_final_checkpoint(self, step: Optional[int] = None, prefix: str = "final_ckpt") -> str:
        ckpt_path = self.save_checkpoint(
            step=step,
            prefix=prefix,
            include_optimizer_state=False,
            include_scaler_state=False,
            include_training_step=False,
        )
        self._save_checkpoint_metadata(ckpt_path, os.path.basename(ckpt_path))
        return ckpt_path

    def load_checkpoint(self, path: str, map_location: Optional[torch.device] = None) -> Dict[str, Any]:
        
        if map_location is None:
            map_location = self.device
            
        data = torch.load(path, map_location=map_location)
        if "model_state_dict" not in data:
            raise KeyError(f"Checkpoint missing required key 'model_state_dict': {path}")
        self._unwrap_model().load_state_dict(data["model_state_dict"], strict=True)

        if "optimizer_state_dict" in data and data["optimizer_state_dict"] is not None:
            try:
                self.optimizer.load_state_dict(data["optimizer_state_dict"])
            except Exception:
                self._log_exception("[Trainer] Warning: Failed to load optimizer state")
        if "scaler_state_dict" in data and data["scaler_state_dict"] is not None:
            try:
                self.scaler.load_state_dict(data["scaler_state_dict"])
            except Exception:
                pass
       
        # first callback to check if the steps available in the model file 
        step_value = data.get("training_step", None)
        if step_value is None:
            step_value = data.get("global_step", None)
        if step_value is None:
            # Older final checkpoints may omit step fields; infer from filename suffix.
            name = os.path.basename(path)
            match = re.search(r"_(\d+)\.pt$", name)
            if match:
                step_value = int(match.group(1))
        self.global_step = int(step_value or 0)
        self._lr_schedule_anchor_step = int(data.get("lr_schedule_anchor_step", 0) or 0)
        saved_cfg = data.get("config", None)

        # fallback check if checkpoints not available there load config.json.
        if saved_cfg is None:
            try:
                cfg_path = os.path.join(os.path.dirname(os.path.abspath(path)), "config.json")
                if os.path.isfile(cfg_path):
                    with open(cfg_path, "r", encoding="utf-8") as fp:
                        saved_cfg = json.load(fp)
            except Exception:
                saved_cfg = None

        if saved_cfg and self.total_steps is None:
            self.total_steps = int(saved_cfg.get("total_steps")) if saved_cfg.get("total_steps") else None
            self.warmup_steps = int(saved_cfg.get("warmup_steps", self.warmup_steps))

        self._update_lr()
        self._unwrap_model().to(self.device)
        self._log(f"[Trainer] Loaded checkpoint from {path} (step {self.global_step})")

        return data
  
    # Validation loss estimation   
    def estimate_validation_loss(self, val_dataloader, num_batches: Optional[int] = None) -> float: 
        """Memory-efficient validation loss estimation."""
        model_for_eval = self._unwrap_model()
        model_for_eval.eval()
        num_batches = int(num_batches or self.config.validation_batch_count)
        total_loss = 0.0
        seen = 0 

        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                if seen >= num_batches:
                    break
                inputs, targets = self._unpack_batch(batch)
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                with torch.amp.autocast(enabled=self.use_amp, device_type=self.device.type, dtype=self.amp_dtype):
                    model_out = model_for_eval(inputs)
                    loss = self._compute_loss(model_out, targets)
                total_loss += loss.detach()
                del inputs, targets, model_out, loss
                seen += 1

        self.model.train()
        return (total_loss / max(1, seen)).item()
    

    def log_dataset_info(self, dataset, dataloader, config: TrainingConfig) -> None:
        import math
        samples = len(dataset)
        batches_per_epoch = len(dataloader)
        epochs_needed = math.ceil(config.total_steps / batches_per_epoch) if batches_per_epoch > 0 else 0
        print(f"[DATA] Samples: {samples} | Batch size: {config.train_batch_size} | Batches/epoch: {batches_per_epoch}")
        print(f"[DATA] Total steps: {config.total_steps} | Epochs needed: {epochs_needed}")

    # Core training primitives  
    def _unpack_batch(self, batch):
        """Standardize dataloader batch -> (inputs, targets)."""
        if isinstance(batch, (list, tuple)):
            if len(batch) == 1:
                return batch[0], batch[0]
            return batch[0], batch[1]
        if isinstance(batch, dict):
            inp = self._get_first_present(batch, ["input_ids", "inputs", "input"])
            tgt = self._get_first_present(batch, ["labels", "targets", "targets_ids"])
            if inp is None:
                raise ValueError("Dict batch missing input_ids/inputs key")
            if tgt is None:
                tgt = inp
            return inp, tgt
        return batch, batch

    def _compute_loss(self, logits, targets):  

        if isinstance(logits, dict) and "loss" in logits:
            return logits["loss"]
        if isinstance(logits, (tuple, list)):
            logits_tensor = logits[0]
        else:
            logits_tensor = logits

        B, T, V = logits_tensor.shape
        loss = nn.functional.cross_entropy(
            logits_tensor.reshape(-1, V),
            targets.reshape(-1),
            # ignore_index=-100,
            # reduction="mean",
        )
        return loss

    def train_step(self, batch, sync_gradients: bool = True) -> Dict[str, Any]:
        """Perform forward + backward for a single batch."""
        self.model.train()
        inputs, targets = self._unpack_batch(batch)
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        accum_steps = max(1, int(self.config.grad_accum_steps)) 

        sync_ctx = nullcontext()
        if not sync_gradients and hasattr(self.model, "no_sync"):
            sync_ctx = self.model.no_sync()

        with sync_ctx:
            with torch.amp.autocast(enabled=self.use_amp, device_type=self.device.type, dtype=self.amp_dtype):
                model_out = self.model(inputs)
                loss = self._compute_loss(model_out, targets)

            scaled_loss = loss / accum_steps
            self.scaler.scale(scaled_loss).backward() 
        self._accum_counter += 1

        did_step = False
        grad_norm = None
        current_lr = None

        if self._accum_counter >= accum_steps:
            try:
                self.scaler.unscale_(self.optimizer)
            except Exception:
                pass

            if self.config.grad_clip_norm and self.config.grad_clip_norm > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                grad_norm = float(grad_norm) if hasattr(grad_norm, "item") else float(grad_norm)
            else: 
                grad_norm = None

            try:
                self.scaler.step(self.optimizer)
            except Exception:
                self.scaler.update()
                raise
            else:
                self.scaler.update()
                did_step = True
                self.global_step += 1
                self._update_lr()
               
            finally:
                self.optimizer.zero_grad(set_to_none=True)
                self._accum_counter = 0 
                current_lr = float(self.optimizer.param_groups[0]["lr"])

        return {
            "loss": loss.item(),
            "did_step": did_step,
            "grad_norm": grad_norm,
            "lr": current_lr,
        }
  
    # Training loop with tqdm 
    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        epochs: Optional[int] = 1,
        max_steps: Optional[int] = None,
    ):
        """
        High-level training loop with tqdm progress bar.
        - train_dataloader: iterable of batches
        - val_dataloader: optional validation dataloader for periodic eval
        - epochs: number of epochs to run if total_steps not specified
        - max_steps: optional override for total optimizer steps
        """
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        # Infer total_steps
        if max_steps is not None:
            self.total_steps = int(max_steps)
        elif self.total_steps is None:
            try:
                steps_per_epoch = len(train_dataloader)
            except Exception:
                steps_per_epoch = None
            if steps_per_epoch:
                approx_total = int(math.ceil(steps_per_epoch * float(epochs) / max(1, self.config.grad_accum_steps)))
                self.total_steps = approx_total

        self._train_start_time = time.perf_counter()
        self._last_step_time = self._train_start_time
        self.model.train()

        global_step_target = None if self.total_steps is None else int(self.total_steps)
        
        # Create progress bar
        pbar = tqdm(
            total=global_step_target,
            initial=self.global_step,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            leave=True,
            file=sys.stdout,
            disable=not self.show_progress_bar,
        )
        
        # Tracking for smoother progress bar updates
        running_loss = 0.0
        loss_count = 0
        last_val_loss = None
        eval_interval_steps = self._eval_interval_steps()

        stop_requested = False
        epoch = 0
        use_epoch_limit = global_step_target is None

        try:
            while not stop_requested:
                if use_epoch_limit and epoch >= epochs:
                    break
                epoch += 1
                for batch in train_dataloader:
                    step_info = self.train_step(batch)
                    
                    # Accumulate loss for averaging
                    running_loss += step_info["loss"]
                    loss_count += 1

                    self._track_step_metadata(step_info["loss"], batch)

                    if step_info["did_step"]:
                        now = time.perf_counter()
                        iter_time_s = now - self._last_step_time if self._last_step_time else None
                        self._last_step_time = now

                        avg_loss = running_loss / loss_count
                        validated_this_step = False

                        if val_dataloader is not None and self.config.validation_batch_count and eval_interval_steps > 0:
                            if self.global_step % eval_interval_steps == 0:
                                try:
                                    last_val_loss = self.estimate_validation_loss(
                                        val_dataloader,
                                        num_batches=self.config.validation_batch_count
                                    )
                                    self._final_val_loss = last_val_loss
                                    if last_val_loss < self._best_val_loss:
                                        self._best_val_loss = last_val_loss
                                        self._best_val_loss_step = self.global_step
                                    validated_this_step = True
                                except Exception:
                                    self._log_exception("[Trainer] Warning: Validation failed")

                        postfix = self._build_progress_postfix(
                            avg_loss=avg_loss,
                            lr=step_info["lr"],
                            grad_norm=step_info["grad_norm"],
                            val_loss=last_val_loss,
                        )
                        pbar.set_postfix(postfix)
                        pbar.update(1)
                        # Keep reporting the latest known val loss between eval intervals.
                        current_val_loss = last_val_loss
                        self._log_step_metrics(
                            step=self.global_step,
                            train_loss=avg_loss,
                            lr=step_info["lr"],
                            grad_norm=step_info["grad_norm"],
                            val_loss=current_val_loss,
                            iter_time_s=iter_time_s,
                        )

                        if self._should_log_step(self.global_step):
                            if self.config.log_per_iteration_time:
                                step_time = iter_time_s
                            else:
                                step_time = None
                            self._log(
                                self._build_step_log_message(
                                    step=self.global_step,
                                    train_loss=avg_loss,
                                    lr=step_info["lr"],
                                    grad_norm=step_info["grad_norm"],
                                    val_loss=current_val_loss,
                                    iter_time_s=step_time,
                                )
                            )
                        
                        # Reset running loss periodically
                        if self.global_step % 100 == 0:
                            running_loss = 0.0
                            loss_count = 0

                        # Checkpointing
                        if self.ckpt_dir and self.config.ckpt_interval_steps:
                            if self.global_step % int(self.config.ckpt_interval_steps) == 0:
                                try:
                                    self.save_checkpoint(step=self.global_step)
                                except Exception:
                                    self._log_exception("[Trainer] Warning: Checkpoint save failed")

                        self._maybe_clear_cuda_cache()

                    # Stopping condition
                    if global_step_target is not None and self.global_step >= global_step_target:
                        stop_requested = True
                        break
                
                if stop_requested:
                    self._log(f"[Trainer] Stop requested ...")
                    break

        finally:
            pbar.close()

        total_wall = time.perf_counter() - self._train_start_time
        self._total_training_time = total_wall
        
        # Final summary
        self._log(f"[Trainer] Training complete: {self.global_step} steps in {total_wall:.1f}s ({total_wall/60:.1f}min)")
        if self.device.type == "cuda":
            mem = self.get_memory_summary()
            self._log(f"[Trainer] Peak GPU memory: {mem['peak_gb']:.2f}GB")
        self.finalize_vertex_tracking(status="completed")
        
 

    def train_distributed(
        self,
        train_dataloader,
        val_dataloader=None,
        epochs: Optional[int] = 1,
        max_steps: Optional[int] = None,
    ):
        """
        Distributed training loop using torch.distributed.
        Safely handles DistributedSampler epochs, rank-0 logging/checkpointing,
        and DDP-aware validation.
        """
        import torch.distributed as dist
        
        is_dist = dist.is_available() and dist.is_initialized()
        if not is_dist:
            self._log("[Trainer] torch.distributed is not initialized. Falling back to single-process train().")
            return self.train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                epochs=epochs,
                max_steps=max_steps,
            )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        is_main = rank == 0

        if not isinstance(self.model, DDP):
            if self.device.type == "cuda":
                device_index = self.device.index
                if device_index is None:
                    device_index = torch.cuda.current_device()
                self.model = DDP(
                    self.model,
                    device_ids=[device_index],
                    output_device=device_index,
                    broadcast_buffers=False,
                    find_unused_parameters=False,
                )
            else:
                self.model = DDP(self.model, broadcast_buffers=False)
            if is_main:
                self._log(f"[Trainer] Wrapped model with DDP (world_size={world_size})")

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed + rank)

        # Infer total_steps
        if max_steps is not None:
            self.total_steps = int(max_steps)
        elif self.total_steps is None:
            try:
                steps_per_epoch = len(train_dataloader)
            except Exception:
                steps_per_epoch = None
            if steps_per_epoch:
                approx_total = int(math.ceil(steps_per_epoch * float(epochs) / max(1, self.config.grad_accum_steps)))
                self.total_steps = approx_total

        if is_main:
            self._train_start_time = time.perf_counter()
            self._last_step_time = self._train_start_time
            
        self.model.train()

        global_step_target = None if self.total_steps is None else int(self.total_steps)
        
        pbar = None
        if is_main:
            pbar = tqdm(
                total=global_step_target,
                initial=self.global_step,
                desc=f"Distributed Training (world={world_size})",
                unit="step",
                dynamic_ncols=True,
                leave=True,
                file=sys.stdout,
                disable=not self.show_progress_bar,
            )
        
        running_loss = 0.0
        loss_count = 0
        last_val_loss = None
        eval_interval_steps = self._eval_interval_steps()

        stop_requested = False
        epoch = 0
        use_epoch_limit = global_step_target is None
        accum_steps = max(1, int(self.config.grad_accum_steps))

        try:
            while not stop_requested:
                if use_epoch_limit and epoch >= epochs:
                    break
                    
                # Support DistributedSampler
                if hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
                    train_dataloader.sampler.set_epoch(epoch)
                    
                epoch += 1
                for batch in train_dataloader:
                    should_sync = (self._accum_counter + 1) >= accum_steps
                    step_info = self.train_step(batch, sync_gradients=should_sync)
                    
                    running_loss += step_info["loss"]
                    loss_count += 1

                    if is_main:
                        self._track_step_metadata(step_info["loss"], batch)

                    if step_info["did_step"]:
                        validated_this_step = False
                        if (
                            is_main
                            and val_dataloader is not None
                            and self.config.validation_batch_count
                            and eval_interval_steps > 0
                        ):
                            if self.global_step % eval_interval_steps == 0:
                                try:
                                    val_loss = self.estimate_validation_loss(
                                        val_dataloader,
                                        num_batches=self.config.validation_batch_count
                                    )
                                    last_val_loss = val_loss
                                    self._final_val_loss = last_val_loss
                                    if last_val_loss < self._best_val_loss:
                                        self._best_val_loss = last_val_loss
                                        self._best_val_loss_step = self.global_step
                                    validated_this_step = True
                                except Exception:
                                    self._log_exception("[Trainer] Warning: Validation failed")

                        if is_main:
                            now = time.perf_counter()
                            iter_time_s = now - self._last_step_time if self._last_step_time else None
                            self._last_step_time = now

                            avg_loss = running_loss / loss_count

                            postfix = self._build_progress_postfix(
                                avg_loss=avg_loss,
                                lr=step_info["lr"],
                                grad_norm=step_info["grad_norm"],
                                val_loss=last_val_loss,
                            )
                            pbar.set_postfix(postfix)
                            pbar.update(1)
                            # Keep reporting the latest known val loss between eval intervals.
                            current_val_loss = last_val_loss
                            self._log_step_metrics(
                                step=self.global_step,
                                train_loss=avg_loss,
                                lr=step_info["lr"],
                                grad_norm=step_info["grad_norm"],
                                val_loss=current_val_loss,
                                iter_time_s=iter_time_s,
                            )

                            if self._should_log_step(self.global_step):
                                step_time = iter_time_s if self.config.log_per_iteration_time else None
                                self._log(
                                    self._build_step_log_message(
                                        step=self.global_step,
                                        train_loss=avg_loss,
                                        lr=step_info["lr"],
                                        grad_norm=step_info["grad_norm"],
                                        val_loss=current_val_loss,
                                        iter_time_s=step_time,
                                    )
                                )
                            
                        if self.global_step % 100 == 0:
                            running_loss = 0.0
                            loss_count = 0

                        # Checkpointing (Main node only)
                        if is_main and self.ckpt_dir and self.config.ckpt_interval_steps:
                            if self.global_step % int(self.config.ckpt_interval_steps) == 0:
                                try:
                                    self.save_checkpoint(step=self.global_step)
                                except Exception:
                                    self._log_exception("[Trainer] Warning: Checkpoint save failed")

                        self._maybe_clear_cuda_cache()

                    if global_step_target is not None and self.global_step >= global_step_target:
                        stop_requested = True
                        break
                
                if stop_requested:
                    if is_main:
                        self._log(f"[Trainer] Stop requested ...")
                    break

        finally:
            if pbar is not None:
                pbar.close()

        if is_main:
            total_wall = time.perf_counter() - self._train_start_time
            self._total_training_time = total_wall
            
            self._log(f"[Trainer] Training complete: {self.global_step} steps in {total_wall:.1f}s ({total_wall/60:.1f}min)")
            if self.device.type == "cuda":
                mem = self.get_memory_summary()
                self._log(f"[Trainer] Peak GPU memory: {mem['peak_gb']:.2f}GB")
            self.finalize_vertex_tracking(status="completed")
            
    def state_dict(self) -> Dict[str, Any]: 
        state = {
            "model_state_dict": self._unwrap_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": getattr(self.scaler, "state_dict", lambda: None)(),
            "global_step": self.global_step,
            "lr_schedule_anchor_step": int(self._lr_schedule_anchor_step),
            "config": asdict(self.config),
        }
        return state

    def load_state_dict(self, state: Dict[str, Any]):  
        self._unwrap_model().load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state and state["optimizer_state_dict"] is not None:
            try:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
            except Exception:
                self._log_exception("[Trainer] Warning: optimizer state load failed")
        if "scaler_state_dict" in state and state["scaler_state_dict"] is not None:
            try:
                self.scaler.load_state_dict(state["scaler_state_dict"])
            except Exception:
                pass
        self.global_step = int(state.get("global_step", 0))
        self._lr_schedule_anchor_step = int(state.get("lr_schedule_anchor_step", 0) or 0)
        self._update_lr()
