"""Training loop for both curriculum stages.

Plain PyTorch: no Lightning, no Hydra.  One class, one loop, explicit gradient accumulation, so what
runs on a laptop is the same code that runs under ``torchrun`` on a node of H100s.

The two stages differ only in configuration:

* **Stage 1** ``stage: deterministic`` -- weighted MAE, one member per step, long (100 epochs).
* **Stage 2** ``stage: probabilistic`` -- weighted CRPS, two members per step, short (8 epochs),
  ``init_from`` pointing at the Stage-1 checkpoint.

Repeat Stage 2 from the *same* Stage-1 checkpoint with different seeds to build the deep ensemble.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .checkpoint import load_checkpoint, load_weights_into, save_checkpoint
from .config import ExperimentConfig
from .data import accumulation_steps, build_dataloader, build_datasets, build_normalizer
from .data.base import ForecastDataset
from .ema import ExponentialMovingAverage
from .forecaster import UCast, build_forecaster
from .losses import WeightedLoss, build_loss
from .metrics import MetricCollection
from .optim import build_optimizer, set_muon_momentum
from .optim.muon import muon_momentum_schedule
from .utils import (
    barrier,
    count_parameters,
    distributed_info,
    format_count,
    get_logger,
    resolve_device,
    seed_everything,
    setup_logging,
)

__all__ = ["Trainer", "train"]

log = get_logger(__name__)

_AUTOCAST_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "16": torch.float16,
}


class _LossModule(nn.Module):
    """Thin wrapper so the loss is computed *inside* a module's ``forward``.

    DDP only synchronises gradients for work done inside the wrapped module's forward pass, so the
    training step has to enter the model through here rather than calling ``UCast.compute_loss``.
    """

    def __init__(self, model: UCast):
        super().__init__()
        self.model = model

    def forward(self, batch: Mapping[str, Tensor], loss_fn: WeightedLoss, members: int):
        return self.model.compute_loss(batch, loss_fn, members=members)


class Trainer:
    """Owns everything a run needs: data, model, optimizer, EMA, metrics, checkpoints and logging.

    Args:
        config: The experiment configuration.
        datasets: Pre-built datasets keyed by split, if you would rather construct them yourself
            (the tests do this).  Otherwise they are built from ``config.data``.
        device: Override the device; defaults to CUDA, then MPS, then CPU.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        datasets: Mapping[str, ForecastDataset] | None = None,
        device: str | torch.device | None = None,
    ):
        self.config = config
        self.dist = distributed_info()
        setup_logging(rank=self.dist.rank)
        self._init_process_group()
        self.device = resolve_device(device) if device is not None else self._default_device()
        seed_everything(config.train.seed + self.dist.rank, deterministic=config.train.deterministic)

        self.output_dir = Path(config.train.output_dir)
        if self.dist.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "config.json").write_text(config.to_json(indent=2))

        # ---------------------------------------------------------------- data
        self.datasets: dict[str, ForecastDataset] = dict(datasets) if datasets else build_datasets(config)
        if "train" not in self.datasets:
            raise ValueError("a 'train' dataset is required")
        self.spec = self.datasets["train"].spec
        self.train_loader, self.train_sampler = build_dataloader(
            self.datasets["train"], config.data, "train", self.dist.world_size, self.dist.rank, config.train.seed
        )
        self.val_loader, self.val_sampler = None, None
        if "val" in self.datasets:
            self.val_loader, self.val_sampler = build_dataloader(
                self.datasets["val"], config.data, "val", self.dist.world_size, self.dist.rank, config.train.seed
            )

        # ---------------------------------------------------------------- model
        self.normalizer = build_normalizer(
            self.spec.input_variables,
            config.data.statistics,
            dataset=self.datasets["train"],
            max_samples=config.data.compute_statistics_samples or 256,
        )
        self.model: UCast = build_forecaster(config, self.spec, self.normalizer).to(self.device)
        if config.train.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        if config.train.init_from:
            load_weights_into(
                self.model, config.train.init_from, strict=config.train.init_from_strict, map_location=self.device
            )
        self.ema = ExponentialMovingAverage(self.model, decay=config.train.ema_decay).to(self.device)

        self.loss_fn = build_loss(config.train.resolved_loss(), weights=self._loss_weights()).to(self.device)
        self.members = config.train.resolved_ensemble_members()

        # ---------------------------------------------------------------- optimisation
        self.accumulation = accumulation_steps(config.data, self.dist.world_size)
        self.steps_per_epoch = max(1, math.ceil(len(self.train_loader) / self.accumulation))
        self.total_steps = max(1, self.steps_per_epoch * config.train.max_epochs)
        if config.train.max_steps is not None:
            self.total_steps = min(self.total_steps, config.train.max_steps)
        self.optimizer, self.schedule = build_optimizer(self.model, config.optimizer, total_steps=self.total_steps)

        self.autocast_dtype = _AUTOCAST_DTYPES.get(str(config.train.precision).lower())
        use_scaler = self.autocast_dtype is torch.float16 and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=use_scaler)

        self.train_module: nn.Module = _LossModule(self.model)
        if self.dist.enabled:
            self.train_module = nn.parallel.DistributedDataParallel(
                self.train_module,
                device_ids=[self.dist.local_rank] if self.device.type == "cuda" else None,
                find_unused_parameters=False,
            )
        if config.train.compile:
            self.train_module = torch.compile(self.train_module)  # type: ignore[assignment]

        self.epoch = 0
        self.global_step = 0
        self.best_metric = math.inf if config.train.monitor_mode == "min" else -math.inf
        self._wandb = None
        if config.train.resume_from:
            self._resume(config.train.resume_from)

        log.info(
            "Stage %s | loss=%s | members=%d | %s parameters | %d steps/epoch x %d epochs "
            "(effective batch %d = %d per device x %d accumulation x %d ranks)",
            config.train.stage,
            config.train.resolved_loss(),
            self.members,
            format_count(count_parameters(self.model)),
            self.steps_per_epoch,
            config.train.max_epochs,
            config.data.batch_size_per_device * self.accumulation * self.dist.world_size,
            config.data.batch_size_per_device,
            self.accumulation,
            self.dist.world_size,
        )

    # ------------------------------------------------------------------ setup helpers
    def _init_process_group(self) -> None:
        import torch.distributed as dist

        if not self.dist.enabled or dist.is_initialized():
            return
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        log.info("Initialised the %s process group: %s", backend, self.dist)

    def _default_device(self) -> torch.device:
        if torch.cuda.is_available():
            torch.cuda.set_device(self.dist.local_rank)
            return torch.device("cuda", self.dist.local_rank)
        return resolve_device()

    def _loss_weights(self) -> Tensor:
        """``(C_out, H, 1)`` area x variable weights for the training loss."""
        channel_weights = self.spec.output_variables.loss_weights()
        return self.spec.grid.channel_area_weights(channel_weights, device=self.device)

    def _autocast(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def _resume(self, path: str) -> None:
        payload = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        if "ema" in payload:
            self.ema.load_state_dict(payload["ema"], strict=False)
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        if "schedule" in payload:
            self.schedule.load_state_dict(payload["schedule"])
        self.epoch = int(payload.get("epoch", 0)) + 1
        self.global_step = int(payload.get("global_step", 0))
        log.info("Resumed from %s at epoch %d (step %d).", path, self.epoch, self.global_step)

    # ------------------------------------------------------------------ logging
    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if not self.dist.is_main:
            return
        record = {"step": self.global_step if step is None else step, "epoch": self.epoch, **dict(metrics)}
        with (self.output_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if self._wandb is not None:
            self._wandb.log(record, step=record["step"])

    def _maybe_init_wandb(self) -> None:
        project = self.config.train.wandb_project
        if not project or not self.dist.is_main or self._wandb is not None:
            return
        try:
            import wandb
        except ImportError:
            log.warning("wandb_project is set but wandb is not installed; logging to metrics.jsonl only.")
            return
        self._wandb = wandb.init(
            project=project,
            entity=self.config.train.wandb_entity,
            name=self.config.name,
            config=self.config.to_dict(),
            dir=str(self.output_dir),
        )

    # ------------------------------------------------------------------ training
    def to_device(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                value = value.to(self.device, non_blocking=True)
                if self.config.train.channels_last and value.ndim == 4:
                    value = value.contiguous(memory_format=torch.channels_last)
            out[key] = value
        return out

    def backward(self, batch: Mapping[str, Tensor], accumulate: bool) -> float:
        """Forward + backward for one micro-batch.  Skips gradient sync while accumulating."""
        needs_no_sync = accumulate and isinstance(self.train_module, nn.parallel.DistributedDataParallel)
        with self.train_module.no_sync() if needs_no_sync else nullcontext():
            with self._autocast():
                loss, _ = self.train_module(batch, self.loss_fn, self.members)
            self.scaler.scale(loss / self.accumulation).backward()
        return float(loss.detach())

    def optimizer_step(self) -> None:
        """Clip, step, advance the LR/momentum schedules and update the EMA."""
        if self.config.train.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.global_step += 1
        self.schedule.step(self.global_step)
        muon = self.config.optimizer.muon
        set_muon_momentum(
            self.optimizer,
            muon_momentum_schedule(
                self.global_step,
                self.total_steps,
                warmup_steps=max(1, self.config.optimizer.schedule.warmup_steps),
                cooldown_steps=max(1, self.config.optimizer.schedule.warmup_steps // 10),
                momentum_min=muon.momentum - muon.momentum_min_offset,
                momentum_max=muon.momentum,
            ),
        )
        self.ema.update(self.model)

    def train_epoch(self) -> dict[str, float]:
        self.train_module.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)
        num_batches = len(self.train_loader)
        epoch_started = window_started = time.time()
        running, seen, total_loss, total_seen = 0.0, 0, 0.0, 0

        for index, raw_batch in enumerate(self.train_loader):
            batch = self.to_device(raw_batch)
            is_last = index == num_batches - 1
            accumulate = ((index + 1) % self.accumulation != 0) and not is_last
            loss = self.backward(batch, accumulate=accumulate)
            running += loss
            seen += 1
            total_loss += loss
            total_seen += 1
            if accumulate:
                continue

            self.optimizer_step()
            if self.global_step % max(1, self.config.train.log_every) == 0:
                lrs = self.schedule.get_last_lr()
                elapsed = max(1e-9, time.time() - window_started)
                self.log_metrics(
                    {
                        "train/loss": running / max(1, seen),
                        "train/lr": lrs[0],
                        "train/lr_max": max(lrs),
                        "train/samples_per_second": seen * self.config.data.batch_size_per_device / elapsed,
                    }
                )
                log.info(
                    "epoch %d | step %d/%d | loss %.5f | lr %.2e",
                    self.epoch,
                    self.global_step,
                    self.total_steps,
                    running / max(1, seen),
                    lrs[0],
                )
                running, seen = 0.0, 0
                window_started = time.time()
            if self.config.train.max_steps is not None and self.global_step >= self.config.train.max_steps:
                break

        return {
            "train/epoch_loss": total_loss / max(1, total_seen),
            "train/epoch_seconds": time.time() - epoch_started,
        }

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def validate(self, max_batches: int | None = None) -> dict[str, float]:
        """Roll out on the validation split and return area-weighted scores."""
        if self.val_loader is None:
            return {}
        eval_config = self.config.eval
        dataset = self.datasets["val"]
        steps = max(1, min(eval_config.rollout_steps, dataset.rollout_steps))
        step_hours = self.spec.step_hours
        selected = [s for s in (eval_config.lead_times or range(1, steps + 1)) if 1 <= s <= steps]
        metrics = MetricCollection(
            self.spec.output_variables,
            self.spec.grid,
            lead_times=[s * step_hours for s in selected],
            device=self.device,
        )

        self.model.eval()
        ensemble_size = max(1, eval_config.ensemble_size)
        limit = max_batches if max_batches is not None else eval_config.max_batches
        started = time.time()
        with self.ema.average_parameters(self.model) if eval_config.use_ema else nullcontext():
            for index, raw_batch in enumerate(self.val_loader):
                if limit is not None and index >= limit:
                    break
                batch = self.to_device(raw_batch)
                with self._autocast():
                    forecast = self.model.rollout(
                        batch,
                        steps=steps,
                        ensemble_size=ensemble_size,
                        members_per_forward=eval_config.members_per_forward,
                        use_mc_dropout=eval_config.mc_dropout and ensemble_size > 1,
                    )
                truth = batch["dynamics"].index_select(2, self.model.output_indices)
                for step in selected:
                    metrics.update(
                        step * step_hours,
                        forecast[:, :, step - 1].float(),
                        truth[:, self.spec.window + step - 1].float(),
                    )
        metrics.all_reduce()
        scores: dict[str, float] = dict(metrics.compute(prefix="val/"))
        scores["val/seconds"] = time.time() - started
        scores["val/ensemble_size"] = float(ensemble_size)
        self.train_module.train()
        return scores

    # ------------------------------------------------------------------ fit
    def fit(self) -> dict[str, float]:
        """Run the configured epochs, validating and checkpointing along the way."""
        self._maybe_init_wandb()
        config = self.config.train
        last_scores: dict[str, float] = {}
        try:
            while self.epoch < config.max_epochs:
                epoch_stats = self.train_epoch()
                self.log_metrics(epoch_stats)
                log.info("epoch %d | mean loss %.5f", self.epoch, epoch_stats["train/epoch_loss"])

                if (self.epoch + 1) % max(1, config.validate_every_epochs) == 0:
                    scores = self.validate()
                    if scores:
                        last_scores = scores
                        self.log_metrics(scores)
                        summary = " | ".join(
                            f"{key.split('avg/')[-1]}={value:.4f}"
                            for key, value in scores.items()
                            if key.startswith("val/avg/") and key.endswith("/avg")
                        )
                        log.info("epoch %d | validation | %s", self.epoch, summary)
                if self.dist.is_main and (self.epoch + 1) % max(1, config.save_every_epochs) == 0:
                    self.save(last_scores)
                barrier()
                self.epoch += 1
                if config.max_steps is not None and self.global_step >= config.max_steps:
                    log.info("Reached max_steps=%d; stopping.", config.max_steps)
                    break
        finally:
            if self._wandb is not None:
                self._wandb.finish()
                self._wandb = None
        return last_scores

    def save(self, scores: Mapping[str, float] | None = None) -> Path:
        """Write ``last.ckpt`` (with optimizer state) and, when the monitor improves, ``best.ckpt``."""
        scores = dict(scores or {})
        common = dict(
            model=self.model,
            config=self.config,
            spec=self.spec,
            normalizer=self.normalizer,
            ema=self.ema,
            epoch=self.epoch,
            global_step=self.global_step,
            metrics=scores,
        )
        last = save_checkpoint(
            self.output_dir / "last.ckpt", optimizer=self.optimizer, schedule=self.schedule, **common
        )
        value = scores.get(self.config.train.monitor)
        if value is not None:
            better = (
                value < self.best_metric if self.config.train.monitor_mode == "min" else value > self.best_metric
            )
            if better:
                self.best_metric = value
                save_checkpoint(self.output_dir / "best.ckpt", **common)
                log.info("New best %s = %.5f at epoch %d.", self.config.train.monitor, value, self.epoch)
        return last


def train(config: ExperimentConfig, **kwargs) -> dict[str, float]:
    """Convenience entry point: build a :class:`Trainer` and fit it."""
    return Trainer(config, **kwargs).fit()
