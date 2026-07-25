"""A minimal but honest training loop.

Schedule follows the paper: linear warmup then cosine decay over a *cycle*, and
the learning rate is restarted at 0.8x its previous peak for each subsequent
cycle.  The two heads can be trained jointly or separately -- the paper trains
one projector and reuses it across all three estimators, which is the cheaper
and more controlled experiment.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import WindowDataset, collate
from .model import Atlas

__all__ = ["TrainConfig", "Trainer", "cosine_cycle_lr", "EMA"]


@dataclass
class TrainConfig:
    lr: float = 1.28e-4
    warmup_steps: int = 2000
    cycle_steps: int = 100_000
    cycles: int = 3
    cycle_decay: float = 0.8
    batch_size: int = 8
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    amp_dtype: str | None = "bfloat16"
    ema_decay: float | None = 0.999
    parts: Sequence[str] = ("latent", "projector", "encoder")
    log_every: int = 50
    checkpoint_every: int = 5000
    out_dir: str = "runs/atlas"
    num_workers: int = 0
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def total_steps(self) -> int:
        return self.cycle_steps * self.cycles


def cosine_cycle_lr(step: int, cfg: TrainConfig) -> float:
    """Warmup + cosine decay, with a decayed restart at each cycle boundary."""
    cycle = min(step // cfg.cycle_steps, cfg.cycles - 1)
    peak = cfg.lr * cfg.cycle_decay**cycle
    within = step - cycle * cfg.cycle_steps
    if cycle == 0 and within < cfg.warmup_steps:
        return peak * (within + 1) / cfg.warmup_steps
    span = max(cfg.cycle_steps - (cfg.warmup_steps if cycle == 0 else 0), 1)
    t = (within - (cfg.warmup_steps if cycle == 0 else 0)) / span
    return 0.5 * peak * (1 + math.cos(math.pi * min(max(t, 0.0), 1.0)))


class EMA:
    """Exponential moving average of the parameters, evaluated at checkpoints."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float() for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow


class Trainer:
    """Single-process trainer; wrap the model in DDP externally for multi-GPU."""

    def __init__(self, model: Atlas, cfg: TrainConfig) -> None:
        self.model = model.to(cfg.device)
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        params = self._trainable_parameters()
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.ema = EMA(self.model, cfg.ema_decay) if cfg.ema_decay else None
        self.step = 0
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._amp = (
            getattr(torch, cfg.amp_dtype)
            if cfg.amp_dtype and cfg.device.startswith("cuda")
            else None
        )

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        parts = set(self.cfg.parts)
        mods = []
        if "latent" in parts:
            mods.append(self.model.backbone)
            mods.append(self.model.estimator)
        if "projector" in parts:
            mods.append(self.model.projector)
        if "encoder" in parts and self.model.resampler.is_learned:
            mods.append(self.model.resampler)
        seen: set[int] = set()
        out = []
        for m in mods:
            for p in m.parameters():
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
        return out

    # -- one step ----------------------------------------------------------

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        cfg = self.cfg
        lr = cosine_cycle_lr(self.step, cfg)
        for g in self.opt.param_groups:
            g["lr"] = lr

        window = batch["window"].to(cfg.device, non_blocking=True)
        nxt = batch["next"].to(cfg.device, non_blocking=True)
        times = batch.get("time")

        ctx = (
            torch.autocast(device_type="cuda", dtype=self._amp)
            if self._amp is not None
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with ctx:
            loss, logs = self.model.training_losses(window, nxt, times, parts=cfg.parts)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            gn = torch.nn.utils.clip_grad_norm_(
                [p for g in self.opt.param_groups for p in g["params"]], cfg.grad_clip
            )
            logs["grad_norm"] = float(gn)
        self.opt.step()
        if self.ema is not None:
            self.ema.update(self.model)
        self.step += 1
        logs["lr"] = lr
        return logs

    # -- loop --------------------------------------------------------------

    def fit(
        self,
        dataset: WindowDataset,
        max_steps: int | None = None,
        callbacks: Iterable[Any] = (),
    ) -> None:
        cfg = self.cfg
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate,
            drop_last=True,
            persistent_workers=cfg.num_workers > 0,
        )
        target = max_steps or cfg.total_steps
        t0 = time.time()
        self.model.train()
        while self.step < target:
            for batch in loader:
                logs = self.train_step(batch)
                if self.step % cfg.log_every == 0:
                    rate = self.step / max(time.time() - t0, 1e-9)
                    msg = " ".join(f"{k}={v:.4g}" for k, v in logs.items())
                    print(f"[{self.step:>7}/{target}] {msg} steps/s={rate:.2f}", flush=True)
                for cb in callbacks:
                    cb(self, logs)
                if cfg.checkpoint_every and self.step % cfg.checkpoint_every == 0:
                    self.save(self.out_dir / f"step_{self.step:08d}.pt")
                if self.step >= target:
                    break
        self.save(self.out_dir / "final.pt")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.model.cfg.to_dict(),
                "state_dict": self.model.state_dict(),
                "ema": self.ema.state_dict() if self.ema else None,
                "optimizer": self.opt.state_dict(),
                "step": self.step,
            },
            path,
        )


def fit_normalizers(
    model: Atlas, dataset: WindowDataset, max_samples: int = 512, stride: int = 1
) -> None:
    """Estimate state and residual statistics from a dataset, in place."""
    from .normalize import compute_statistics  # noqa: PLC0415

    stats = compute_statistics(
        dataset.stat_windows(stride=stride), dt_index=1, max_samples=max_samples
    )
    model.normalizers.state.set_stats(stats["state_mean"], stats["state_std"])
    model.normalizers.residual.set_stats(stats["residual_mean"], stats["residual_std"])
