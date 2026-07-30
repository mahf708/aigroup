"""Per-step learning-rate schedule.

A plain callable rather than a ``torch.optim.lr_scheduler`` subclass: it works with the composite
optimizer, needs no state beyond the step counter, and resumes exactly by being told the step number.
"""

from __future__ import annotations

import math

__all__ = ["WarmupCosineSchedule", "WarmupConstantSchedule", "build_schedule"]


class WarmupCosineSchedule:
    """Linear warmup followed by cosine decay, applied multiplicatively to each group's base LR.

    Both U-Cast stages use this with 1500 warmup steps; Stage 2 simply spans far fewer total steps.

    Args:
        optimizer: Any object exposing ``param_groups`` (including :class:`HybridOptimizer`).
        total_steps: Number of optimizer steps in the run; the cosine reaches ``min_lr_ratio`` here.
        warmup_steps: Steps spent ramping from ``start_lr_ratio`` to the base LR.
        start_lr_ratio / min_lr_ratio: Fractions of the base LR at step 0 and at ``total_steps``.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        start_lr_ratio: float = 0.0,
        min_lr_ratio: float = 0.0,
    ):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = max(0, min(int(warmup_steps), self.total_steps))
        self.start_lr_ratio = start_lr_ratio
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_step = 0
        self.step(0)

    def multiplier(self, step: int) -> float:
        if step < self.warmup_steps:
            progress = step / max(1, self.warmup_steps)
            return self.start_lr_ratio + progress * (1.0 - self.start_lr_ratio)
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step(self, step: int | None = None) -> float:
        """Set the LR for ``step`` (defaults to the next step) and return the multiplier used."""
        self.last_step = self.last_step + 1 if step is None else int(step)
        multiplier = self.multiplier(self.last_step)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * multiplier
        return multiplier

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {"last_step": self.last_step, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict) -> None:
        self.base_lrs = list(state.get("base_lrs", self.base_lrs))
        self.step(int(state["last_step"]))


class WarmupConstantSchedule(WarmupCosineSchedule):
    """Linear warmup, then a constant LR.  Useful for short debugging runs."""

    def multiplier(self, step: int) -> float:
        if step < self.warmup_steps:
            progress = step / max(1, self.warmup_steps)
            return self.start_lr_ratio + progress * (1.0 - self.start_lr_ratio)
        return 1.0


def build_schedule(name: str, optimizer, total_steps: int, **kwargs) -> WarmupCosineSchedule:
    schedules = {"cosine": WarmupCosineSchedule, "constant": WarmupConstantSchedule}
    key = name.strip().lower()
    if key not in schedules:
        raise ValueError(f"Unknown schedule {name!r}; available: {sorted(schedules)}")
    return schedules[key](optimizer, total_steps=total_steps, **kwargs)
