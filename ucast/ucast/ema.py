"""Exponential moving average of model weights.

U-Cast evaluates and forecasts with EMA weights (decay 0.9999).  The averaged weights are what the
reported skill numbers refer to, so this is not an optional nicety.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn

__all__ = ["ExponentialMovingAverage"]


class ExponentialMovingAverage:
    """Shadow copy of a model's floating-point parameters, updated in place after every step.

    Args:
        model: The model to track.
        decay: EMA decay.  1.0 disables averaging (the shadow simply mirrors the live weights).
        warmup: For the first steps, use ``min(decay, (1 + n) / (10 + n))`` so the average is not
            dominated by the random initialisation.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup: bool = True):
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone().float() for name, param in model.named_parameters() if param.requires_grad
        }
        self._backup: dict[str, torch.Tensor] = {}

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = self.current_decay()
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(param.detach().float(), 1.0 - decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite the model's parameters with the averaged ones."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name].to(param.dtype))

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        self._backup = {
            name: param.detach().clone() for name, param in model.named_parameters() if name in self.shadow
        }

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self._backup:
                param.copy_(self._backup[name])
        self._backup = {}

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap in the averaged weights (used for validation and inference)."""
        self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)

    def to(self, device) -> "ExponentialMovingAverage":
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "warmup": self.warmup, "num_updates": self.num_updates, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, object], strict: bool = True) -> None:
        shadow = state["shadow"]
        if not isinstance(shadow, dict):  # pragma: no cover - defensive
            raise TypeError("EMA state_dict['shadow'] must be a dict of tensors")
        missing = set(self.shadow) - set(shadow)
        unexpected = set(shadow) - set(self.shadow)
        if strict and (missing or unexpected):
            raise KeyError(f"EMA state mismatch. missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}")
        for name, tensor in shadow.items():
            if name in self.shadow:
                self.shadow[name] = tensor.to(self.shadow[name].device).float()
        self.decay = float(state.get("decay", self.decay))
        self.warmup = bool(state.get("warmup", self.warmup))
        self.num_updates = int(state.get("num_updates", 0))
