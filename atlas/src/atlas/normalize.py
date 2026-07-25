"""Per-channel normalisation.

Following the paper, *separate* statistics are kept for states and for
residuals: a 6-hour geopotential increment is two orders of magnitude smaller
than the field itself, so sharing statistics would leave the residual target
far from unit variance and starve the probabilistic head of signal.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .spec import VariableSet

__all__ = ["ChannelNormalizer", "NormalizerPair", "compute_statistics"]


class ChannelNormalizer(nn.Module):
    """Affine per-channel standardisation with buffered statistics."""

    def __init__(self, n_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(1, n_channels, 1, 1))
        self.register_buffer("std", torch.ones(1, n_channels, 1, 1))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    def set_stats(self, mean, std) -> ChannelNormalizer:
        mean = torch.as_tensor(np.asarray(mean), dtype=torch.float32).reshape(1, -1, 1, 1)
        std = torch.as_tensor(np.asarray(std), dtype=torch.float32).reshape(1, -1, 1, 1)
        if mean.shape != self.mean.shape:
            raise ValueError(
                f"expected stats of shape {tuple(self.mean.shape)}, "
                f"got {tuple(mean.shape)}"
            )
        self.mean.copy_(mean)
        self.std.copy_(std)
        self.fitted.fill_(True)
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.dtype)) / (self.std.to(x.dtype) + self.eps)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std.to(x.dtype) + self.eps) + self.mean.to(x.dtype)

    def scale_only(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise a difference of states (mean cancels)."""
        return x / (self.std.to(x.dtype) + self.eps)

    forward = normalize


class NormalizerPair(nn.Module):
    """The ``(state, residual)`` statistics pair used throughout the model."""

    def __init__(self, n_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.state = ChannelNormalizer(n_channels, eps)
        self.residual = ChannelNormalizer(n_channels, eps)

    @property
    def fitted(self) -> bool:
        return bool(self.state.fitted) and bool(self.residual.fitted)

    def save(self, path: str | Path, variables: VariableSet | None = None) -> None:
        d = {
            "state_mean": self.state.mean.squeeze().cpu().numpy(),
            "state_std": self.state.std.squeeze().cpu().numpy(),
            "residual_mean": self.residual.mean.squeeze().cpu().numpy(),
            "residual_std": self.residual.std.squeeze().cpu().numpy(),
        }
        if variables is not None:
            d["names"] = np.array(variables.names)
        np.savez(path, **d)

    def load(self, path: str | Path, variables: VariableSet | None = None) -> NormalizerPair:
        d = np.load(path, allow_pickle=False)
        if variables is not None and "names" in d:
            stored = [str(s) for s in d["names"]]
            if stored != variables.names:
                idx = [stored.index(n) for n in variables.names]
                self.state.set_stats(d["state_mean"][idx], d["state_std"][idx])
                self.residual.set_stats(d["residual_mean"][idx], d["residual_std"][idx])
                return self
        self.state.set_stats(d["state_mean"], d["state_std"])
        self.residual.set_stats(d["residual_mean"], d["residual_std"])
        return self


def compute_statistics(
    samples: Iterable[np.ndarray], dt_index: int = 1, max_samples: int | None = None
) -> dict[str, np.ndarray]:
    """Streaming per-channel mean/std for states and for temporal residuals.

    ``samples`` yields ``(T, C, H, W)`` windows with ``T >= dt_index + 1``;
    residual statistics are taken from ``x[dt_index] - x[dt_index - 1]``.
    Uses Welford accumulation so a full training archive can be streamed.
    """
    acc: dict[str, list] = {}
    for k, window in enumerate(samples):
        if max_samples is not None and k >= max_samples:
            break
        w = np.asarray(window, dtype=np.float64)
        parts = {"state": w[dt_index], "residual": w[dt_index] - w[dt_index - 1]}
        for key, arr in parts.items():
            flat = arr.reshape(arr.shape[0], -1)
            nb = flat.shape[1]
            mb = flat.mean(axis=1)
            m2b = ((flat - mb[:, None]) ** 2).sum(axis=1)
            if key not in acc:
                acc[key] = [0, np.zeros(flat.shape[0]), np.zeros(flat.shape[0])]
            na, ma, m2a = acc[key]
            nt = na + nb
            delta = mb - ma
            acc[key] = [
                nt,
                ma + delta * nb / nt,
                m2a + m2b + delta**2 * na * nb / nt,
            ]
    if not acc:
        raise ValueError("no samples provided")
    out = {}
    for key, (n, mean, m2) in acc.items():
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = np.sqrt(m2 / max(n - 1, 1))
    return out
