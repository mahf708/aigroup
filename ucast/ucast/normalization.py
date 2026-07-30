"""Per-channel normalisation, including the residual scaling U-Cast trains against.

The model predicts the *increment* from the last input frame rather than the next state.  Increments
are far smaller than the states themselves, so they get their own standard deviation: the network sees
targets scaled by ``std / residual_std`` (order 1 for every variable, whether it is a fast-varying
surface field or a slowly-drifting stratospheric one) and its output is scaled back before being added
to the previous state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .variables import VariableSet

__all__ = ["Normalizer", "compute_statistics"]


def _as_channel_tensor(values, num_channels: int, name: str) -> Tensor:
    tensor = torch.as_tensor(np.asarray(values, dtype=np.float64), dtype=torch.float32).reshape(-1)
    if tensor.numel() == 1:
        tensor = tensor.expand(num_channels).clone()
    if tensor.numel() != num_channels:
        raise ValueError(f"{name} has {tensor.numel()} entries but {num_channels} channels were expected")
    return tensor


class Normalizer(nn.Module):
    """Channel-wise standardisation with an optional residual scale.

    Statistics are registered as buffers, so they follow the model across devices and land in the
    checkpoint.

    Args:
        variables: The channel axis these statistics describe.
        mean / std: Per-channel statistics of the state, shape ``(C,)``.
        residual_std: Per-channel standard deviation of the one-step increment, shape ``(C,)``.  When
            omitted, residual scaling is the identity and the model predicts normalised increments
            directly.
        eps: Floor applied to standard deviations to keep constant channels finite.
    """

    def __init__(
        self,
        variables: VariableSet,
        mean,
        std,
        residual_std=None,
        eps: float = 1e-8,
    ):
        super().__init__()
        num_channels = len(variables)
        self.variables = variables
        self.eps = eps
        mean_t = _as_channel_tensor(mean, num_channels, "mean")
        std_t = _as_channel_tensor(std, num_channels, "std").clamp_min(eps)
        self.register_buffer("mean", mean_t.reshape(num_channels, 1, 1))
        self.register_buffer("std", std_t.reshape(num_channels, 1, 1))
        if residual_std is None:
            residual = torch.ones_like(std_t)
        else:
            residual = _as_channel_tensor(residual_std, num_channels, "residual_std").clamp_min(eps)
        self.has_residual_std = residual_std is not None
        self.register_buffer("residual_std", residual.reshape(num_channels, 1, 1))

    # ------------------------------------------------------------------ state <-> normalised state
    def normalize(self, x: Tensor, channels: Sequence[int] | None = None) -> Tensor:
        mean, std = self._stats(("mean", "std"), channels)
        return (x - mean) / std

    def denormalize(self, x: Tensor, channels: Sequence[int] | None = None) -> Tensor:
        mean, std = self._stats(("mean", "std"), channels)
        return x * std + mean

    # ------------------------------------------------------------------ normalised <-> residual scale
    def to_residual_scale(self, increment: Tensor, channels: Sequence[int] | None = None) -> Tensor:
        """Scale a *normalised* increment to the unit-variance space the network is trained in."""
        std, residual_std = self._stats(("std", "residual_std"), channels)
        return increment * (std / residual_std)

    def from_residual_scale(self, increment: Tensor, channels: Sequence[int] | None = None) -> Tensor:
        """Inverse of :meth:`to_residual_scale`: network output -> normalised increment."""
        std, residual_std = self._stats(("std", "residual_std"), channels)
        return increment * (residual_std / std)

    def _stats(self, names: Iterable[str], channels: Sequence[int] | None) -> tuple[Tensor, ...]:
        out = []
        index = None if channels is None else torch.as_tensor(list(channels), device=self.mean.device)
        for name in names:
            buffer: Tensor = getattr(self, name)
            out.append(buffer if index is None else buffer.index_select(0, index))
        return tuple(out)

    def subset(self, variables: VariableSet) -> "Normalizer":
        """A normalizer restricted (and reordered) to ``variables``."""
        idx = variables.indices_in(self.variables)
        index = torch.as_tensor(idx)
        return Normalizer(
            variables,
            mean=self.mean.flatten().index_select(0, index),
            std=self.std.flatten().index_select(0, index),
            residual_std=self.residual_std.flatten().index_select(0, index) if self.has_residual_std else None,
            eps=self.eps,
        )

    # ------------------------------------------------------------------ I/O
    def save(self, path: str | Path) -> None:
        """Write statistics to a ``.npz`` file (dependency-free and human-inspectable)."""
        np.savez(
            Path(path),
            keys=np.asarray(self.variables.keys),
            mean=self.mean.flatten().cpu().numpy(),
            std=self.std.flatten().cpu().numpy(),
            residual_std=self.residual_std.flatten().cpu().numpy(),
            has_residual_std=np.asarray(self.has_residual_std),
        )

    @classmethod
    def load(cls, path: str | Path, variables: VariableSet | None = None) -> "Normalizer":
        """Read statistics written by :meth:`save`, optionally re-selecting ``variables``."""
        with np.load(Path(path), allow_pickle=False) as data:
            stored = VariableSet([str(k) for k in data["keys"]])
            normalizer = cls(
                stored,
                mean=data["mean"],
                std=data["std"],
                residual_std=data["residual_std"] if bool(data["has_residual_std"]) else None,
            )
        return normalizer if variables is None else normalizer.subset(variables)

    @classmethod
    def from_mapping(
        cls,
        variables: VariableSet,
        mean: Mapping[str, float],
        std: Mapping[str, float],
        residual_std: Mapping[str, float] | None = None,
    ) -> "Normalizer":
        """Build from ``{variable_key: value}`` mappings, e.g. parsed from a stats NetCDF file."""

        def gather(source: Mapping[str, float], what: str) -> list[float]:
            missing = [v.key for v in variables if v.key not in source]
            if missing:
                raise KeyError(f"{what} is missing entries for: {missing[:8]}{'...' if len(missing) > 8 else ''}")
            return [float(source[v.key]) for v in variables]

        return cls(
            variables,
            mean=gather(mean, "mean"),
            std=gather(std, "std"),
            residual_std=None if residual_std is None else gather(residual_std, "residual_std"),
        )

    @classmethod
    def identity(cls, variables: VariableSet) -> "Normalizer":
        """A no-op normalizer, useful for tests and for pre-normalised data."""
        n = len(variables)
        return cls(variables, mean=np.zeros(n), std=np.ones(n))

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"channels={len(self.variables)}, has_residual_std={self.has_residual_std}"


@torch.no_grad()
def compute_statistics(
    dataset,
    variables: VariableSet,
    max_samples: int | None = 512,
    time_axis_key: str = "dynamics",
    residual_step: int = 1,
) -> Normalizer:
    """Estimate per-channel mean/std and one-step increment std by streaming over a dataset.

    Handy when running U-Cast on data that has no published statistics file (E3SM output, a regional
    subset, a new variable set).  Uses Welford-free two-pass-free accumulation of raw moments, which is
    plenty accurate in float64.

    Args:
        dataset: Anything indexable yielding ``{time_axis_key: (T, C, H, W)}`` in physical units.
        variables: Channel definitions matching ``C``.
        max_samples: Cap on the number of dataset items visited (``None`` for all of them).
        time_axis_key: Key of the state tensor inside each item.
        residual_step: Time offset used for the increment statistics.
    """
    num_channels = len(variables)
    total = torch.zeros(num_channels, dtype=torch.float64)
    total_sq = torch.zeros(num_channels, dtype=torch.float64)
    res_total = torch.zeros(num_channels, dtype=torch.float64)
    res_total_sq = torch.zeros(num_channels, dtype=torch.float64)
    count = 0.0
    res_count = 0.0

    limit = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    if limit == 0:
        raise ValueError("cannot compute statistics from an empty dataset")
    for i in range(limit):
        item = dataset[i]
        state = torch.as_tensor(item[time_axis_key]).to(torch.float64)
        if state.ndim != 4 or state.shape[1] != num_channels:
            raise ValueError(f"expected (T, {num_channels}, H, W) states, got {tuple(state.shape)}")
        flat = state.transpose(0, 1).reshape(num_channels, -1)
        total += flat.sum(dim=1)
        total_sq += (flat**2).sum(dim=1)
        count += flat.shape[1]
        if state.shape[0] > residual_step:
            increment = state[residual_step:] - state[:-residual_step]
            flat_inc = increment.transpose(0, 1).reshape(num_channels, -1)
            res_total += flat_inc.sum(dim=1)
            res_total_sq += (flat_inc**2).sum(dim=1)
            res_count += flat_inc.shape[1]

    mean = total / count
    std = (total_sq / count - mean**2).clamp_min(0).sqrt()
    residual_std = None
    if res_count > 0:
        res_mean = res_total / res_count
        residual_std = (res_total_sq / res_count - res_mean**2).clamp_min(0).sqrt()
    return Normalizer(variables, mean=mean, std=std, residual_std=residual_std)
