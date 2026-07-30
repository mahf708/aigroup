"""Streaming, area-weighted verification metrics.

Everything is accumulated as running sums so a full evaluation pass over hundreds of initial
conditions never holds more than one batch of forecasts in memory, and so metrics can be reduced
across ranks with a single all-reduce.

Definitions follow WeatherBench-2 (Rasp et al., 2024), which is what makes the numbers comparable to
the public leaderboard:

* ``rmse``   -- area-weighted RMSE of the ensemble mean,
* ``crps``   -- fair ensemble CRPS,
* ``spread`` -- square root of the area-weighted mean ensemble variance,
* ``ssr``    -- spread/skill ratio ``sqrt((M+1)/M) * spread / rmse``; 1 is calibrated, below 1
  under-dispersive,
* ``mae``, ``bias`` -- of the ensemble mean.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor

from .grid import Grid
from .losses import crps_ensemble
from .variables import VariableSet

__all__ = ["ForecastMetrics", "MetricCollection"]

_ACCUMULATORS = ("sq_error", "abs_error", "error", "crps", "variance", "count")


class ForecastMetrics:
    """Accumulates per-variable scores for one lead time.

    Args:
        variables: Channel definitions; scores are reported per ``variable.key``.
        grid: Provides the latitude area weights.
        device / dtype: Where to accumulate (float64 by default, so long runs do not drift).
    """

    def __init__(
        self,
        variables: VariableSet,
        grid: Grid,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.variables = variables
        self.grid = grid
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self._weights = grid.area_weights(dtype=dtype, device=self.device)  # (H, 1)
        self.num_members: int | None = None
        self._sums: dict[str, Tensor] = {}
        self.reset()

    def reset(self) -> None:
        zeros = torch.zeros(len(self.variables), dtype=self.dtype, device=self.device)
        self._sums = {name: zeros.clone() for name in _ACCUMULATORS if name != "count"}
        self._sums["count"] = torch.zeros((), dtype=self.dtype, device=self.device)
        self.num_members = None

    def to(self, device: torch.device | str) -> "ForecastMetrics":
        self.device = torch.device(device)
        self._weights = self._weights.to(self.device)
        self._sums = {k: v.to(self.device) for k, v in self._sums.items()}
        return self

    def _spatial_mean(self, x: Tensor) -> Tensor:
        """Area-weighted mean over ``(H, W)``, summed over any leading axes -> ``(C,)``."""
        per_sample = (x.to(self.dtype) * self._weights).mean(dim=(-2, -1))  # (..., C)
        return per_sample.reshape(-1, per_sample.shape[-1]).sum(dim=0)

    @torch.no_grad()
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Accumulate one batch.

        Args:
            predictions: ``(M, B, C, H, W)`` ensemble forecasts, or ``(B, C, H, W)`` for a single member.
            targets: ``(B, C, H, W)`` ground truth, in the same (physical) units as ``predictions``.
        """
        if predictions.ndim == targets.ndim:
            predictions = predictions.unsqueeze(0)
        if predictions.ndim != 5 or targets.ndim != 4:
            raise ValueError(
                f"expected (M,B,C,H,W) and (B,C,H,W), got {tuple(predictions.shape)} / {tuple(targets.shape)}"
            )
        if predictions.shape[1:] != targets.shape:
            raise ValueError(f"prediction/target mismatch: {tuple(predictions.shape)} vs {tuple(targets.shape)}")
        if predictions.shape[2] != len(self.variables):
            raise ValueError(f"expected {len(self.variables)} channels, got {predictions.shape[2]}")

        predictions = predictions.to(device=self.device, dtype=self.dtype)
        targets = targets.to(device=self.device, dtype=self.dtype)
        members = predictions.shape[0]
        if self.num_members is None:
            self.num_members = members
        elif self.num_members != members:
            raise ValueError(f"ensemble size changed from {self.num_members} to {members}")

        mean = predictions.mean(dim=0)
        error = mean - targets
        self._sums["sq_error"] += self._spatial_mean(error**2)
        self._sums["abs_error"] += self._spatial_mean(error.abs())
        self._sums["error"] += self._spatial_mean(error)
        self._sums["crps"] += self._spatial_mean(crps_ensemble(predictions, targets))
        if members > 1:
            self._sums["variance"] += self._spatial_mean(predictions.var(dim=0, unbiased=True))
        self._sums["count"] += targets.shape[0]

    def all_reduce(self, group=None) -> None:
        """Sum the accumulators across distributed ranks (no-op when not distributed)."""
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return
        for tensor in self._sums.values():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)

    def compute(self, prefix: str = "") -> "OrderedDict[str, float]":
        """Per-variable and averaged scores as a flat ``{name: value}`` mapping."""
        count = float(self._sums["count"].item())
        if count == 0:
            return OrderedDict()
        members = self.num_members or 1
        rmse = torch.sqrt(self._sums["sq_error"] / count)
        mae = self._sums["abs_error"] / count
        bias = self._sums["error"] / count
        crps = self._sums["crps"] / count
        spread = torch.sqrt(self._sums["variance"] / count)
        inflation = ((members + 1) / members) ** 0.5
        ssr = inflation * spread / rmse.clamp_min(torch.finfo(torch.float64).tiny)

        out: "OrderedDict[str, float]" = OrderedDict()
        per_metric = {"rmse": rmse, "mae": mae, "bias": bias, "crps": crps}
        if members > 1:
            per_metric["spread"] = spread
            per_metric["ssr"] = ssr
        for metric, values in per_metric.items():
            for i, var in enumerate(self.variables):
                out[f"{prefix}{metric}/{var.key}"] = float(values[i].item())
            out[f"{prefix}{metric}/avg"] = float(values.mean().item())
        out[f"{prefix}num_members"] = float(members)
        out[f"{prefix}count"] = count
        return out


class MetricCollection:
    """One :class:`ForecastMetrics` per lead time, keyed by lead time in hours."""

    def __init__(
        self,
        variables: VariableSet,
        grid: Grid,
        lead_times: list[int],
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.lead_times = list(lead_times)
        self.metrics: "OrderedDict[int, ForecastMetrics]" = OrderedDict(
            (lead, ForecastMetrics(variables, grid, device=device, dtype=dtype)) for lead in self.lead_times
        )

    def __getitem__(self, lead_time: int) -> ForecastMetrics:
        return self.metrics[lead_time]

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()

    def to(self, device) -> "MetricCollection":
        for metric in self.metrics.values():
            metric.to(device)
        return self

    def update(self, lead_time: int, predictions: Tensor, targets: Tensor) -> None:
        self.metrics[lead_time].update(predictions, targets)

    def all_reduce(self, group=None) -> None:
        for metric in self.metrics.values():
            metric.all_reduce(group)

    def compute(self, prefix: str = "") -> "OrderedDict[str, float]":
        """Flat metrics for every lead time, plus lead-time averages under ``<prefix>avg/``.

        Keys look like ``val/t24/rmse/z500`` and ``val/avg/rmse/z500``.
        """
        out: "OrderedDict[str, float]" = OrderedDict()
        per_key: dict[str, list[float]] = {}
        for lead, metric in self.metrics.items():
            scores = metric.compute(prefix=f"{prefix}t{lead}/")
            out.update(scores)
            for key, value in scores.items():
                base = key[len(f"{prefix}t{lead}/") :]
                if base in ("count", "num_members"):
                    continue
                per_key.setdefault(base, []).append(value)
        for base, values in per_key.items():
            out[f"{prefix}avg/{base}"] = float(sum(values) / len(values))
        return out
