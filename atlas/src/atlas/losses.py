"""Weighted reductions, training losses and verification metrics.

Metric definitions follow Section 3.2 of the paper: area weights proportional
to ``sin(lat_u) - sin(lat_l)`` and normalised to one, the *fair* (unbiased)
CRPS estimator, ensemble-mean RMSE and the spread-skill ratio.
"""

from __future__ import annotations

import torch

__all__ = [
    "Weights",
    "weighted_mean",
    "spatial_mean",
    "spatial_rms",
    "l1_loss",
    "l2_loss",
    "fair_crps",
    "ensemble_rmse",
    "ensemble_spread",
    "spread_skill_ratio",
    "acc",
]


class Weights:
    """Area and channel weights applied to any ``(..., C, H, W)`` tensor."""

    def __init__(
        self,
        area: torch.Tensor | None = None,
        channel: torch.Tensor | None = None,
    ) -> None:
        self.area = area
        self.channel = channel

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> Weights:
        a = self.area.to(device=device, dtype=dtype) if self.area is not None else None
        c = self.channel.to(device=device, dtype=dtype) if self.channel is not None else None
        return Weights(a, c)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.area is not None:
            x = x * self.area.reshape(*([1] * (x.ndim - 2)), -1, 1).to(x.device, x.dtype)
        if self.channel is not None:
            x = x * self.channel.reshape(*([1] * (x.ndim - 3)), -1, 1, 1).to(x.device, x.dtype)
        return x

    @property
    def denom(self) -> float:
        """Sum of the applied weights, so ``weighted_mean`` stays a mean."""
        d = 1.0
        if self.area is not None:
            d *= float(self.area.sum())
        if self.channel is not None:
            d *= float(self.channel.mean())
        return d


def spatial_mean(x: torch.Tensor, area: torch.Tensor | None = None) -> torch.Tensor:
    """Area-weighted mean over ``(H, W)`` only, keeping the channel axis.

    Use this whenever the answer should be *per variable* -- diagnostics,
    response matrices, per-channel verification -- as opposed to
    :func:`weighted_mean`, which collapses channels too.
    """
    if area is None:
        return x.mean(dim=(-2, -1))
    w = area.reshape(*([1] * (x.ndim - 2)), -1, 1).to(x.device, x.dtype)
    return (x * w).sum(dim=(-2, -1)) / (float(area.sum()) * x.shape[-1])


def spatial_rms(x: torch.Tensor, area: torch.Tensor | None = None) -> torch.Tensor:
    """Per-channel area-weighted RMS."""
    return spatial_mean(x**2, area).sqrt()


def weighted_mean(x: torch.Tensor, weights: Weights | None = None) -> torch.Tensor:
    """Mean over ``(C, H, W)``, leaving leading dimensions intact."""
    if weights is None:
        return x.mean(dim=(-3, -2, -1))
    w = weights.apply(x)
    # area weights sum to 1 over lat, so divide only by nlon and the channel mean
    denom = x.shape[-1]
    if weights.area is None:
        denom *= x.shape[-2]
    scale = weights.channel.mean() if weights.channel is not None else 1.0
    return w.sum(dim=(-2, -1)).mean(dim=-1) / (denom * scale)


def l1_loss(pred: torch.Tensor, target: torch.Tensor, weights: Weights | None = None):
    return weighted_mean((pred - target).abs(), weights).mean()


def l2_loss(pred: torch.Tensor, target: torch.Tensor, weights: Weights | None = None):
    return weighted_mean((pred - target) ** 2, weights).mean()


# ---------------------------------------------------------------------------
# probabilistic verification
# ---------------------------------------------------------------------------


def fair_crps(
    ensemble: torch.Tensor, truth: torch.Tensor, weights: Weights | None = None
) -> torch.Tensor:
    """Fair CRPS with ensemble on dim 0.

    ``ensemble``: ``(M, ..., C, H, W)``; ``truth``: ``(..., C, H, W)``.
    The second term uses the unbiased ``1 / (M (M - 1))`` normalisation, so the
    score is comparable across ensemble sizes.
    """
    m = ensemble.shape[0]
    if m < 2:
        raise ValueError("fair CRPS needs at least 2 members")
    skill = (ensemble - truth.unsqueeze(0)).abs().mean(0)
    diff = (ensemble.unsqueeze(0) - ensemble.unsqueeze(1)).abs()
    spread = diff.sum(dim=(0, 1)) / (m * (m - 1))
    return weighted_mean(skill - 0.5 * spread, weights)


def ensemble_rmse(
    ensemble: torch.Tensor, truth: torch.Tensor, weights: Weights | None = None
) -> torch.Tensor:
    err = (ensemble.mean(0) - truth) ** 2
    return weighted_mean(err, weights).sqrt()


def ensemble_spread(ensemble: torch.Tensor, weights: Weights | None = None) -> torch.Tensor:
    var = ensemble.var(dim=0, unbiased=True)
    return weighted_mean(var, weights).sqrt()


def spread_skill_ratio(
    ensemble: torch.Tensor, truth: torch.Tensor, weights: Weights | None = None
) -> torch.Tensor:
    m = ensemble.shape[0]
    scale = ((m + 1) / m) ** 0.5
    return scale * ensemble_spread(ensemble, weights) / ensemble_rmse(ensemble, truth, weights)


def acc(
    pred: torch.Tensor,
    truth: torch.Tensor,
    climatology: torch.Tensor,
    weights: Weights | None = None,
) -> torch.Tensor:
    """Area-weighted anomaly correlation coefficient."""
    a = pred - climatology
    b = truth - climatology
    num = weighted_mean(a * b, weights)
    den = (weighted_mean(a * a, weights) * weighted_mean(b * b, weights)).sqrt()
    return num / den.clamp_min(1e-12)
