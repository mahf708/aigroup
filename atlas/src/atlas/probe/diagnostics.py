"""Diagnostics for the latent space itself.

The paper's Section 5 argues for bilinear downsampling over a learned encoder
on the grounds that the resulting latent has better-behaved *spectra* and
respects temporal continuity.  Those two claims are directly measurable, and
the functions here measure them, so an alternative encoder can be evaluated on
the same terms rather than only on downstream forecast skill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..grids import SphericalTransform
from ..losses import Weights, spatial_rms, weighted_mean
from ..spec import GridSpec

__all__ = [
    "power_spectrum",
    "spectral_ratio",
    "latent_continuity",
    "latent_eof",
    "trajectory_summary",
    "reconstruction_report",
]


def power_spectrum(
    x: torch.Tensor, grid: GridSpec, transform: SphericalTransform | None = None
) -> torch.Tensor:
    """Angular power spectrum per channel: ``(..., C, n_wavenumber)``."""
    tr = transform or SphericalTransform(grid)
    lead = x.shape[:-3]
    flat = x.reshape(-1, *x.shape[-3:])
    p = tr.power(flat)
    return p.reshape(*lead, *p.shape[-2:])


def spectral_ratio(
    fine: torch.Tensor,
    reconstructed: torch.Tensor,
    grid: GridSpec,
    transform: SphericalTransform | None = None,
    rel_floor: float = 1e-8,
) -> torch.Tensor:
    """Ratio of reconstructed to true power by wavenumber.

    A value near 1 out to the latent cutoff and a clean roll-off beyond it is
    the signature of a well-behaved encoder.  A long flat tail at high
    wavenumbers is the spectral pathology the paper reports for VAE-style
    latents.
    """
    tr = transform or SphericalTransform(grid)
    ref = power_spectrum(fine, grid, tr)
    got = power_spectrum(reconstructed, grid, tr)
    ratio = got / ref.clamp_min(1e-30)
    # Bins whose reference power is pure round-off carry no information; a
    # band-limited field has many of them and they would otherwise dominate
    # any average over wavenumber.
    floor = rel_floor * ref.amax(dim=-1, keepdim=True)
    return torch.where(ref > floor, ratio, torch.full_like(ratio, float("nan")))


def latent_continuity(latents: torch.Tensor, weights: Weights | None = None) -> dict[str, float]:
    """Is the latent temporally smooth?

    ``latents`` is ``(T, C, h, w)`` for consecutive times.  Reports the mean
    step size relative to the spread of the whole trajectory and the lag-1
    autocorrelation of the increments.  A learned encoder that maps adjacent
    states far apart shows up here as a step ratio close to 1.
    """
    d = latents[1:] - latents[:-1]
    step = weighted_mean(d**2, weights).sqrt().mean()
    total = weighted_mean(
        (latents - latents.mean(0, keepdim=True)) ** 2, weights
    ).sqrt().mean()
    d_flat = d.reshape(d.shape[0], -1)
    dn = d_flat - d_flat.mean(0, keepdim=True)
    denom = (dn[:-1].pow(2).sum() * dn[1:].pow(2).sum()).sqrt().clamp_min(1e-12)
    lag1 = float((dn[:-1] * dn[1:]).sum() / denom)
    return {
        "step_rms": float(step),
        "trajectory_rms": float(total),
        "step_ratio": float(step / total.clamp_min(1e-12)),
        "increment_lag1_autocorr": lag1,
    }


@dataclass
class EOFResult:
    modes: torch.Tensor  # (k, C, h, w)
    variance_fraction: torch.Tensor  # (k,)
    coefficients: torch.Tensor  # (n, k)


def latent_eof(latents: torch.Tensor, k: int = 8, area: torch.Tensor | None = None) -> EOFResult:
    """Area-weighted EOFs of a set of latent fields.

    Applied to a *time series* these are the model's dominant modes of
    variability; applied to an *ensemble at fixed time* they are the directions
    the probabilistic head chooses to disagree along -- which is the more
    interesting object, and rarely the same thing.
    """
    n = latents.shape[0]
    x = latents.reshape(n, -1).float()
    if area is not None:
        w = area.reshape(1, 1, -1, 1).sqrt().to(x.device)
        x = (latents * w).reshape(n, -1).float()
    mu = x.mean(0, keepdim=True)
    xc = x - mu
    k = min(k, min(xc.shape) - 1) if min(xc.shape) > 1 else 1
    u, s, v = torch.pca_lowrank(xc, q=k, center=False)
    var = s**2 / (xc.pow(2).sum()).clamp_min(1e-12)
    return EOFResult(
        modes=v[:, :k].T.reshape(k, *latents.shape[1:]),
        variance_fraction=var[:k],
        coefficients=u[:, :k] * s[:k],
    )


def trajectory_summary(
    trajectory: torch.Tensor, weights: Weights | None = None
) -> dict[str, torch.Tensor]:
    """Summarise an SDE sampling path in latent space.

    ``trajectory`` is ``(n_steps + 1, B, C, h, w)`` from
    ``predict_latent(..., record_trajectory=True)``.  The per-step displacement
    tells you where in the integration the sample actually acquires its
    structure -- for the interpolant this is typically concentrated near
    ``t = 1``, and a schedule that spends its steps elsewhere is wasting them.
    """
    d = trajectory[1:] - trajectory[:-1]
    step = weighted_mean(d**2, weights).sqrt()
    total = weighted_mean(
        (trajectory[-1] - trajectory[0]) ** 2, weights
    ).sqrt()
    cum = step.cumsum(0)
    return {
        "step_rms": step.mean(-1) if step.ndim > 1 else step,
        "total_rms": total.mean(-1) if total.ndim > 1 else total,
        "cumulative_fraction": (cum / cum[-1:].clamp_min(1e-12)).mean(-1),
    }


@torch.no_grad()
def reconstruction_report(
    model, x_cur: torch.Tensor, x_next: torch.Tensor, times: np.ndarray | None = None
) -> dict[str, torch.Tensor]:
    """How much error is the *encoder/decoder* responsible for?

    Compares three reconstructions of the true increment: naive bilinear
    upsampling of the latent residual, the learned projector given the true
    latent residual, and the identity (persistence).  The paper's argument for
    the latent approach rests on the projector's error being an order of
    magnitude below the predictive model's, and this is the check for it.
    """
    truth = x_next - x_cur
    r = model.encode_residual(x_next, x_cur)
    area = model.fine_area

    naive = model.normalizers.residual.denormalize(model.resampler.upsample(r))
    learned = model.decode(r, x_cur, times)

    def rmse(a: torch.Tensor) -> torch.Tensor:
        return spatial_rms(a - truth, area).mean(0)

    return {
        "projector_rmse": rmse(learned),
        "bilinear_rmse": rmse(naive),
        "persistence_rmse": rmse(torch.zeros_like(truth)),
        "channel_names": model.cfg.variables.names,
    }
