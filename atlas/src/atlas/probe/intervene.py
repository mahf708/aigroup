"""Causal interventions.

Probes are correlational.  To claim the model *uses* a piece of information you
have to change it and watch the forecast move.  ATLAS makes that unusually
tractable, because the bottleneck between "global dynamics" and "local
physics" is an explicit, named, physically-scaled field: the latent residual.
Editing latent channel ``q850`` is a well-posed physical question in a way that
editing a VAE code is not.

Three intervention points are exposed:

``latent_channel_response``
    Perturb one latent channel, measure the response in every output channel.
    Sweeping over channels builds a directed influence matrix -- an empirical
    read-out of which variables the projector believes are coupled.
``patch_site``
    Overwrite or edit any recorded activation site mid-forward.
``steer``
    Add a direction (an SAE feature, a probe weight vector) to the residual
    stream with a tunable coefficient.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..losses import spatial_rms
from .record import match_module

__all__ = [
    "patch_site",
    "steer",
    "latent_channel_response",
    "influence_matrix",
    "noise_sensitivity",
    "ResponseResult",
]


@contextmanager
def patch_site(
    model: nn.Module, pattern: str, fn: Callable[[torch.Tensor], torch.Tensor]
) -> Iterator[list[str]]:
    """Apply ``fn`` to the output of every module matching ``pattern``.

    >>> with patch_site(model, "backbone.blocks.6", lambda h: h * 0):
    ...     out = model.step(...)      # layer 6 contributes nothing
    """
    handles = []
    matched: list[str] = []
    for name, mod in model.named_modules():
        if name and match_module(name, pattern):
            matched.append(name)
            handles.append(
                mod.register_forward_hook(lambda _m, _i, o, fn=fn: fn(o))
            )
    if not matched:
        for h in handles:
            h.remove()
        raise ValueError(f"no module matched {pattern!r}")
    try:
        yield matched
    finally:
        for h in handles:
            h.remove()


def steer(
    model: nn.Module,
    pattern: str,
    direction: torch.Tensor,
    coefficient: float = 1.0,
    token_mask: torch.Tensor | None = None,
):
    """Context manager adding ``coefficient * direction`` to a residual stream.

    ``direction`` is a ``(E,)`` vector -- an SAE decoder column, a probe weight
    row, or a difference-of-means direction.  ``token_mask`` restricts the edit
    to a region, which is how you ask "what if this feature were active *here*".
    """

    def fn(h: torch.Tensor) -> torch.Tensor:
        d = direction.to(h.device, h.dtype).reshape(*([1] * (h.ndim - 1)), -1)
        delta = coefficient * d
        if token_mask is not None:
            m = token_mask.to(h.device, h.dtype).reshape(1, -1, 1)
            delta = delta * m
        return h + delta

    return patch_site(model, pattern, fn)


# ---------------------------------------------------------------------------
# latent-space causal sweeps
# ---------------------------------------------------------------------------


@dataclass
class ResponseResult:
    """Change in each output channel caused by one perturbation."""

    delta: torch.Tensor  # (C_out,) area-weighted RMS response
    baseline: torch.Tensor  # (C_out,) area-weighted RMS of the control forecast
    source: str
    amplitude: float

    @property
    def relative(self) -> torch.Tensor:
        return self.delta / self.baseline.clamp_min(1e-12)


@torch.no_grad()
def latent_channel_response(
    model,
    x_fine: torch.Tensor,
    z0: torch.Tensor,
    history: torch.Tensor | None,
    channel: int | str,
    amplitude: float = 1.0,
    times: np.ndarray | None = None,
    pattern: str = "uniform",
    generator: torch.Generator | None = None,
) -> ResponseResult:
    """Perturb one latent-residual channel and decode; report the response.

    The perturbation is applied *after* the probabilistic sample is drawn and
    the same sample is used for the control, so the difference is purely the
    projector's response and is not contaminated by sampling noise.

    ``amplitude`` is in units of the residual's normalised standard deviation,
    so 1.0 is "a typical 6-hour increment for this variable".
    """
    if isinstance(channel, str):
        channel = model.cfg.variables.index(channel)
    res = model.predict_latent(z0, history, n_ensemble=1, generator=generator)
    r = res.sample

    if pattern == "uniform":
        bump = torch.ones_like(r[:, channel])
    elif pattern == "tropics":
        lat = torch.as_tensor(model.cfg.latent_grid.lat, dtype=r.dtype, device=r.device)
        bump = torch.exp(-((lat / 15.0) ** 2)).reshape(1, -1, 1).expand_as(r[:, channel])
    else:
        raise ValueError(f"unknown perturbation pattern {pattern!r}")

    r_pert = r.clone()
    r_pert[:, channel] = r_pert[:, channel] + amplitude * bump

    base = model.decode(r, x_fine, times)
    pert = model.decode(r_pert, x_fine, times)

    area = model.fine_area
    delta = spatial_rms(pert - base, area)
    baseline = spatial_rms(base, area)
    return ResponseResult(
        delta=delta.mean(0),
        baseline=baseline.mean(0),
        source=model.cfg.variables[channel].name,
        amplitude=amplitude,
    )


@torch.no_grad()
def influence_matrix(
    model,
    x_fine: torch.Tensor,
    z0: torch.Tensor,
    history: torch.Tensor | None,
    channels: Sequence[int | str] | None = None,
    amplitude: float = 1.0,
    times: np.ndarray | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """Directed influence matrix ``M[i, j]``: response of output ``j`` to latent ``i``.

    Rows are normalised by the control forecast's amplitude so the entries are
    comparable across variables with wildly different units.  The result is a
    learned, empirical analogue of a Jacobian coupling diagram -- read the
    off-diagonal structure to see which variables the decoder has tied
    together.
    """
    vs = model.cfg.variables
    idx = [vs.index(c) if isinstance(c, str) else c for c in (channels or range(len(vs)))]
    rows = []
    for c in idx:
        rows.append(
            latent_channel_response(
                model, x_fine, z0, history, c, amplitude, times, generator=generator
            ).relative
        )
    return torch.stack(rows), [vs[c].name for c in idx]


@torch.no_grad()
def noise_sensitivity(
    model,
    z0: torch.Tensor,
    history: torch.Tensor | None,
    n_ensemble: int = 16,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Where does ensemble spread come from, in latent space?

    Draws an ensemble of latent residuals and reports per-channel spread and
    the leading principal directions of the ensemble's deviation from its
    mean.  Those directions are the model's own estimate of the fastest-growing
    uncertainty -- the learned analogue of singular vectors.
    """
    res = model.predict_latent(z0, history, n_ensemble=n_ensemble, generator=generator)
    r = res.sample  # (B*M, C, h, w)
    b = z0.shape[0]
    r = r.reshape(b, n_ensemble, *r.shape[1:])
    dev = r - r.mean(1, keepdim=True)

    spread = spatial_rms(dev, model.latent_area).mean(dim=(0, 1))

    flat = dev.reshape(b * n_ensemble, -1)
    k = min(8, n_ensemble - 1, flat.shape[0])
    u, s, v = torch.pca_lowrank(flat, q=max(k, 1), center=True)
    var = s**2 / (s**2).sum().clamp_min(1e-12)
    modes = v[:, :k].T.reshape(k, *r.shape[2:])
    return {
        "channel_spread": spread,
        "mode_variance": var[:k],
        "modes": modes,
    }
