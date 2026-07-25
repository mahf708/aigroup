"""Probabilistic estimators sharing one latent backbone.

The paper's method-agnostic claim is implemented literally here: all three
estimators consume the *same* :class:`~atlas.backbones.LatentDiT` and expose
the same two methods.

``loss(net, z0, history, r_target)``
    Scalar training objective.
``sample(net, z0, history, n_samples)``
    Draws from ``rho_c(r_1 | z_0, z_-1)``.

Each sampler can return its full integration trajectory, which turns the
sampling path itself into an object of study -- see
:func:`atlas.probe.diagnostics.trajectory_summary`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .grids import SphericalTransform, spherical_white_noise
from .losses import Weights, weighted_mean

__all__ = [
    "Estimator",
    "SampleResult",
    "StochasticInterpolant",
    "EDMDiffusion",
    "CRPSEstimator",
    "build_estimator",
]

NetFn = Callable[..., torch.Tensor]


@dataclass
class SampleResult:
    """A draw from the latent conditional, plus optional integration history."""

    sample: torch.Tensor
    trajectory: torch.Tensor | None = None
    times: torch.Tensor | None = None


class Estimator(nn.Module):
    """Common interface; subclasses implement ``loss`` and ``sample``."""

    #: whether repeated calls with the same inputs give different answers
    stochastic: bool = True

    def __init__(self, noise_kind: str = "white", studentt_deg: float | None = None) -> None:
        super().__init__()
        self.noise_kind = noise_kind
        self.studentt_deg = studentt_deg
        self._sphere: SphericalTransform | None = None

    def attach_sphere(self, transform: SphericalTransform | None) -> None:
        """Give the estimator a spherical transform for isotropic noise."""
        self._sphere = transform

    def noise_like(
        self, x: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        if self.noise_kind == "spherical" and self._sphere is not None:
            return spherical_white_noise(
                list(x.shape),
                device=x.device,
                generator=generator,
                studentt_deg=self.studentt_deg,
                transform=self._sphere,
            ).to(x.dtype)
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)

    # -- to be provided by subclasses --------------------------------------

    def loss(  # pragma: no cover - abstract
        self,
        net: NetFn,
        z0: torch.Tensor,
        history: torch.Tensor | None,
        target: torch.Tensor,
        weights: Weights | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def sample(  # pragma: no cover - abstract
        self,
        net: NetFn,
        z0: torch.Tensor,
        history: torch.Tensor | None,
        n_samples: int = 1,
        generator: torch.Generator | None = None,
        record_trajectory: bool = False,
    ) -> SampleResult:
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _tile(x: torch.Tensor | None, n: int) -> torch.Tensor | None:
        if x is None or n == 1:
            return x
        return x.repeat_interleave(n, dim=0)


def _bshape(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-sample scalar to ``ref``'s rank."""
    return t.reshape(-1, *([1] * (ref.ndim - 1)))


# ---------------------------------------------------------------------------
# stochastic interpolants (Section 2.4.1)
# ---------------------------------------------------------------------------


class StochasticInterpolant(Estimator):
    r"""Time-series stochastic interpolant with a linear schedule.

    The interpolant bridges the current latent state and the latent residual,

    .. math:: I_t = \alpha(t) z_0 + \beta(t) r_1 + \sigma(t) W_t,

    with :math:`\alpha = 1-t`, :math:`\beta = t`, :math:`\sigma = \epsilon(1-t)`
    so that :math:`I_0 = z_0` and :math:`I_1 = r_1`.  The network learns the
    drift of the SDE whose law matches the bridge.

    Following the paper, inputs and the regression target are rescaled by
    ``c_in`` / ``c_out`` so both are order-one at every ``t``; the analytic
    part of the drift (``c_skip * z_0``) is added outside the network.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        beta_power: float = 1.0,
        sample_method: str = "heun",
        steps: int = 60,
        noise_kind: str = "white",
        studentt_deg: float | None = None,
        t_eps: float = 1e-4,
    ) -> None:
        super().__init__(noise_kind, studentt_deg)
        self.epsilon = float(epsilon)
        self.beta_power = float(beta_power)
        self.sample_method = sample_method
        self.steps = int(steps)
        self.t_eps = float(t_eps)

    # schedule ------------------------------------------------------------

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return t**self.beta_power

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        if self.beta_power == 1.0:
            return torch.ones_like(t)
        return self.beta_power * t ** (self.beta_power - 1.0)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.epsilon * (1.0 - t)

    def sigma_dot(self, t: torch.Tensor) -> torch.Tensor:
        return -self.epsilon * torch.ones_like(t)

    def preconditioning(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """``c_skip`` / ``c_out`` / ``c_in`` for the drift parameterisation."""
        a, ad = self.alpha(t), self.alpha_dot(t)
        b, bd = self.beta(t), self.beta_dot(t)
        s, sd = self.sigma(t), self.sigma_dot(t)
        return {
            "c_skip": ad,
            "c_out": torch.sqrt(bd**2 + sd**2 * t),
            "c_in": 1.0 / torch.sqrt(a**2 + b**2 + s**2 * t),
        }

    # training ------------------------------------------------------------

    def loss(self, net, z0, history, target, weights=None):
        b = z0.shape[0]
        t = torch.rand(b, device=z0.device, dtype=z0.dtype).clamp(self.t_eps, 1.0 - self.t_eps)
        xi = self.noise_like(z0)
        tb = _bshape(t, z0)
        w = torch.sqrt(tb) * xi  # W_t | t  ~  sqrt(t) * xi

        interp = self.alpha(tb) * z0 + self.beta(tb) * target + self.sigma(tb) * w
        drift = self.alpha_dot(tb) * z0 + self.beta_dot(tb) * target + self.sigma_dot(tb) * w

        pc = {k: _bshape(v, z0) for k, v in self.preconditioning(t).items()}
        pred = net(pc["c_in"] * interp, t, z0, history)
        goal = (drift - pc["c_skip"] * z0) / pc["c_out"]
        loss = weighted_mean((pred - goal) ** 2, weights).mean()
        return loss, {"si_loss": float(loss.detach())}

    # sampling ------------------------------------------------------------

    def _drift(self, net, x, t, z0, history) -> torch.Tensor:
        pc = {k: _bshape(v, x) for k, v in self.preconditioning(t).items()}
        out = net(pc["c_in"] * x, t, z0, history)
        return pc["c_skip"] * z0 + pc["c_out"] * out

    @torch.no_grad()
    def sample(
        self,
        net,
        z0,
        history=None,
        n_samples: int = 1,
        generator: torch.Generator | None = None,
        record_trajectory: bool = False,
        steps: int | None = None,
    ) -> SampleResult:
        z0 = self._tile(z0, n_samples)
        history = self._tile(history, n_samples)
        steps = steps or self.steps
        ts = torch.linspace(0.0, 1.0, steps + 1, device=z0.device, dtype=z0.dtype)
        x = z0.clone()

        traj = [x.clone()] if record_trajectory else None
        for i in range(steps):
            t0 = ts[i].expand(x.shape[0])
            t1 = ts[i + 1].expand(x.shape[0])
            dt = (ts[i + 1] - ts[i]).item()
            g = _bshape(self.sigma(t0), x)
            noise = self.noise_like(x, generator)

            if self.sample_method == "euler":
                b0 = self._drift(net, x, t0, z0, history)
                x = x + dt * b0 + math.sqrt(dt) * g * noise
            elif self.sample_method == "heun":
                # Roberts' modified-Euler scheme for SDEs (paper ref [43]).
                s = (torch.randint(0, 2, (1,), generator=generator).item() - 0.5) * 2
                b0 = self._drift(net, x, t0, z0, history)
                g1 = _bshape(self.sigma(t1), x)
                y1 = dt * b0 + g * (math.sqrt(dt) * noise - math.sqrt(dt) * s * 0.5)
                b1 = self._drift(net, x + y1, t1, z0, history)
                y2 = dt * b1 + g1 * (math.sqrt(dt) * noise + math.sqrt(dt) * s * 0.5)
                x = x + 0.5 * (y1 + y2)
            else:
                raise ValueError(f"unknown sample_method {self.sample_method!r}")
            if traj is not None:
                traj.append(x.clone())

        return SampleResult(
            x,
            torch.stack(traj) if traj is not None else None,
            ts if traj is not None else None,
        )


# ---------------------------------------------------------------------------
# EDM diffusion (Section 2.4.2)
# ---------------------------------------------------------------------------


class EDMDiffusion(Estimator):
    """Karras-style score model on the latent residual.

    Uses the EDM preconditioning and loss weighting so that the network sees
    unit-variance inputs and targets at every noise level, and a Heun sampler
    over the standard ``rho = 7`` noise schedule.
    """

    def __init__(
        self,
        sigma_data: float = 1.0,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        steps: int = 40,
        s_churn: float = 0.0,
        s_min: float = 0.0,
        s_max: float = float("inf"),
        s_noise: float = 1.0,
        noise_kind: str = "white",
        studentt_deg: float | None = None,
    ) -> None:
        super().__init__(noise_kind, studentt_deg)
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.p_mean = p_mean
        self.p_std = p_std
        self.steps = steps
        self.s_churn = s_churn
        self.s_min = s_min
        self.s_max = s_max
        self.s_noise = s_noise

    def preconditioning(self, sigma: torch.Tensor) -> dict[str, torch.Tensor]:
        sd = self.sigma_data
        return {
            "c_skip": sd**2 / (sigma**2 + sd**2),
            "c_out": sigma * sd / torch.sqrt(sigma**2 + sd**2),
            "c_in": 1.0 / torch.sqrt(sigma**2 + sd**2),
            "c_noise": torch.log(sigma.clamp_min(1e-12)) / 4.0,
        }

    def denoise(self, net, x, sigma, z0, history) -> torch.Tensor:
        pc = self.preconditioning(sigma)
        out = net(_bshape(pc["c_in"], x) * x, pc["c_noise"], z0, history)
        return _bshape(pc["c_skip"], x) * x + _bshape(pc["c_out"], x) * out

    def loss(self, net, z0, history, target, weights=None):
        b = z0.shape[0]
        rnd = torch.randn(b, device=z0.device, dtype=z0.dtype)
        sigma = (rnd * self.p_std + self.p_mean).exp()
        n = self.noise_like(target) * _bshape(sigma, target)
        d = self.denoise(net, target + n, sigma, z0, history)
        lam = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        err = weighted_mean((d - target) ** 2, weights)
        loss = (lam * err).mean()
        return loss, {"edm_loss": float(loss.detach())}

    def _sigmas(self, steps: int, device, dtype) -> torch.Tensor:
        i = torch.arange(steps, device=device, dtype=dtype)
        inv = 1.0 / self.rho
        s = (
            self.sigma_max**inv
            + i / max(steps - 1, 1) * (self.sigma_min**inv - self.sigma_max**inv)
        ) ** self.rho
        return torch.cat([s, torch.zeros(1, device=device, dtype=dtype)])

    @torch.no_grad()
    def sample(
        self,
        net,
        z0,
        history=None,
        n_samples: int = 1,
        generator: torch.Generator | None = None,
        record_trajectory: bool = False,
        steps: int | None = None,
    ) -> SampleResult:
        z0 = self._tile(z0, n_samples)
        history = self._tile(history, n_samples)
        steps = steps or self.steps
        sig = self._sigmas(steps, z0.device, z0.dtype)
        x = self.noise_like(z0, generator) * sig[0]
        traj = [x.clone()] if record_trajectory else None

        for i in range(steps):
            s_cur, s_next = sig[i], sig[i + 1]
            gamma = (
                min(self.s_churn / steps, math.sqrt(2) - 1)
                if self.s_min <= float(s_cur) <= self.s_max
                else 0.0
            )
            s_hat = s_cur * (1 + gamma)
            if gamma > 0:
                x = x + (s_hat**2 - s_cur**2).sqrt() * self.s_noise * self.noise_like(
                    x, generator
                )
            sb = s_hat.expand(x.shape[0])
            d = (x - self.denoise(net, x, sb, z0, history)) / s_hat
            x_next = x + (s_next - s_hat) * d
            if s_next > 0:
                sb2 = s_next.expand(x.shape[0])
                d2 = (x_next - self.denoise(net, x_next, sb2, z0, history)) / s_next
                x_next = x + (s_next - s_hat) * 0.5 * (d + d2)
            x = x_next
            if traj is not None:
                traj.append(x.clone())

        return SampleResult(
            x,
            torch.stack(traj) if traj is not None else None,
            sig if traj is not None else None,
        )


# ---------------------------------------------------------------------------
# CRPS (Section 2.4.3)
# ---------------------------------------------------------------------------


class CRPSEstimator(Estimator):
    r"""Direct generator trained on a spectrally regularised CRPS.

    The network maps ``(z_0, \xi)`` straight to a latent residual, so sampling
    is a single forward pass -- roughly 30x cheaper than the SDE-based variants
    at inference.  The plain two-sample CRPS is spectrally biased (it damps
    high wavenumbers), so the paper adds a CRPS computed on the magnitudes of
    spherical-harmonic coefficients; ``lambda_spectral`` controls its weight.
    """

    def __init__(
        self,
        noise_dim: int = 256,
        lambda_spectral: float = 0.1,
        sphere: SphericalTransform | None = None,
        noise_kind: str = "white",
        studentt_deg: float | None = None,
    ) -> None:
        super().__init__(noise_kind, studentt_deg)
        self.noise_dim = noise_dim
        self.lambda_spectral = lambda_spectral
        self._spectral = sphere

    def attach_sphere(self, transform: SphericalTransform | None) -> None:
        super().attach_sphere(transform)
        if self._spectral is None:
            self._spectral = transform

    def _xi(self, b: int, device, dtype, generator=None) -> torch.Tensor:
        return torch.randn(b, self.noise_dim, device=device, dtype=dtype, generator=generator)

    def _spectral_crps(self, f1, f2, target) -> torch.Tensor:
        if self._spectral is None:
            return torch.zeros((), device=f1.device, dtype=f1.dtype)
        s1 = self._spectral(f1).abs()
        s2 = self._spectral(f2).abs()
        st = self._spectral(target).abs()
        return ((s1 - st).abs() + (s2 - st).abs() - (s1 - s2).abs()).mean()

    def loss(self, net, z0, history, target, weights=None):
        b = z0.shape[0]
        ones = torch.ones(b, device=z0.device, dtype=z0.dtype)
        zero = torch.zeros_like(z0)
        f1 = net(zero, ones, z0, history, self._xi(b, z0.device, z0.dtype))
        f2 = net(zero, ones, z0, history, self._xi(b, z0.device, z0.dtype))
        crps = weighted_mean(
            (f1 - target).abs() + (f2 - target).abs() - (f1 - f2).abs(), weights
        ).mean()
        out = {"crps": float(crps.detach())}
        loss = crps
        if self.lambda_spectral > 0:
            spec = self._spectral_crps(f1, f2, target)
            loss = loss + self.lambda_spectral * spec
            out["spectral_crps"] = float(spec.detach())
        out["loss"] = float(loss.detach())
        return loss, out

    @torch.no_grad()
    def sample(
        self,
        net,
        z0,
        history=None,
        n_samples: int = 1,
        generator: torch.Generator | None = None,
        record_trajectory: bool = False,
        steps: int | None = None,
    ) -> SampleResult:
        z0 = self._tile(z0, n_samples)
        history = self._tile(history, n_samples)
        b = z0.shape[0]
        ones = torch.ones(b, device=z0.device, dtype=z0.dtype)
        xi = self._xi(b, z0.device, z0.dtype, generator)
        out = net(torch.zeros_like(z0), ones, z0, history, xi)
        return SampleResult(out, out.unsqueeze(0) if record_trajectory else None, None)


# ---------------------------------------------------------------------------


def build_estimator(kind: str, **options: Any) -> Estimator:
    """Factory used by :class:`atlas.spec.EstimatorConfig`."""
    kind = kind.lower()
    if kind in {"si", "interpolant", "stochastic_interpolant"}:
        return StochasticInterpolant(**options)
    if kind in {"edm", "diffusion"}:
        return EDMDiffusion(**options)
    if kind == "crps":
        return CRPSEstimator(**options)
    raise ValueError(f"unknown estimator {kind!r}; expected one of si | edm | crps")
