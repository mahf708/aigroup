"""Linear probes: what physical information is linearly decodable, and where.

The standard experiment is a *depth sweep*: fit a ridge probe from every DiT
block's residual stream to some physical diagnostic and plot skill against
layer.  The shape of that curve says something concrete -- an index that is
already decodable at layer 0 was handed to the model by the input encoding; one
that only becomes decodable in the middle of the stack is being *computed*.

All probes here are deliberately linear.  A non-linear probe can decode almost
anything from a 3,328-dimensional representation and tells you very little.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ..spec import GridSpec, VariableSet

__all__ = [
    "LinearProbe",
    "fit_ridge",
    "pool_tokens",
    "depth_sweep",
    "TargetFn",
    "builtin_targets",
]

TargetFn = Callable[[torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# ridge
# ---------------------------------------------------------------------------


def fit_ridge(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, fit_intercept: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form ridge in float64; returns ``(weight, bias)``.

    ``x`` is ``(n, d)`` and ``y`` is ``(n, k)``.  Solved through the normal
    equations, or their dual when ``d > n`` (the usual regime when probing a
    3,328-dimensional stream with a few hundred forecast samples).
    """
    x = x.double()
    y = y.double()
    if fit_intercept:
        xm, ym = x.mean(0, keepdim=True), y.mean(0, keepdim=True)
        xc, yc = x - xm, y - ym
    else:
        xm = torch.zeros(1, x.shape[1], dtype=x.dtype, device=x.device)
        ym = torch.zeros(1, y.shape[1], dtype=y.dtype, device=y.device)
        xc, yc = x, y

    n, d = xc.shape
    if d <= n:
        a = xc.T @ xc + alpha * torch.eye(d, dtype=x.dtype, device=x.device)
        w = torch.linalg.solve(a, xc.T @ yc)
    else:  # dual form: cheaper and better conditioned when features outnumber samples
        k = xc @ xc.T + alpha * torch.eye(n, dtype=x.dtype, device=x.device)
        w = xc.T @ torch.linalg.solve(k, yc)
    b = ym - xm @ w
    return w.float(), b.float().squeeze(0)


def r2_score(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ss_res = ((pred - y) ** 2).sum(0)
    ss_tot = ((y - y.mean(0, keepdim=True)) ** 2).sum(0)
    return 1.0 - ss_res / ss_tot.clamp_min(1e-12)


@dataclass
class ProbeResult:
    r2: torch.Tensor
    r2_train: torch.Tensor
    alpha: float
    n_features: int
    n_samples: int

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ProbeResult(r2={self.r2.mean():.3f}, train={self.r2_train.mean():.3f}, "
            f"alpha={self.alpha:g}, d={self.n_features}, n={self.n_samples})"
        )


class LinearProbe:
    """Ridge probe with a held-out split and automatic alpha selection."""

    def __init__(self, alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0)) -> None:
        self.alphas = list(alphas)
        self.weight: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        self.result: ProbeResult | None = None

    def fit(
        self, x: torch.Tensor, y: torch.Tensor, val_fraction: float = 0.25, seed: int = 0
    ) -> ProbeResult:
        x = x.reshape(x.shape[0], -1).float()
        y = y.reshape(y.shape[0], -1).float()
        n = x.shape[0]
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        n_val = max(1, int(round(val_fraction * n)))
        vi, ti = perm[:n_val], perm[n_val:]
        if ti.numel() < 2:
            raise ValueError("not enough samples to fit a probe")

        best_score = -float("inf")
        for a in self.alphas:
            w, b = fit_ridge(x[ti], y[ti], alpha=a)
            r2v = r2_score(x[vi] @ w + b, y[vi])
            score = float(r2v.mean())
            if score > best_score:
                best_score = score
                self.weight, self.bias = w, b
                self.result = ProbeResult(
                    r2=r2v,
                    r2_train=r2_score(x[ti] @ w + b, y[ti]),
                    alpha=a,
                    n_features=x.shape[1],
                    n_samples=n,
                )
        assert self.result is not None
        return self.result

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            raise RuntimeError("probe is not fitted")
        return x.reshape(x.shape[0], -1).float() @ self.weight + self.bias


# ---------------------------------------------------------------------------
# feature preparation
# ---------------------------------------------------------------------------


def pool_tokens(
    tokens: torch.Tensor,
    how: str = "mean",
    indices: torch.Tensor | None = None,
    n_components: int | None = None,
) -> torch.Tensor:
    """Reduce ``(B, N, E)`` activations to ``(B, d)`` probe features.

    ``mean``
        Global average pooling; the right default for global indices.
    ``select``
        Concatenate the tokens named by ``indices``; use for local targets
        (a cyclone's position, a regional index).
    ``flatten``
        Everything.  Only sane after ``n_components`` PCA.
    """
    if tokens.ndim != 3:
        raise ValueError(f"expected (B, N, E) activations, got {tuple(tokens.shape)}")
    if how == "mean":
        feats = tokens.mean(1)
    elif how == "select":
        if indices is None:
            raise ValueError("how='select' needs indices")
        feats = tokens[:, indices.to(tokens.device)].reshape(tokens.shape[0], -1)
    elif how == "flatten":
        feats = tokens.reshape(tokens.shape[0], -1)
    else:
        raise ValueError(f"unknown pooling {how!r}")
    if n_components:
        feats = pca_reduce(feats, n_components)
    return feats


def pca_reduce(x: torch.Tensor, k: int, center: bool = True) -> torch.Tensor:
    """Project onto the leading ``k`` principal components."""
    x = x.float()
    mu = x.mean(0, keepdim=True) if center else torch.zeros_like(x[:1])
    xc = x - mu
    k = min(k, min(xc.shape))
    _, _, v = torch.pca_lowrank(xc, q=k, center=False)
    return xc @ v[:, :k]


def depth_sweep(
    layer_activations: dict[str, torch.Tensor],
    target: torch.Tensor,
    pooling: str = "mean",
    n_components: int | None = 128,
    alphas: Sequence[float] = (1.0, 10.0, 100.0, 1000.0),
    val_fraction: float = 0.25,
) -> dict[str, ProbeResult]:
    """Fit one probe per layer and return skill as a function of depth."""
    out: dict[str, ProbeResult] = {}
    for name, act in layer_activations.items():
        feats = pool_tokens(act, pooling, n_components=n_components)
        probe = LinearProbe(alphas)
        out[name] = probe.fit(feats, target, val_fraction=val_fraction)
    return out


# ---------------------------------------------------------------------------
# physical targets
# ---------------------------------------------------------------------------


def _region_mean(
    x: torch.Tensor, grid: GridSpec, lat_range, lon_range, weights: np.ndarray | None = None
) -> torch.Tensor:
    lat = grid.lat
    lon = np.mod(grid.lon, 360.0)
    la = (lat >= min(lat_range)) & (lat <= max(lat_range))
    lo0, lo1 = np.mod(lon_range[0], 360.0), np.mod(lon_range[1], 360.0)
    lo = (lon >= lo0) & (lon <= lo1) if lo0 <= lo1 else (lon >= lo0) | (lon <= lo1)
    w = torch.as_tensor(
        (weights if weights is not None else grid.area_weights())[la], dtype=x.dtype
    ).to(x.device)
    sub = x[..., la, :][..., lo]
    return (sub * w).sum(-2).mean(-1) / w.sum()


def builtin_targets(
    variables: VariableSet, grid: GridSpec
) -> dict[str, TargetFn]:
    """Diagnostics computable from a physical state ``(B, C, nlat, nlon)``.

    Only the entries whose input channels exist in ``variables`` are returned,
    so the same call works for the ERA5 set and for an E3SM channel set.
    """
    names = set(variables.names)
    fns: dict[str, TargetFn] = {}

    def add_global_mean(chan: str) -> None:
        i = variables.index(chan)
        w = torch.as_tensor(grid.area_weights())

        def fn(x: torch.Tensor, i=i, w=w) -> torch.Tensor:
            return (x[:, i] * w.to(x.device, x.dtype)).sum(-2).mean(-1)

        fns[f"global_mean_{chan}"] = fn

    for cand in ("t2m", "TS", "tcwv", "TMQ", "msl", "PS"):
        if cand in names:
            add_global_mean(cand)

    sst_like = next((c for c in ("sst", "TS", "t2m") if c in names), None)
    if sst_like is not None:
        i = variables.index(sst_like)

        def nino34(x: torch.Tensor, i=i) -> torch.Tensor:
            return _region_mean(x[:, i], grid, (-5, 5), (190, 240))

        fns["nino34"] = nino34

    jet_var = next((c for c in ("u850", "u250", "U_lev20") if c in names), None)
    if jet_var is not None:
        i = variables.index(jet_var)
        lat_t = torch.as_tensor(grid.lat, dtype=torch.float32)

        def jet_latitude(x: torch.Tensor, i=i, lat_t=lat_t) -> torch.Tensor:
            """Weighted centroid latitude of the SH midlatitude westerlies."""
            zonal = x[:, i].mean(-1)
            mask = (lat_t >= -65) & (lat_t <= -25)
            u = zonal[:, mask.to(zonal.device)].clamp_min(0)
            la = lat_t[mask].to(u.device, u.dtype)
            return (u * la).sum(-1) / u.sum(-1).clamp_min(1e-6)

        fns["jet_latitude_sh"] = jet_latitude

    if "u850" in names and "v850" in names:
        iu, iv = variables.index("u850"), variables.index("v850")
        w = torch.as_tensor(grid.area_weights())

        def eke850(x: torch.Tensor, iu=iu, iv=iv, w=w) -> torch.Tensor:
            u = x[:, iu] - x[:, iu].mean(-1, keepdim=True)
            v = x[:, iv] - x[:, iv].mean(-1, keepdim=True)
            e = 0.5 * (u**2 + v**2)
            return (e * w.to(x.device, x.dtype)).sum(-2).mean(-1)

        fns["eke850"] = eke850

    if "z500" in names:
        i = variables.index("z500")

        def nao(x: torch.Tensor, i=i) -> torch.Tensor:
            north = _region_mean(x[:, i], grid, (60, 70), (300, 340))
            south = _region_mean(x[:, i], grid, (35, 45), (330, 20))
            return south - north

        fns["nao_like"] = nao

    return fns
