"""Grid operations: resampling, sphere-aware padding, positional encodings.

Kept free of any model-specific logic so the same helpers back the encoder,
the projector and the diagnostics in :mod:`atlas.probe`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spec import GridSpec

__all__ = [
    "bilinear_resample",
    "area_downsample",
    "Resampler",
    "LearnedCorrection",
    "SphericalPad",
    "sincos_2d_embedding",
    "latlon_embedding",
    "build_pos_embedding",
    "SphericalTransform",
    "spherical_white_noise",
]


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------


def bilinear_resample(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Bilinear interpolation to ``shape``; the paper's encoder ``B``.

    ``align_corners=True`` keeps the poles (and hence the global extrema)
    exactly on the target grid, which is what makes 721 -> 181 a clean 4x
    reduction.
    """
    if tuple(x.shape[-2:]) == tuple(shape):
        return x
    return F.interpolate(x, size=tuple(shape), mode="bilinear", align_corners=True)


def area_downsample(
    x: torch.Tensor, shape: tuple[int, int], weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Area-weighted average pooling down to ``shape``.

    Unlike bilinear interpolation this conserves the area-weighted global mean
    of every channel, which is often the property you want when the model is
    going to be judged on E3SM energy/water budgets.  Falls back to adaptive
    average pooling when the ratio is not integral.
    """
    if tuple(x.shape[-2:]) == tuple(shape):
        return x
    if weights is None:
        return F.adaptive_avg_pool2d(x, tuple(shape))
    w = weights.to(x.dtype).to(x.device)
    num = F.adaptive_avg_pool2d(x * w, tuple(shape))
    den = F.adaptive_avg_pool2d(w.expand_as(x), tuple(shape))
    return num / den.clamp_min(1e-12)


class Resampler(nn.Module):
    """Encoder/decoder pair between a fine grid and a coarse latent grid.

    This is the (deliberately trivial) ``B`` operator of the paper.  Keeping it
    as a module means the choice of encoder is a swappable experiment knob and
    that the latent stays a *named, physical* object: latent channel ``c`` is
    always a coarse-grained version of state channel ``c``.
    """

    def __init__(
        self,
        grid: GridSpec,
        latent_shape: tuple[int, int],
        mode: str = "bilinear",
        lmax: int | None = None,
        n_channels: int | None = None,
        hidden_dim: int = 256,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.latent_shape = tuple(latent_shape)
        self.grid_shape = grid.shape
        self.register_buffer(
            "area", torch.from_numpy(grid.area_weights()).view(1, 1, -1, 1), persistent=False
        )
        self._sht: SphericalTransform | None = None
        if mode == "spectral":
            self._sht = SphericalTransform(grid, lmax=lmax or self.latent_shape[0])
        self.learned: LearnedCorrection | None = None
        if mode == "learned":
            if n_channels is None:
                raise ValueError("mode='learned' needs n_channels")
            self.learned = LearnedCorrection(
                n_channels, grid, self.latent_shape, hidden_dim, depth, grid.periodic_lon
            )

    @property
    def is_learned(self) -> bool:
        return self.learned is not None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "bilinear":
            return bilinear_resample(x, self.latent_shape)
        if self.mode == "area":
            return area_downsample(x, self.latent_shape, self.area)
        if self.mode == "spectral":
            assert self._sht is not None
            return self._sht.truncate(x, self.latent_shape)
        if self.mode == "learned":
            assert self.learned is not None
            return self.learned(x, bilinear_resample(x, self.latent_shape))
        raise ValueError(f"unknown resampler mode {self.mode!r}")

    def upsample(self, x: torch.Tensor) -> torch.Tensor:
        """Naive inverse; only used as a baseline against the learned projector."""
        return bilinear_resample(x, self.grid_shape)

    forward = encode


class LearnedCorrection(nn.Module):
    """A trainable encoder that keeps the latent's channel identity.

    This is the middle ground the paper's Section 5 ablation is really about.
    A full VAE-style encoder mixes all variables into a common channel space,
    and the paper reports two failure modes for that: a flat high-frequency
    tail in the latent spectrum, and a loss of temporal continuity.  Both
    stem from the mixing, not from learning as such.

    So this encoder learns a *correction* to the coarse-grained field rather
    than a new code::

        z = B(x) + g(B(x), P(x))

    where ``P`` is a strided convolution that lets sub-grid structure inform
    the coarse value (an area-average cannot).  Latent channel ``c`` still
    corresponds to physical variable ``c``, so residual normalisation, the
    latent update ``z + r`` and every diagnostic in :mod:`atlas.probe` keep
    working; the correction is initialised to zero, so training starts exactly
    at the bilinear baseline and any departure from it is measurable.

    Train it through the projector's reconstruction loss (``parts`` including
    ``"encoder"``), not through the estimator: the latent target must not be
    free to move while the probabilistic head is chasing it.
    """

    def __init__(
        self,
        n_channels: int,
        grid: GridSpec,
        latent_shape: tuple[int, int],
        hidden_dim: int = 256,
        depth: int = 4,
        periodic_lon: bool = True,
    ) -> None:
        super().__init__()
        patch = (
            -(-grid.shape[0] // latent_shape[0]),
            -(-grid.shape[1] // latent_shape[1]),
        )
        self.patch = patch
        self.latent_shape = tuple(latent_shape)
        self.periodic_lon = periodic_lon
        self.fine_proj = nn.Conv2d(n_channels, hidden_dim, kernel_size=patch, stride=patch)
        layers: list[nn.Module] = [
            nn.Conv2d(hidden_dim + n_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(approximate="tanh"),
        ]
        for _ in range(max(depth - 1, 0)):
            layers += [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(approximate="tanh"),
            ]
        out = nn.Conv2d(hidden_dim, n_channels, kernel_size=1)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        ph = (-fine.shape[-2]) % self.patch[0]
        pw = (-fine.shape[-1]) % self.patch[1]
        if pw:
            fine = F.pad(
                fine, (0, pw, 0, 0), mode="circular" if self.periodic_lon else "replicate"
            )
        if ph:
            fine = F.pad(fine, (0, 0, 0, ph), mode="replicate")
        feats = self.fine_proj(fine)[..., : self.latent_shape[0], : self.latent_shape[1]]
        return coarse + self.net(torch.cat([feats, coarse], dim=1))


# ---------------------------------------------------------------------------
# padding
# ---------------------------------------------------------------------------


class SphericalPad(nn.Module):
    """Pad a lat-lon field with topologically correct halos.

    Longitude wraps circularly on a global grid.  Across a pole the physically
    adjacent cell sits at the same latitude ring but 180 degrees away in
    longitude, so ``mode="pole"`` rolls the mirrored rows by ``nlon // 2``.
    That is the "minimal spherically consistent padding" the paper applies
    before local attention; ``mode="reflect"`` reproduces the cheaper variant
    and ``mode="replicate"`` is the right choice for regional domains.
    """

    def __init__(
        self,
        pad_lat: int = 0,
        pad_lon: int = 0,
        mode: str = "pole",
        periodic_lon: bool = True,
    ) -> None:
        super().__init__()
        self.pad_lat = int(pad_lat)
        self.pad_lon = int(pad_lon)
        self.mode = mode
        self.periodic_lon = periodic_lon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_lon:
            lon_mode = "circular" if self.periodic_lon else "replicate"
            x = F.pad(x, (self.pad_lon, self.pad_lon, 0, 0), mode=lon_mode)
        if self.pad_lat:
            p = self.pad_lat
            if self.mode == "pole":
                nlon = x.shape[-1]
                top = torch.roll(x[..., :p, :].flip(-2), shifts=nlon // 2, dims=-1)
                bot = torch.roll(x[..., -p:, :].flip(-2), shifts=nlon // 2, dims=-1)
                x = torch.cat([top, x, bot], dim=-2)
            elif self.mode == "reflect":
                x = F.pad(x, (0, 0, p, p), mode="reflect")
            else:
                x = F.pad(x, (0, 0, p, p), mode="replicate")
        return x


# ---------------------------------------------------------------------------
# positional encodings
# ---------------------------------------------------------------------------


def sincos_2d_embedding(dim: int, h: int, w: int, temperature: float = 10000.0) -> torch.Tensor:
    """Standard non-learned 2-D sine-cosine embedding, shape ``(h * w, dim)``."""
    if dim % 4 != 0:
        raise ValueError("sincos embedding dim must be divisible by 4")
    quarter = dim // 4
    omega = 1.0 / (temperature ** (torch.arange(quarter, dtype=torch.float64) / quarter))
    gy, gx = torch.meshgrid(
        torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij"
    )
    out = []
    for g in (gy, gx):
        a = g.reshape(-1, 1) * omega.reshape(1, -1)
        out += [a.sin(), a.cos()]
    return torch.cat(out, dim=1).float()


def latlon_embedding(dim: int, lat: np.ndarray, lon: np.ndarray, n_freq: int = 8) -> torch.Tensor:
    """Geometry-aware embedding built from the actual coordinates.

    Unlike index-based sine-cosine this stays meaningful when the grid is
    regional, non-uniform, or a different resolution from the one the weights
    were trained on -- useful when moving between E3SM configurations.
    """
    la = torch.deg2rad(torch.as_tensor(lat, dtype=torch.float64))
    lo = torch.deg2rad(torch.as_tensor(lon, dtype=torch.float64))
    LA, LO = torch.meshgrid(la, lo, indexing="ij")
    base = torch.stack(
        [LA.sin(), LA.cos() * LO.cos(), LA.cos() * LO.sin()], dim=-1
    ).reshape(-1, 3)
    freqs = 2.0 ** torch.arange(n_freq, dtype=torch.float64)
    feats = [base]
    for f in freqs:
        feats += [torch.sin(f * math.pi * base), torch.cos(f * math.pi * base)]
    emb = torch.cat(feats, dim=-1).float()
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb[:, :dim].contiguous()


def build_pos_embedding(
    kind: str, dim: int, h: int, w: int, grid: GridSpec | None = None
) -> nn.Parameter:
    """Return a positional-embedding parameter of shape ``(1, h * w, dim)``.

    ``learned`` is trainable; the analytic variants are frozen buffers exposed
    as non-trainable parameters so they still appear in ``state_dict``.
    """
    if kind == "none":
        return nn.Parameter(torch.zeros(1, h * w, dim), requires_grad=False)
    if kind == "learned":
        p = nn.Parameter(torch.zeros(1, h * w, dim))
        nn.init.trunc_normal_(p, std=0.02)
        return p
    if kind == "sincos2d":
        emb = sincos_2d_embedding(dim, h, w)
    elif kind == "latlon":
        if grid is None:
            raise ValueError("pos_embed='latlon' needs a GridSpec")
        lat = np.linspace(grid.lat[0], grid.lat[-1], h)
        lon = np.linspace(grid.lon[0], grid.lon[0] + 360.0, w, endpoint=False)
        emb = latlon_embedding(dim, lat, lon)
    else:
        raise ValueError(f"unknown pos_embed {kind!r}")
    return nn.Parameter(emb.unsqueeze(0), requires_grad=False)


# ---------------------------------------------------------------------------
# spectral tools
# ---------------------------------------------------------------------------


class SphericalTransform(nn.Module):
    """Spherical-harmonic transform with a graceful FFT fallback.

    ``torch-harmonics`` gives the true SHT used by the paper's spectral CRPS
    term.  Without it we fall back to a 2-D real FFT, which is *not* the same
    operator but is still a monotone map from field to scale-resolved power --
    good enough for a regularisation term and for qualitative spectra, and it
    keeps the package importable with nothing but PyTorch installed.
    """

    def __init__(self, grid: GridSpec, lmax: int | None = None, backend: str = "auto") -> None:
        super().__init__()
        self.nlat, self.nlon = grid.shape
        self.lmax = lmax or self.nlat
        self.backend = backend
        self._sht = None
        self._isht = None
        if backend in {"auto", "sht"}:
            try:  # pragma: no cover - depends on optional dependency
                from torch_harmonics import InverseRealSHT, RealSHT

                gridname = "equiangular"
                self._sht = RealSHT(
                    self.nlat, self.nlon, lmax=self.lmax, mmax=self.nlon // 2 + 1,
                    grid=gridname, norm="ortho",
                )
                self._isht = InverseRealSHT(
                    self.nlat, self.nlon, lmax=self.lmax, mmax=self.nlon // 2 + 1,
                    grid=gridname, norm="ortho",
                )
                self.backend = "sht"
            except Exception:
                self.backend = "fft"
        if self._sht is None:
            self.backend = "fft"

    @property
    def available(self) -> bool:
        return self._sht is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Complex spectral coefficients of a ``(B, C, nlat, nlon)`` field."""
        if self._sht is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                return self._sht(x.float())
        return torch.fft.rfft2(x.float(), norm="ortho")

    def power(self, x: torch.Tensor) -> torch.Tensor:
        """Power as a function of total wavenumber, shape ``(B, C, n_wave)``."""
        coeffs = self.forward(x)
        p = coeffs.abs() ** 2
        if self._sht is not None:
            return p.sum(dim=-1)
        ky = torch.fft.fftfreq(x.shape[-2], d=1.0 / x.shape[-2], device=x.device)
        kx = torch.fft.rfftfreq(x.shape[-1], d=1.0 / x.shape[-1], device=x.device)
        k = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
        nbins = min(x.shape[-2] // 2, x.shape[-1] // 2)
        idx = torch.clamp(k.round().long(), max=nbins - 1).reshape(-1)
        flat = p.reshape(*p.shape[:2], -1)
        out = torch.zeros(*p.shape[:2], nbins, device=x.device, dtype=flat.dtype)
        out.index_add_(-1, idx, flat)
        return out

    def truncate(self, x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        """Band-limit ``x`` and evaluate on a coarser grid."""
        if self._sht is None:
            return bilinear_resample(x, shape)
        from torch_harmonics import InverseRealSHT  # pragma: no cover

        key = (shape[0], shape[1])
        cache = getattr(self, "_trunc_cache", {})
        if key not in cache:
            lmax = min(self.lmax, shape[0])
            cache[key] = (
                type(self._sht)(
                    self.nlat, self.nlon, lmax=lmax, mmax=shape[1] // 2 + 1,
                    grid="equiangular", norm="ortho",
                ).to(x.device),
                InverseRealSHT(
                    shape[0], shape[1], lmax=lmax, mmax=shape[1] // 2 + 1,
                    grid="equiangular", norm="ortho",
                ).to(x.device),
            )
            self._trunc_cache = cache
        sht, isht = cache[key]
        with torch.autocast(device_type=x.device.type, enabled=False):
            return isht(sht(x.float()))


def spherical_white_noise(
    shape: Sequence[int],
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
    scale: float = 3.56,
    studentt_deg: float | None = None,
    transform: SphericalTransform | None = None,
) -> torch.Tensor:
    """Isotropic noise on the sphere (grid-white noise is *not* isotropic).

    Falls back to i.i.d. Gaussian noise when no spherical transform is
    available.  ``studentt_deg`` swaps the Gaussian for a Student-t, which the
    reference implementation exposes as a heavier-tailed sampling option.
    """
    shape = list(shape)
    if transform is None or not transform.available:
        if studentt_deg is None:
            return torch.randn(shape, device=device, generator=generator)
        dist = torch.distributions.StudentT(studentt_deg)
        return dist.sample(torch.Size(shape)).to(device)
    isht = transform._isht  # noqa: SLF001 - internal by design
    lmax, mmax = isht.lmax, isht.mmax
    csh = shape[:-2] + [lmax, mmax, 2]
    if studentt_deg is None:
        noise = torch.randn(csh, device=device, generator=generator)
    else:
        noise = torch.distributions.StudentT(studentt_deg).sample(torch.Size(csh)).to(device)
    noise = noise * scale / math.sqrt(float(shape[-1] * shape[-2]))
    noise = torch.tril(torch.view_as_complex(noise.contiguous()))
    with torch.autocast(device_type=noise.device.type, enabled=False):
        return isht(noise)
