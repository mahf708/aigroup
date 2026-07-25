"""Deterministic forcing channels appended to the projector's input.

The paper feeds the decoder the cosine zenith angle, surface geopotential, the
land-sea mask and the SST mask on top of the 75 prognostic channels, to absorb
the diurnal and geographic non-stationarity that the stationary-process
formulation would otherwise ignore.

For E3SM-style experiments two more kinds of forcing matter and are supported
here: explicit calendar encodings (needed when integrating for years rather
than days) and *scalar* forcings such as the CO2 concentration or the solar
constant, which are broadcast to full fields.  Those are what let the same
architecture be used for forced-response experiments rather than only
medium-range weather.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .spec import ForcingConfig, GridSpec

__all__ = ["cos_zenith_angle", "ForcingProvider"]

_J2000 = np.datetime64("2000-01-01T12:00:00", "s")


def _days_since_j2000(times: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype="datetime64[s]")
    return (t - _J2000).astype("float64") / 86400.0


def cos_zenith_angle(times: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Cosine of the solar zenith angle on a lat-lon grid.

    Standard low-precision solar position (mean longitude / mean anomaly /
    ecliptic longitude), which is accurate to well under a degree -- far below
    anything that matters as a network input.

    Parameters
    ----------
    times
        ``datetime64`` array of shape ``(T,)`` (or a scalar).
    lat, lon
        1-D degree coordinates of length ``nlat`` and ``nlon``.

    Returns
    -------
    ndarray
        ``(T, nlat, nlon)`` array in ``[-1, 1]``.
    """
    times = np.atleast_1d(np.asarray(times, dtype="datetime64[s]"))
    d = _days_since_j2000(times)

    g = np.deg2rad(np.mod(357.529 + 0.98560028 * d, 360.0))  # mean anomaly
    q = np.mod(280.459 + 0.98564736 * d, 360.0)  # mean longitude
    lam = np.deg2rad(np.mod(q + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g), 360.0))
    eps = np.deg2rad(23.439 - 3.6e-7 * d)  # obliquity

    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))

    # Greenwich mean sidereal time in radians
    gmst = np.deg2rad(np.mod(280.46061837 + 360.98564736629 * d, 360.0))

    lat_r = np.deg2rad(np.asarray(lat, dtype="float64"))[None, :, None]
    lon_r = np.deg2rad(np.asarray(lon, dtype="float64"))[None, None, :]
    hour_angle = gmst[:, None, None] + lon_r - ra[:, None, None]

    dec_b = dec[:, None, None]
    cosz = np.sin(lat_r) * np.sin(dec_b) + np.cos(lat_r) * np.cos(dec_b) * np.cos(hour_angle)
    return np.clip(cosz, -1.0, 1.0)


class ForcingProvider(nn.Module):
    """Assembles the extra input channels for a batch of valid times.

    Static fields are registered once (as normalised ``[-1, 1]`` maps) and
    scalar forcings are supplied per sample at call time, so a single trained
    model can be driven with different CO2 pathways or SST boundary conditions
    without touching the network.
    """

    def __init__(self, cfg: ForcingConfig, grid: GridSpec) -> None:
        super().__init__()
        self.cfg = cfg
        self.grid = grid
        self.static_names = list(cfg.static_fields)
        self.scalar_names = list(cfg.scalar_fields)
        nlat, nlon = grid.shape
        self.register_buffer(
            "static", torch.zeros(len(self.static_names), nlat, nlon), persistent=True
        )
        self._lat = grid.lat
        self._lon = grid.lon

    @property
    def n_channels(self) -> int:
        return self.cfg.n_channels()

    def set_static(self, name: str, field: torch.Tensor | np.ndarray) -> None:
        """Install one static map, rescaled to ``[-1, 1]``."""
        if name not in self.static_names:
            raise KeyError(f"{name!r} not in configured static fields {self.static_names}")
        f = torch.as_tensor(np.asarray(field), dtype=torch.float32)
        if tuple(f.shape) != self.grid.shape:
            raise ValueError(
                f"static field {name!r} has shape {tuple(f.shape)}, "
                f"expected {self.grid.shape}"
            )
        lo, hi = f.min(), f.max()
        f = 2 * (f - lo) / (hi - lo).clamp_min(1e-12) - 1 if hi > lo else torch.zeros_like(f)
        self.static[self.static_names.index(name)] = f

    def forward(
        self,
        times: np.ndarray | None,
        batch: int,
        device: torch.device | str = "cpu",
        scalars: dict[str, float | torch.Tensor] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """``(batch, n_channels, nlat, nlon)``; empty tensor when unconfigured."""
        nlat, nlon = self.grid.shape
        chans: list[torch.Tensor] = []

        if self.cfg.cos_zenith:
            if times is None:
                raise ValueError("cos_zenith forcing requires valid times")
            cz = cos_zenith_angle(times, self._lat, self._lon)
            chans.append(torch.as_tensor(cz, dtype=dtype, device=device).unsqueeze(1))

        if self.cfg.time_of_year or self.cfg.time_of_day:
            if times is None:
                raise ValueError("calendar forcing requires valid times")
            t = np.atleast_1d(np.asarray(times, dtype="datetime64[s]"))
            if self.cfg.time_of_year:
                year = t.astype("datetime64[Y]")
                doy = (t - year).astype("timedelta64[s]").astype("float64") / 86400.0
                ang = 2 * np.pi * doy / 365.25
                for v in (np.sin(ang), np.cos(ang)):
                    chans.append(
                        torch.as_tensor(v, dtype=dtype, device=device)
                        .reshape(-1, 1, 1, 1)
                        .expand(-1, 1, nlat, nlon)
                    )
            if self.cfg.time_of_day:
                day = t.astype("datetime64[D]")
                sod = (t - day).astype("timedelta64[s]").astype("float64")
                ang = 2 * np.pi * sod / 86400.0
                for v in (np.sin(ang), np.cos(ang)):
                    chans.append(
                        torch.as_tensor(v, dtype=dtype, device=device)
                        .reshape(-1, 1, 1, 1)
                        .expand(-1, 1, nlat, nlon)
                    )

        if self.static_names:
            s = self.static.to(device=device, dtype=dtype)
            chans.append(s.unsqueeze(0).expand(batch, -1, -1, -1))

        for name in self.scalar_names:
            val = (scalars or {}).get(name, 0.0)
            v = torch.as_tensor(val, dtype=dtype, device=device).reshape(-1)
            if v.numel() == 1:
                v = v.expand(batch)
            chans.append(v.reshape(-1, 1, 1, 1).expand(-1, 1, nlat, nlon))

        if not chans:
            return torch.zeros(batch, 0, nlat, nlon, device=device, dtype=dtype)
        out = [c if c.shape[0] == batch else c.expand(batch, -1, -1, -1) for c in chans]
        return torch.cat(out, dim=1)
