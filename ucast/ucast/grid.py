"""Lat/lon grid description and area weighting.

The whole package uses the layout ``(..., C, H, W)`` with ``H`` latitude (increasing, south to north)
and ``W`` longitude (increasing, periodic).  Convolutions wrap around ``W`` and are zero/replicate
padded along ``H``; area-weighted metrics use the latitude weights computed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def equiangular_with_poles(num_lat: int) -> np.ndarray:
    """Latitudes of an equiangular grid that includes both poles, e.g. 121 points for 1.5 degrees."""
    if num_lat < 2:
        raise ValueError(f"num_lat must be >= 2, got {num_lat}")
    return np.linspace(-90.0, 90.0, num_lat, dtype=np.float64)


def equiangular_without_poles(num_lat: int) -> np.ndarray:
    """Cell-centred latitudes that exclude the poles, e.g. 32 points for the 5.625 degree grid."""
    if num_lat < 1:
        raise ValueError(f"num_lat must be >= 1, got {num_lat}")
    delta = 180.0 / num_lat
    return np.linspace(-90.0 + delta / 2, 90.0 - delta / 2, num_lat, dtype=np.float64)


def regular_longitudes(num_lon: int) -> np.ndarray:
    """Longitudes in ``[0, 360)``."""
    if num_lon < 1:
        raise ValueError(f"num_lon must be >= 1, got {num_lon}")
    return np.linspace(0.0, 360.0, num_lon, endpoint=False, dtype=np.float64)


def latitude_cell_bounds(latitudes: np.ndarray) -> np.ndarray:
    """Cell edges (in radians) for latitude *centres*, clamped to the poles."""
    lat = np.deg2rad(np.asarray(latitudes, dtype=np.float64))
    if lat.size == 1:
        return np.array([-np.pi / 2, np.pi / 2])
    if not np.all(np.diff(lat) > 0):
        raise ValueError("latitudes must be strictly increasing (south to north)")
    interior = (lat[:-1] + lat[1:]) / 2
    return np.concatenate([[-np.pi / 2], interior, [np.pi / 2]])


def latitude_area_weights(latitudes: np.ndarray) -> np.ndarray:
    """Area weights per latitude, normalised to mean 1.

    The unnormalised weight of a cell is ``sin(upper) - sin(lower)``, i.e. the exact fraction of the
    sphere it covers.  This is the WeatherBench-2 convention, so scores are comparable to the
    published leaderboard.
    """
    bounds = latitude_cell_bounds(latitudes)
    weights = np.sin(bounds[1:]) - np.sin(bounds[:-1])
    if not np.all(weights > 0):
        raise ValueError(f"non-positive area weights derived from latitudes {latitudes}")
    return weights / weights.mean()


@dataclass(frozen=True)
class Grid:
    """A regular lat/lon grid.

    Args:
        latitudes: Strictly increasing latitudes in degrees, length ``H``.
        longitudes: Increasing longitudes in degrees, length ``W``.
        periodic_longitude: Whether ``W`` wraps around (true for a global grid, false for a region).
    """

    latitudes: tuple[float, ...]
    longitudes: tuple[float, ...]
    periodic_longitude: bool = True

    def __post_init__(self) -> None:
        latitude_cell_bounds(np.asarray(self.latitudes))  # validates monotonicity / range

    @classmethod
    def from_arrays(cls, latitudes, longitudes, periodic_longitude: bool = True) -> "Grid":
        lat = np.asarray(latitudes, dtype=np.float64).ravel()
        lon = np.asarray(longitudes, dtype=np.float64).ravel()
        if lat.size > 1 and lat[0] > lat[-1]:
            lat = lat[::-1]
        return cls(tuple(lat.tolist()), tuple(lon.tolist()), periodic_longitude)

    @classmethod
    def equiangular(cls, num_lat: int, num_lon: int, with_poles: bool = True, **kwargs) -> "Grid":
        """Standard global grid, e.g. ``Grid.equiangular(121, 240)`` for ERA5 at 1.5 degrees."""
        lat = equiangular_with_poles(num_lat) if with_poles else equiangular_without_poles(num_lat)
        return cls.from_arrays(lat, regular_longitudes(num_lon), **kwargs)

    # ------------------------------------------------------------------ Shape helpers
    @property
    def num_lat(self) -> int:
        return len(self.latitudes)

    @property
    def num_lon(self) -> int:
        return len(self.longitudes)

    @property
    def shape(self) -> tuple[int, int]:
        """``(H, W)``."""
        return self.num_lat, self.num_lon

    # ------------------------------------------------------------------ Weights
    def latitude_weights(self) -> np.ndarray:
        """Area weights of shape ``(H,)``, mean 1."""
        return latitude_area_weights(np.asarray(self.latitudes))

    def area_weights(self, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
        """Area weights broadcast to ``(H, 1)`` so they multiply straight onto ``(..., C, H, W)``."""
        w = torch.as_tensor(self.latitude_weights(), dtype=dtype, device=device)
        return w.reshape(self.num_lat, 1)

    def channel_area_weights(
        self,
        channel_weights: np.ndarray | torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        device=None,
    ) -> torch.Tensor:
        """Combined ``(C, H, 1)`` (or ``(1, H, 1)``) weights for area- and variable-weighted losses.

        The result has mean 1 over the grid for each channel, so the weighted loss stays on the same
        scale as its unweighted counterpart.
        """
        area = self.area_weights(dtype=dtype, device=device).unsqueeze(0)  # (1, H, 1)
        if channel_weights is None:
            return area
        cw = torch.as_tensor(np.asarray(channel_weights), dtype=dtype, device=device).reshape(-1, 1, 1)
        return cw * area

    def to_dict(self) -> dict[str, object]:
        return {
            "latitudes": list(self.latitudes),
            "longitudes": list(self.longitudes),
            "periodic_longitude": self.periodic_longitude,
        }
