"""Clock forcings: where the model learns the diurnal and seasonal cycles from.

Two scalars per time step -- progress through the local day and through the year -- each encoded as a
sin/cos pair so they are continuous across midnight and New Year.  Day progress is longitude-dependent
(local solar time), year progress is not.  Same construction as GraphCast/GenCast.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SECONDS_PER_DAY", "clock_forcings", "CLOCK_FORCING_NAMES", "year_progress", "day_progress"]

SECONDS_PER_DAY = 86_400
_MEAN_DAYS_PER_YEAR = 365.24219
SECONDS_PER_YEAR = SECONDS_PER_DAY * _MEAN_DAYS_PER_YEAR

CLOCK_FORCING_NAMES: tuple[str, ...] = (
    "year_progress_sin",
    "year_progress_cos",
    "day_progress_sin",
    "day_progress_cos",
)


def year_progress(seconds_since_epoch: np.ndarray) -> np.ndarray:
    """Fraction through the (mean tropical) year, in ``[0, 1)``.  Shape ``(T,)``."""
    seconds = np.asarray(seconds_since_epoch, dtype=np.float64)
    return np.mod(seconds / SECONDS_PER_YEAR, 1.0)


def day_progress(seconds_since_epoch: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Fraction through the local day at each longitude, in ``[0, 1)``.  Shape ``(T, W)``."""
    seconds = np.asarray(seconds_since_epoch, dtype=np.float64)
    greenwich = np.mod(seconds, SECONDS_PER_DAY) / SECONDS_PER_DAY
    offsets = np.deg2rad(np.asarray(longitudes, dtype=np.float64)) / (2 * np.pi)
    return np.mod(greenwich[:, None] + offsets[None, :], 1.0)


def clock_forcings(
    seconds_since_epoch: np.ndarray,
    num_lat: int,
    longitudes: np.ndarray,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Clock forcing fields of shape ``(T, 4, num_lat, len(longitudes))``.

    Channel order matches :data:`CLOCK_FORCING_NAMES`.
    """
    seconds = np.asarray(seconds_since_epoch, dtype=np.float64).reshape(-1)
    num_time = seconds.size
    num_lon = len(longitudes)

    year_phase = year_progress(seconds) * (2 * np.pi)  # (T,)
    day_phase = day_progress(seconds, longitudes) * (2 * np.pi)  # (T, W)

    out = np.empty((num_time, 4, num_lat, num_lon), dtype=dtype)
    out[:, 0] = np.sin(year_phase)[:, None, None]
    out[:, 1] = np.cos(year_phase)[:, None, None]
    out[:, 2] = np.sin(day_phase)[:, None, :]
    out[:, 3] = np.cos(day_phase)[:, None, :]
    return out
