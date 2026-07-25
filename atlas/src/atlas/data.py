"""Data access.

A *store* is anything that can hand back a ``(C, nlat, nlon)`` array for a
time index, so the model is agnostic to whether the archive is ERA5 in zarr,
E3SM history files on a regridded lat-lon mesh, or a synthetic test signal.

:class:`WindowDataset` turns a store into the ``(history window, next state)``
pairs the model trains on.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from .spec import Channel, GridSpec, VariableSet

__all__ = [
    "StateStore",
    "SyntheticStore",
    "XarrayStore",
    "WindowDataset",
    "collate",
    "resolve_channel",
]


class StateStore(Protocol):
    """Minimal read interface the training/inference code depends on."""

    variables: VariableSet
    grid: GridSpec
    times: np.ndarray

    def __len__(self) -> int: ...

    def read(self, index: int) -> np.ndarray:
        """``(C, nlat, nlon)`` float32 array in physical units."""
        ...


# ---------------------------------------------------------------------------
# synthetic
# ---------------------------------------------------------------------------


class SyntheticStore:
    """Advecting Rossby-wave-like fields; lets the whole stack run with no data.

    Each channel is a superposition of a few zonally propagating harmonics
    with channel-dependent phase speed and a slow seasonal modulation.  It is
    not physics, but it has the properties the machinery cares about: spatial
    correlation, a red spectrum, a diurnal/annual cycle, and a genuine
    dependence of the increment on the two most recent states.
    """

    def __init__(
        self,
        variables: VariableSet,
        grid: GridSpec,
        n_times: int = 256,
        dt_hours: float = 6.0,
        start: str = "2000-01-01",
        seed: int = 0,
    ) -> None:
        self.variables = variables
        self.grid = grid
        self.dt_hours = dt_hours
        self.times = np.datetime64(start, "s") + np.arange(n_times) * np.timedelta64(
            int(dt_hours * 3600), "s"
        )
        rng = np.random.default_rng(seed)
        c = len(variables)
        self._k = rng.integers(1, 6, size=(c, 3))
        self._l = rng.integers(1, 4, size=(c, 3))
        self._speed = rng.uniform(-0.4, 0.4, size=(c, 3))
        self._amp = rng.uniform(0.3, 1.0, size=(c, 3)) / np.arange(1, 4)[None, :]
        self._offset = rng.normal(0, 1, size=c) * 10.0
        self._scale = rng.uniform(0.5, 5.0, size=c)
        lon = np.deg2rad(grid.lon)[None, :]
        lat = np.deg2rad(grid.lat)[:, None]
        self._lon = lon
        self._lat = lat

    def __len__(self) -> int:
        return int(self.times.size)

    def read(self, index: int) -> np.ndarray:
        t = index * self.dt_hours / 24.0
        season = np.sin(2 * np.pi * t / 365.25)
        out = np.zeros((len(self.variables), *self.grid.shape), dtype=np.float32)
        for c in range(out.shape[0]):
            f = np.zeros(self.grid.shape)
            for m in range(3):
                f += self._amp[c, m] * np.cos(
                    self._k[c, m] * (self._lon - self._speed[c, m] * t)
                ) * np.cos(self._l[c, m] * self._lat) ** 2
            f = f * (1.0 + 0.3 * season)
            out[c] = self._offset[c] + self._scale[c] * f
        return out


# ---------------------------------------------------------------------------
# xarray
# ---------------------------------------------------------------------------

_PRESSURE_RE = re.compile(r"^([a-z]+)(\d{2,4})$")
_LEVEL_RE = re.compile(r"^(.+?)_lev(\d+)$")


@dataclass
class ChannelRef:
    """Where a channel lives in a dataset."""

    var: str
    dim: str | None = None
    value: Any = None
    by_index: bool = False


def resolve_channel(
    ch: Channel,
    overrides: dict[str, ChannelRef] | None = None,
    level_dim: str | None = None,
) -> ChannelRef:
    """Map a :class:`~atlas.spec.Channel` onto a dataset variable and level.

    Handles the two naming conventions the package ships with:
    ``t850``-style ERA5 pressure-level names and ``T_lev07``-style E3SM
    hybrid-level names.  Anything else should be supplied through
    ``overrides``.
    """
    if overrides and ch.name in overrides:
        return overrides[ch.name]
    m = _LEVEL_RE.match(ch.name)
    if m:
        return ChannelRef(m.group(1), level_dim or "lev", int(m.group(2)), by_index=True)
    if ch.level_kind == "pressure" and ch.level is not None:
        m = _PRESSURE_RE.match(ch.name)
        var = m.group(1) if m else ch.name
        return ChannelRef(var, level_dim or "level", float(ch.level))
    return ChannelRef(ch.name)


class XarrayStore:
    """Read channels out of an ``xarray.Dataset``.

    The dataset may be lazily backed (zarr, netCDF, kerchunk); only the
    requested time slice is loaded.  Latitude is reordered to match the
    :class:`~atlas.spec.GridSpec` and longitude is rolled to start at the
    grid's first value, so ERA5 (90 -> -90, 0 -> 360) and E3SM
    (-90 -> 90, 0 -> 360) archives can back the same model.
    """

    def __init__(
        self,
        dataset: Any,
        variables: VariableSet,
        grid: GridSpec | None = None,
        time_dim: str = "time",
        lat_dim: str = "lat",
        lon_dim: str = "lon",
        level_dim: str | None = None,
        overrides: dict[str, ChannelRef] | None = None,
        fill_value: float = 0.0,
    ) -> None:
        self.ds = dataset
        self.variables = variables
        self.time_dim = time_dim
        self.lat_dim = lat_dim
        self.lon_dim = lon_dim
        self.fill_value = fill_value
        self.refs = [resolve_channel(c, overrides, level_dim) for c in variables]

        lat = np.asarray(dataset[lat_dim].values, dtype=np.float64)
        lon = np.mod(np.asarray(dataset[lon_dim].values, dtype=np.float64), 360.0)
        self.grid = grid or GridSpec(lat=lat, lon=lon)
        self._flip_lat = bool((lat[0] > lat[-1]) != self.grid.descending_lat)
        self._roll = int(-np.argmin(np.abs(lon - self.grid.lon[0])))
        self.times = np.asarray(dataset[time_dim].values, dtype="datetime64[s]")

    def __len__(self) -> int:
        return int(self.times.size)

    def read(self, index: int) -> np.ndarray:
        out = np.empty((len(self.variables), *self.grid.shape), dtype=np.float32)
        for i, ref in enumerate(self.refs):
            if ref.var not in self.ds:
                out[i] = self.fill_value
                continue
            da = self.ds[ref.var].isel({self.time_dim: index})
            if ref.dim is not None and ref.dim in da.dims:
                da = (
                    da.isel({ref.dim: int(ref.value)})
                    if ref.by_index
                    else da.sel({ref.dim: ref.value}, method="nearest")
                )
            arr = np.asarray(da.values, dtype=np.float32)
            if self._flip_lat:
                arr = arr[::-1]
            if self._roll:
                arr = np.roll(arr, self._roll, axis=-1)
            out[i] = np.nan_to_num(arr, nan=self.fill_value)
        return out


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------


class WindowDataset(Dataset):
    """``(history window, next state)`` pairs drawn from a store.

    Parameters
    ----------
    store
        Anything satisfying :class:`StateStore`.
    history
        Number of *additional* past states beyond the current one; the window
        has ``history + 1`` entries.  The paper uses 1.
    stride
        Spacing (in store indices) between the states of a window.  Use this to
        train a 12-hour model on 6-hourly data without resampling the archive.
    indices
        Restrict to these window start indices, e.g. to hold out a year.
    """

    def __init__(
        self,
        store: StateStore,
        history: int = 1,
        stride: int = 1,
        indices: Sequence[int] | None = None,
    ) -> None:
        self.store = store
        self.history = history
        self.stride = stride
        self.span = (history + 1) * stride
        n = len(store)
        valid = range(0, n - self.span)
        self.indices = list(indices) if indices is not None else list(valid)
        if any(i < 0 or i + self.span >= n for i in self.indices):
            raise ValueError("window indices out of range for this store")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, Any]:
        start = self.indices[i]
        offs = [start + j * self.stride for j in range(self.history + 1)]
        window = np.stack([self.store.read(o) for o in offs])
        nxt = self.store.read(offs[-1] + self.stride)
        return {
            "window": torch.from_numpy(window),
            "next": torch.from_numpy(nxt),
            "time": self.store.times[offs[-1]],
            "index": start,
        }

    def stat_windows(self, stride: int = 1) -> Iterator[np.ndarray]:
        """Windows as raw arrays, for :func:`atlas.normalize.compute_statistics`."""
        for i in range(0, len(self.indices), stride):
            start = self.indices[i]
            yield np.stack(
                [self.store.read(start), self.store.read(start + self.stride)]
            )


def collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Default collate that keeps ``datetime64`` times as a numpy array."""
    return {
        "window": torch.stack([b["window"] for b in batch]),
        "next": torch.stack([b["next"] for b in batch]),
        "time": np.array([b["time"] for b in batch], dtype="datetime64[s]"),
        "index": np.array([b["index"] for b in batch]),
    }
