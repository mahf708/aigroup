"""Dataset backed by an xarray-readable archive (WeatherBench-2 zarr, NetCDF, E3SM history files).

This is the adapter that turns "a directory of gridded data" into the tensor contract in
:mod:`ucast.data.base`.  It is written against xarray rather than against ERA5 specifically: variable
names, level coordinate, dimension order, time subsampling and NaN handling are all configuration, so
the same class serves the WeatherBench-2 1.5-degree ERA5 zarr the paper uses and, say, a regridded
E3SM run.

``xarray`` (plus ``zarr`` for zarr stores) is an optional dependency; install with
``pip install -e '.[data]'``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..grid import Grid
from ..utils import get_logger
from ..variables import VariableSet
from .base import DatasetSpec, ForecastDataset
from .forcings import clock_forcings

__all__ = ["XarrayForecastDataset", "open_xarray"]

log = get_logger(__name__)


def open_xarray(
    path: str,
    engine: str | None = None,
    storage_options: Mapping[str, Any] | None = None,
    chunks: Any = None,
):
    """Open a zarr store or any xarray-readable file, tolerating both zarr v2 and v3 layouts.

    ``chunks`` defaults to ``None``: xarray's own lazy indexing reads exactly the window each item
    needs, and no dask dependency is pulled in.  Pass ``chunks={}`` (with dask installed) if you would
    rather have dask-backed arrays.
    """
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "XarrayForecastDataset needs xarray (and zarr for .zarr stores): pip install -e '.[data]'"
        ) from exc

    is_zarr = engine == "zarr" or (engine is None and str(path).rstrip("/").endswith(".zarr"))
    if not is_zarr:
        return xr.open_dataset(path, engine=engine, chunks=chunks)
    kwargs: dict[str, Any] = {"chunks": chunks}
    if storage_options:
        kwargs["storage_options"] = dict(storage_options)
    errors = []
    for attempt in ({"consolidated": True}, {"consolidated": False}):
        try:
            return xr.open_zarr(path, **attempt, **kwargs)
        except Exception as exc:  # noqa: BLE001 - fall through to the next strategy
            errors.append(exc)
    raise RuntimeError(f"could not open zarr store {path!r}: {errors[-1]}") from errors[-1]


class XarrayForecastDataset(ForecastDataset):
    """Forecast cases sliced out of an xarray dataset.

    Args:
        path: Zarr store, NetCDF file, or anything ``engine`` can open (``gs://`` URLs work with gcsfs).
        variables: Input variables as flat keys (``"temperature_850"``) or mappings.
        output_variables: Predicted subset; defaults to ``variables``.
        window / rollout_steps: Input frames and target frames per item.
        time_range: ``(start, stop)`` passed to ``.sel(time=slice(...))``; either may be ``None``.
        step_hours: Spacing of the frames the model sees.  If the archive is finer, it is subsampled by
            an integer stride (e.g. 12-hourly frames from 6-hourly ERA5, as in the paper).
        initial_hours: Restrict initial conditions to these UTC hours (``[0, 12]`` reproduces the
            WeatherBench-2 evaluation protocol).  ``None`` uses every available frame -- what you want
            for training.
        static_fields: Time-invariant variables to pass as static conditioning; standardised per field.
        clock_forcings_enabled: Add the four sin/cos clock channels.
        extra_forcing_fields: Additional time-varying variables to pass as forcings (e.g.
            ``toa_incident_solar_radiation``), standardised with the given mean/std or per-field stats.
        fill_values: NaN handling per variable name: a float, ``"min"`` or ``"mean"`` (computed per
            frame over valid points).  WeatherBench-2 SST is NaN over land, so
            ``{"sea_surface_temperature": "min"}`` mirrors the reference implementation.
        allow_nans: Skip the "no NaNs remain" check (off by default -- silent NaNs poison training).
        max_items: Subsample the item list to at most this many, evenly spaced.  Handy for quick
            validation passes.
        engine / storage_options: Passed to :func:`open_xarray`.
        dim_names: Overrides for the coordinate names, e.g. ``{"latitude": "lat"}``.
        periodic_longitude: Whether longitude wraps (False for a regional subset).
    """

    def __init__(
        self,
        path: str,
        variables: Sequence[Any],
        output_variables: Sequence[Any] | None = None,
        window: int = 2,
        rollout_steps: int = 1,
        time_range: Sequence[str | None] | None = None,
        step_hours: int = 12,
        initial_hours: Sequence[int] | None = None,
        static_fields: Sequence[str] = ("land_sea_mask", "geopotential_at_surface"),
        clock_forcings_enabled: bool = True,
        extra_forcing_fields: Sequence[str] = (),
        fill_values: Mapping[str, float | str] | None = None,
        allow_nans: bool = False,
        max_items: int | None = None,
        engine: str | None = None,
        storage_options: Mapping[str, Any] | None = None,
        dim_names: Mapping[str, str] | None = None,
        periodic_longitude: bool = True,
    ):
        super().__init__()
        names = {"time": "time", "latitude": "latitude", "longitude": "longitude", "level": "level"}
        names.update(dim_names or {})
        self._names = names
        self.path = path
        self.fill_values = dict(fill_values or {})
        self.allow_nans = allow_nans
        self.rollout_steps = int(rollout_steps)

        dataset = open_xarray(path, engine=engine, storage_options=storage_options)
        if time_range is not None:
            start, stop = (time_range[0], time_range[1])
            dataset = dataset.sel({names["time"]: slice(start, stop)})
        dataset = self._orient(dataset)
        dataset, stride = self._subsample_time(dataset, step_hours)
        self.stride_hours = step_hours
        self._dataset = dataset

        input_vars = VariableSet(variables)
        out_vars = VariableSet(output_variables) if output_variables is not None else input_vars
        grid = Grid.from_arrays(
            dataset[names["latitude"]].values,
            dataset[names["longitude"]].values,
            periodic_longitude=periodic_longitude,
        )
        self._forcing_names: list[str] = []
        if clock_forcings_enabled:
            self._forcing_names += ["year_progress_sin", "year_progress_cos", "day_progress_sin", "day_progress_cos"]
        self._extra_forcing_fields = list(extra_forcing_fields)
        self._forcing_names += self._extra_forcing_fields

        self.spec = DatasetSpec(
            input_variables=input_vars,
            output_variables=out_vars,
            grid=grid,
            window=int(window),
            num_forcing_channels=len(self._forcing_names),
            num_static_channels=len(static_fields),
            step_hours=int(step_hours),
        )

        self._times = dataset[names["time"]].values
        self._seconds = self._times.astype("datetime64[s]").astype(np.int64)
        self._plan = self._build_extraction_plan(input_vars)
        self._statics = self._load_statics(list(static_fields))
        self._extra_forcing_stats = self._load_forcing_stats(self._extra_forcing_fields)
        self._indices = self._valid_indices(initial_hours, max_items)
        log.info(
            "%s: %d items from %s (%s .. %s, every %dh)",
            type(self).__name__,
            len(self._indices),
            path,
            str(self._times[0]) if len(self._times) else "-",
            str(self._times[-1]) if len(self._times) else "-",
            step_hours,
        )
        if stride > 1:
            log.info("Subsampled the archive by a stride of %d to reach %dh frames.", stride, step_hours)

    # ------------------------------------------------------------------ setup helpers
    def _orient(self, dataset):
        """Make latitude ascending so area weights and padding conventions hold."""
        lat_name = self._names["latitude"]
        lat = dataset[lat_name].values
        if lat.size > 1 and lat[0] > lat[-1]:
            dataset = dataset.isel({lat_name: slice(None, None, -1)})
        return dataset

    def _subsample_time(self, dataset, step_hours: int):
        time_name = self._names["time"]
        times = dataset[time_name].values
        if times.size < 2:
            return dataset, 1
        delta = (times[1] - times[0]).astype("timedelta64[h]").astype(int)
        if delta <= 0:
            raise ValueError("the time coordinate must be strictly increasing")
        if step_hours % delta != 0:
            raise ValueError(f"step_hours={step_hours} is not a multiple of the archive's {delta}h spacing")
        stride = step_hours // delta
        return (dataset.isel({time_name: slice(None, None, stride)}) if stride > 1 else dataset), stride

    def _build_extraction_plan(self, variables: VariableSet) -> list[tuple[str, list[int] | None, list[int]]]:
        """Group the requested channels by source variable: ``(name, levels, channel_positions)``."""
        surface: dict[str, list[int]] = defaultdict(list)
        atmospheric: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for channel, var in enumerate(variables):
            if var.level is None:
                surface[var.name].append(channel)
            else:
                atmospheric[var.name].append((var.level, channel))

        available = set(self._dataset.data_vars)
        missing = [n for n in (*surface, *atmospheric) if n not in available]
        if missing:
            raise KeyError(f"{self.path} has no variables named {missing}; available: {sorted(available)[:20]}")

        plan: list[tuple[str, list[int] | None, list[int]]] = []
        for name, channels in surface.items():
            plan.append((name, None, channels))
        for name, pairs in atmospheric.items():
            levels = [level for level, _ in pairs]
            channels = [channel for _, channel in pairs]
            plan.append((name, levels, channels))
        return plan

    def _spatial_transpose(self, array):
        lat, lon = self._names["latitude"], self._names["longitude"]
        leading = [d for d in array.dims if d not in (lat, lon)]
        return array.transpose(*leading, lat, lon)

    def _load_statics(self, static_fields: Sequence[str]) -> torch.Tensor | None:
        if not static_fields:
            return None
        arrays = []
        for name in static_fields:
            if name not in self._dataset:
                raise KeyError(f"static field {name!r} not found in {self.path}")
            values = self._spatial_transpose(self._dataset[name]).values
            values = np.asarray(values, dtype=np.float64)
            if values.ndim != 2:
                raise ValueError(f"static field {name!r} should be 2-D, got shape {values.shape}")
            arrays.append(values)
        statics = np.stack(arrays)
        mean = statics.mean(axis=(-2, -1), keepdims=True)
        std = statics.std(axis=(-2, -1), keepdims=True)
        statics = (statics - mean) / np.clip(std, 1e-6, None)
        return torch.from_numpy(statics.astype(np.float32))

    def _load_forcing_stats(self, fields: Sequence[str]) -> dict[str, tuple[float, float]]:
        """Mean/std of extra forcing fields, estimated from a handful of frames."""
        stats: dict[str, tuple[float, float]] = {}
        if not fields:
            return stats
        time_name = self._names["time"]
        probe_count = min(len(self._dataset[time_name]), 64)
        probe = self._dataset.isel({time_name: slice(0, probe_count)})
        for name in fields:
            if name not in self._dataset:
                raise KeyError(f"forcing field {name!r} not found in {self.path}")
            values = np.asarray(probe[name].values, dtype=np.float64)
            stats[name] = (float(np.nanmean(values)), float(max(np.nanstd(values), 1e-6)))
        return stats

    def _valid_indices(self, initial_hours: Sequence[int] | None, max_items: int | None) -> np.ndarray:
        last = len(self._times) - self.frames_per_item
        if last < 0:
            raise ValueError(
                f"the selected time range holds {len(self._times)} frames, fewer than one item "
                f"({self.frames_per_item} frames)"
            )
        indices = np.arange(last + 1)
        if initial_hours is not None:
            ic_times = self._times[indices + self.spec.window - 1]
            hours = ic_times.astype("datetime64[h]").astype(np.int64) % 24
            indices = indices[np.isin(hours, np.asarray(list(initial_hours), dtype=np.int64))]
        if max_items is not None and len(indices) > max_items:
            picks = np.linspace(0, len(indices) - 1, num=max_items).round().astype(int)
            indices = indices[np.unique(picks)]
        if len(indices) == 0:
            raise ValueError("no valid initial conditions remain after filtering")
        return indices

    # ------------------------------------------------------------------ NaN handling
    def _fill(self, values: np.ndarray, name: str) -> np.ndarray:
        rule = self.fill_values.get(name)
        if rule is None:
            return values
        mask = np.isnan(values)
        if not mask.any():
            return values
        if isinstance(rule, str):
            reducer = {"min": np.nanmin, "mean": np.nanmean, "max": np.nanmax}.get(rule)
            if reducer is None:
                raise ValueError(f"fill rule for {name!r} must be a number, 'min', 'mean' or 'max', got {rule!r}")
            axes = tuple(range(values.ndim - 2, values.ndim))
            replacement = np.broadcast_to(reducer(values, axis=axes, keepdims=True), values.shape)
            return np.where(mask, replacement, values)
        return np.where(mask, float(rule), values)

    # ------------------------------------------------------------------ Dataset protocol
    def __len__(self) -> int:
        return len(self._indices)

    def get_item(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self._indices[index])
        stop = start + self.frames_per_item
        time_name = self._names["time"]
        level_name = self._names["level"]
        window = self._dataset.isel({time_name: slice(start, stop)})

        height, width = self.spec.grid.shape
        dynamics = np.empty((self.frames_per_item, len(self.spec.input_variables), height, width), dtype=np.float32)
        for name, levels, channels in self._plan:
            array = window[name]
            if levels is not None:
                array = array.sel({level_name: levels})
            values = np.asarray(self._spatial_transpose(array).load().values, dtype=np.float32)
            values = self._fill(values, name)
            if levels is None:
                dynamics[:, channels[0]] = values
            else:
                for position, channel in enumerate(channels):
                    dynamics[:, channel] = values[:, position]
        if not self.allow_nans and np.isnan(dynamics).any():
            bad = {
                var.key
                for var, has_nan in zip(self.spec.input_variables, np.isnan(dynamics).any(axis=(0, 2, 3)))
                if has_nan
            }
            raise ValueError(
                f"NaNs remain in {sorted(bad)} at item {index}. Set fill_values for these variables "
                "(e.g. {'sea_surface_temperature': 'min'}) or pass allow_nans=True."
            )

        item: dict[str, torch.Tensor] = {
            "dynamics": torch.from_numpy(dynamics),
            "time": torch.tensor(int(self._seconds[start + self.spec.window - 1]), dtype=torch.int64),
        }
        forcings = self._build_forcings(window, start, stop)
        if forcings is not None:
            item["forcings"] = forcings
        if self._statics is not None:
            item["statics"] = self._statics.clone()
        return item

    def _build_forcings(self, window, start: int, stop: int) -> torch.Tensor | None:
        if not self._forcing_names:
            return None
        height, width = self.spec.grid.shape
        channels: list[np.ndarray] = []
        if "year_progress_sin" in self._forcing_names:
            channels.append(
                clock_forcings(self._seconds[start:stop], height, np.asarray(self.spec.grid.longitudes))
            )
        for name in self._extra_forcing_fields:
            values = np.asarray(self._spatial_transpose(window[name]).load().values, dtype=np.float32)
            mean, std = self._extra_forcing_stats[name]
            values = (np.nan_to_num(values, nan=mean) - mean) / std
            channels.append(values[:, None])
        stacked = np.concatenate(channels, axis=1) if len(channels) > 1 else channels[0]
        return torch.from_numpy(np.ascontiguousarray(stacked, dtype=np.float32))

    # ------------------------------------------------------------------ extras
    def initial_times(self) -> np.ndarray:
        """Initial-condition timestamps, one per item."""
        return self._times[self._indices + self.spec.window - 1]
