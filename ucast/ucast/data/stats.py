"""Load normalisation statistics from NetCDF files, or derive them from a dataset.

WeatherBench-2 style statistics files hold one variable per data variable, with atmospheric variables
carrying a ``level`` coordinate -- the same layout as the ``era5_mean.nc`` / ``era5_std.nc`` /
``era5_residual_std.nc`` files shipped with the reference U-Cast implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..normalization import Normalizer, compute_statistics
from ..utils import get_logger
from ..variables import VariableSet

__all__ = ["normalizer_from_netcdf", "normalizer_from_directory", "build_normalizer"]

log = get_logger(__name__)


def _scalar_lookup(path: str | Path, variables: VariableSet, level_name: str = "level") -> dict[str, float]:
    from .xarray_dataset import open_xarray  # local import: xarray is an optional dependency

    with open_xarray(str(path)) as dataset:
        out: dict[str, float] = {}
        for var in variables:
            if var.key in dataset:
                values = dataset[var.key].values
            elif var.name in dataset:
                array = dataset[var.name]
                values = array.sel({level_name: var.level}).values if var.level is not None else array.values
            else:
                raise KeyError(f"{Path(path).name} has no entry for {var.key!r}")
            flat = np.asarray(values, dtype=np.float64).ravel()
            if flat.size != 1:
                raise ValueError(f"{var.key!r} in {Path(path).name} is not a scalar (shape {np.shape(values)})")
            out[var.key] = float(flat[0])
    return out


def normalizer_from_netcdf(
    variables: VariableSet,
    mean_path: str | Path,
    std_path: str | Path,
    residual_std_path: str | Path | None = None,
    level_name: str = "level",
) -> Normalizer:
    """Build a :class:`~ucast.normalization.Normalizer` from three statistics files."""
    mean = _scalar_lookup(mean_path, variables, level_name)
    std = _scalar_lookup(std_path, variables, level_name)
    residual = _scalar_lookup(residual_std_path, variables, level_name) if residual_std_path is not None else None
    if residual is None:
        log.warning("No residual-std statistics given; the model will predict normalised increments directly.")
    return Normalizer.from_mapping(variables, mean=mean, std=std, residual_std=residual)


def _find_stats_file(directory: Path, prefix: str, kind: str) -> Path | None:
    for suffix in (".nc", ".zarr", ".nc4"):
        candidate = directory / f"{prefix}_{kind}{suffix}"
        if candidate.exists():
            return candidate
    return None


def normalizer_from_directory(
    variables: VariableSet,
    directory: str | Path,
    prefix: str = "era5",
    level_name: str = "level",
) -> Normalizer:
    """Load ``<prefix>_mean``, ``<prefix>_std`` and (if present) ``<prefix>_residual_std``.

    Matches the layout shipped with the reference implementation (``era5_mean.nc`` and friends); zarr
    stores with the same names also work.
    """
    directory = Path(directory)
    mean_path = _find_stats_file(directory, prefix, "mean")
    std_path = _find_stats_file(directory, prefix, "std")
    if mean_path is None or std_path is None:
        raise FileNotFoundError(f"could not find {prefix}_mean / {prefix}_std statistics under {directory}")
    return normalizer_from_netcdf(
        variables,
        mean_path,
        std_path,
        _find_stats_file(directory, prefix, "residual_std"),
        level_name=level_name,
    )


def build_normalizer(
    variables: VariableSet,
    source: str | None,
    dataset: Any | None = None,
    max_samples: int | None = 256,
) -> Normalizer:
    """Resolve the ``data.statistics`` config entry into a normalizer.

    ``source`` may be:

    * ``None`` or ``"compute"`` -- estimate statistics by streaming over ``dataset``,
    * ``"identity"`` -- no normalisation (data is already standardised),
    * a path to a ``.npz`` file written by :meth:`ucast.normalization.Normalizer.save`,
    * a directory holding WeatherBench-2 style ``*_mean.nc`` / ``*_std.nc`` files.
    """
    if source in (None, "compute", "auto"):
        if dataset is None:
            raise ValueError("statistics='compute' needs a dataset to stream over")
        log.info("Computing normalisation statistics from up to %s dataset items...", max_samples)
        return compute_statistics(dataset, variables, max_samples=max_samples)
    if source == "identity":
        return Normalizer.identity(variables)
    path = Path(source)
    if path.is_dir():
        return normalizer_from_directory(variables, path)
    if path.suffix == ".npz":
        return Normalizer.load(path, variables)
    raise ValueError(
        f"cannot interpret statistics={source!r}: expected 'compute', 'identity', a .npz file or a directory"
    )
