"""Datasets and dataloaders.

Datasets are named either by a registry key or by a dotted import path, so a new data source is a
class plus one config line:

.. code-block:: yaml

    data:
      builder: my_project.e3sm_data:E3SMForecastDataset
      common: {path: /lcrc/group/e3sm/...}

Any dataset built this way should accept the keyword arguments ``variables``, ``output_variables``,
``window`` and ``rollout_steps`` (the loader injects them from the experiment config) and subclass
:class:`~ucast.data.base.ForecastDataset`.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from ..config import DataConfig, ExperimentConfig
from ..utils import get_logger, import_object
from .base import DatasetSpec, ForecastDataset
from .forcings import CLOCK_FORCING_NAMES, clock_forcings
from .stats import build_normalizer, normalizer_from_directory, normalizer_from_netcdf
from .synthetic import SyntheticForecastDataset

__all__ = [
    "CLOCK_FORCING_NAMES",
    "DATASET_REGISTRY",
    "DatasetSpec",
    "ForecastDataset",
    "SyntheticForecastDataset",
    "accumulation_steps",
    "build_dataloader",
    "build_dataset",
    "build_datasets",
    "build_normalizer",
    "clock_forcings",
    "normalizer_from_directory",
    "normalizer_from_netcdf",
]

log = get_logger(__name__)

#: Short names accepted by ``data.builder``.
DATASET_REGISTRY: dict[str, str] = {
    "synthetic": "ucast.data.synthetic:SyntheticForecastDataset",
    "xarray": "ucast.data.xarray_dataset:XarrayForecastDataset",
    "zarr": "ucast.data.xarray_dataset:XarrayForecastDataset",
    "era5": "ucast.data.xarray_dataset:XarrayForecastDataset",
}


def build_dataset(builder: str, **kwargs) -> ForecastDataset:
    """Instantiate a dataset from a registry key or a dotted path."""
    target = DATASET_REGISTRY.get(builder, builder)
    factory = import_object(target)
    dataset = factory(**kwargs)
    if not isinstance(dataset, ForecastDataset):
        raise TypeError(f"{target} produced {type(dataset).__name__}, which is not a ForecastDataset")
    return dataset


def build_datasets(config: ExperimentConfig, splits: tuple[str, ...] = ("train", "val")) -> dict[str, ForecastDataset]:
    """Build the requested splits, injecting variables/window/rollout from the experiment config.

    Split kwargs take precedence over the injected values, so a config can always override.
    """
    datasets: dict[str, ForecastDataset] = {}
    for split in splits:
        kwargs: dict[str, Any] = dict(config.data.split_kwargs(split))
        if config.input_variables and "variables" not in kwargs:
            kwargs["variables"] = list(config.input_variables)
        if config.output_variables and "output_variables" not in kwargs:
            kwargs["output_variables"] = list(config.output_variables)
        kwargs.setdefault("window", config.window)
        kwargs.setdefault("rollout_steps", 1 if split == "train" else config.eval.rollout_steps)
        datasets[split] = build_dataset(config.data.builder, **kwargs)
        log.info("%s split -> %s", split, datasets[split].describe())
    return datasets


def build_dataloader(
    dataset: Dataset,
    config: DataConfig,
    split: str,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 0,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Wrap a dataset in a dataloader, sharding across ranks when running distributed.

    Training uses ``batch_size_per_device`` (gradient accumulation makes up the difference to the
    effective ``batch_size``); validation uses ``eval_batch_size``.
    """
    is_train = split == "train"
    batch_size = config.batch_size_per_device if is_train else config.eval_batch_size
    sampler: DistributedSampler | None = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=is_train, seed=seed, drop_last=is_train
        )
    kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=(sampler is None and is_train),
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        drop_last=config.drop_last and is_train,
    )
    if config.num_workers > 0:
        kwargs["persistent_workers"] = config.persistent_workers
        if config.prefetch_factor is not None:
            kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(dataset, **kwargs), sampler


def accumulation_steps(config: DataConfig, world_size: int = 1) -> int:
    """Gradient-accumulation steps needed to reach the effective batch size."""
    per_step = config.batch_size_per_device * world_size
    if per_step <= 0:
        raise ValueError("batch_size_per_device and world_size must be positive")
    steps = max(1, round(config.batch_size / per_step))
    if steps * per_step != config.batch_size:
        log.warning(
            "Effective batch size %d is not divisible by %d x %d; using %d accumulation steps (effective %d).",
            config.batch_size,
            config.batch_size_per_device,
            world_size,
            steps,
            steps * per_step,
        )
    return steps
