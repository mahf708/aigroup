"""Inference: MC-dropout ensembles, deep ensembles, scoring and forecast output.

A single checkpoint already gives an ensemble -- run it ``N`` times with different dropout masks.
Stage 3 of the recipe stacks ``K`` independently fine-tuned checkpoints on top of that for ``K x N``
members, which is cheap here because each extra checkpoint only costs one short Stage-2 run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import load_forecaster
from .config import ExperimentConfig
from .data.base import DatasetSpec, ForecastDataset
from .forecaster import UCast
from .metrics import MetricCollection
from .utils import get_logger, resolve_device

__all__ = ["DeepEnsemble", "evaluate", "forecast_to_dataset", "save_forecast"]

log = get_logger(__name__)


class DeepEnsemble:
    """One or more U-Cast checkpoints, sampled jointly into a single ensemble.

    Args:
        models: The member models.  All must share a channel layout and grid.
        members_per_model: MC-dropout members drawn from each model; the total ensemble size is
            ``len(models) * members_per_model``.
        members_per_forward: Memory cap on members per batched forward pass.
        use_mc_dropout: Keep dropout active at inference.  Turning this off with one model and one
            member gives the deterministic forecast.
    """

    def __init__(
        self,
        models: Sequence[UCast],
        members_per_model: int = 1,
        members_per_forward: int | None = None,
        use_mc_dropout: bool = True,
    ):
        if not models:
            raise ValueError("DeepEnsemble needs at least one model")
        specs = {tuple(m.output_variables.keys) for m in models}
        if len(specs) != 1:
            raise ValueError("all ensemble members must predict the same variables in the same order")
        self.models = list(models)
        self.members_per_model = int(members_per_model)
        self.members_per_forward = members_per_forward
        self.use_mc_dropout = use_mc_dropout
        for model in self.models:
            model.eval()

    @classmethod
    def from_checkpoints(
        cls,
        paths: Iterable[str | Path],
        device: str | torch.device | None = None,
        use_ema: bool = True,
        spec: DatasetSpec | None = None,
        **kwargs,
    ) -> tuple["DeepEnsemble", ExperimentConfig]:
        """Load checkpoints into an ensemble.  Returns ``(ensemble, config_of_the_first_checkpoint)``."""
        device = resolve_device(device)
        models: list[UCast] = []
        first_config: ExperimentConfig | None = None
        for path in paths:
            model, config, loaded_spec = load_forecaster(path, map_location=device, use_ema=use_ema, spec=spec)
            models.append(model.to(device))
            if first_config is None:
                first_config, spec = config, loaded_spec
            log.info("Loaded ensemble member from %s", path)
        assert first_config is not None
        return cls(models, **kwargs), first_config

    @property
    def spec(self) -> DatasetSpec:
        return self.models[0].spec

    @property
    def ensemble_size(self) -> int:
        return len(self.models) * self.members_per_model

    def to(self, device) -> "DeepEnsemble":
        for model in self.models:
            model.to(device)
        return self

    @torch.no_grad()
    def rollout(self, batch: Mapping[str, torch.Tensor], steps: int, denormalize: bool = True) -> torch.Tensor:
        """Forecast with every member.  Returns ``(K*N, B, steps, C_out, H, W)``."""
        forecasts = [
            model.rollout(
                batch,
                steps=steps,
                ensemble_size=self.members_per_model,
                members_per_forward=self.members_per_forward,
                use_mc_dropout=self.use_mc_dropout,
                denormalize=denormalize,
            )
            for model in self.models
        ]
        return torch.cat(forecasts, dim=0)


@torch.no_grad()
def evaluate(
    ensemble: DeepEnsemble,
    dataset: ForecastDataset,
    steps: int | None = None,
    lead_times: Sequence[int] | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    max_batches: int | None = None,
    device: str | torch.device | None = None,
    progress: bool = True,
) -> dict[str, float]:
    """Score an ensemble over a dataset, returning WeatherBench-2 style metrics.

    Args:
        ensemble: The models to score.
        dataset: Validation/test dataset; its ``rollout_steps`` bounds how far ahead it can be scored.
        steps: Autoregressive steps; defaults to the dataset's ``rollout_steps``.
        lead_times: Which steps to score (1-based).  ``None`` scores all of them.
        batch_size / num_workers: Dataloader settings.
        max_batches: Stop early (useful for a smoke check).
        device: Device to run on.
        progress: Log progress every 10 batches.

    Returns:
        Flat metric dictionary, keys like ``t24/rmse/z500`` and ``avg/crps/avg``.
    """
    device = resolve_device(device)
    ensemble.to(device)
    spec = ensemble.spec
    steps = min(steps or dataset.rollout_steps, dataset.rollout_steps)
    selected = [s for s in (lead_times or range(1, steps + 1)) if 1 <= s <= steps]
    if not selected:
        raise ValueError(f"no lead times to score (steps={steps}, lead_times={lead_times})")
    metrics = MetricCollection(
        spec.output_variables,
        spec.grid,
        lead_times=[s * spec.step_hours for s in selected],
        device=device,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    output_indices = ensemble.models[0].output_indices

    for index, raw_batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw_batch.items()}
        forecast = ensemble.rollout(batch, steps=steps)
        truth = batch["dynamics"].index_select(2, output_indices)
        for step in selected:
            metrics.update(
                step * spec.step_hours,
                forecast[:, :, step - 1].float(),
                truth[:, spec.window + step - 1].float(),
            )
        if progress and index % 10 == 0:
            log.info("scored %d/%d batches", index + 1, len(loader) if max_batches is None else max_batches)

    scores = dict(metrics.compute())
    scores["ensemble_size"] = float(ensemble.ensemble_size)
    return scores


def forecast_to_dataset(
    forecast: torch.Tensor,
    spec: DatasetSpec,
    initial_times: Sequence[np.datetime64] | np.ndarray,
    ensemble_dim_name: str = "member",
):
    """Wrap a ``(M, B, steps, C, H, W)`` forecast in an :class:`xarray.Dataset`.

    One data variable per forecast variable, with dimensions
    ``(member, time, prediction_timedelta, latitude, longitude)`` -- the WeatherBench-2 layout, so the
    result can be fed straight to their evaluation tooling.
    """
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Writing forecasts needs xarray: pip install -e '.[data]'") from exc

    if forecast.ndim != 6:
        raise ValueError(f"expected (M, B, steps, C, H, W), got {tuple(forecast.shape)}")
    members, batch, steps, channels, height, width = forecast.shape
    if channels != spec.num_output_channels:
        raise ValueError(f"forecast has {channels} channels but the spec declares {spec.num_output_channels}")
    if len(initial_times) != batch:
        raise ValueError(f"got {len(initial_times)} initial times for a batch of {batch}")

    array = forecast.detach().cpu().numpy()
    lead = np.arange(1, steps + 1) * np.timedelta64(spec.step_hours, "h")
    coords = {
        ensemble_dim_name: np.arange(members),
        "time": np.asarray(initial_times),
        "prediction_timedelta": lead,
        "latitude": np.asarray(spec.grid.latitudes),
        "longitude": np.asarray(spec.grid.longitudes),
    }
    dims = (ensemble_dim_name, "time", "prediction_timedelta", "latitude", "longitude")
    data = {var.key: (dims, array[:, :, :, i]) for i, var in enumerate(spec.output_variables)}
    return xr.Dataset(data, coords=coords)


def save_forecast(path: str | Path, forecast: torch.Tensor, spec: DatasetSpec, initial_times, **kwargs) -> Path:
    """Write a forecast to NetCDF (``.nc``) or zarr (``.zarr``)."""
    dataset = forecast_to_dataset(forecast, spec, initial_times, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zarr":
        dataset.to_zarr(path, mode="w")
    else:
        dataset.to_netcdf(path)
    log.info("Wrote forecast to %s", path)
    return path
