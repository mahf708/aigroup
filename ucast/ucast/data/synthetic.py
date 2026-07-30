"""A synthetic, dependency-free dataset of advecting wave fields.

Cheap enough to run the full pipeline -- both curriculum stages, rollout, scoring -- on a laptop CPU in
seconds, which is what the test suite and the smoke config use.  The fields are smooth travelling waves
with a genuinely predictable next step, so a short training run really does drive the loss down; that
makes this useful for shaking out a new cluster, a new dataset adapter or a config change before
committing GPU hours.
"""

from __future__ import annotations

import numpy as np
import torch

from ..grid import Grid
from ..variables import Variable, VariableSet
from .base import DatasetSpec, ForecastDataset
from .forcings import clock_forcings

__all__ = ["SyntheticForecastDataset"]

_DEFAULT_VARIABLES: tuple[str, ...] = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "geopotential_500",
    "temperature_850",
)


class SyntheticForecastDataset(ForecastDataset):
    """Deterministic synthetic forecast cases on a lat/lon grid.

    Args:
        num_lat / num_lon: Grid size.
        variables: Input variable keys (defaults to a 4-channel mix of surface and upper-air fields).
        output_variables: Predicted subset; defaults to ``variables``.
        window: Input frames per case.
        rollout_steps: Target frames per case (validation needs as many as it scores).
        num_times: Length of the underlying time series.
        step_hours: Hours between frames.
        seed: Controls the wave field; two datasets with the same seed are bit-identical.
        num_modes: Number of Fourier modes summed per channel.
        with_forcings / with_statics: Include clock forcings / orography+land-sea mask.
        start_time: ISO date of the first frame.
        noise: Standard deviation of unpredictable per-frame noise, as a fraction of the channel's
            own standard deviation.  Non-zero values give the CRPS objective something real to spread
            over, which is what makes a probabilistic smoke test meaningful.
    """

    def __init__(
        self,
        num_lat: int = 32,
        num_lon: int = 64,
        variables: list[str] | tuple[str, ...] | None = None,
        output_variables: list[str] | tuple[str, ...] | None = None,
        window: int = 2,
        rollout_steps: int = 1,
        num_times: int = 256,
        step_hours: int = 12,
        seed: int = 0,
        num_modes: int = 6,
        with_forcings: bool = True,
        with_statics: bool = True,
        start_time: str = "2020-01-01",
        noise: float = 0.02,
    ):
        super().__init__()
        input_vars = VariableSet(variables if variables is not None else _DEFAULT_VARIABLES)
        out_vars = VariableSet(output_variables) if output_variables is not None else input_vars
        grid = Grid.equiangular(num_lat, num_lon)
        self.rollout_steps = int(rollout_steps)
        self.spec = DatasetSpec(
            input_variables=input_vars,
            output_variables=out_vars,
            grid=grid,
            window=int(window),
            num_forcing_channels=4 if with_forcings else 0,
            num_static_channels=2 if with_statics else 0,
            step_hours=int(step_hours),
        )
        if num_times < self.frames_per_item:
            raise ValueError(f"num_times={num_times} is shorter than one item ({self.frames_per_item} frames)")
        self.num_times = int(num_times)
        self.noise = float(noise)

        rng = np.random.default_rng(seed)
        self._times = np.datetime64(start_time, "s") + np.arange(self.num_times) * np.timedelta64(
            int(step_hours) * 3600, "s"
        )
        self._seconds = self._times.astype("datetime64[s]").astype(np.int64)
        self._states = self._make_states(rng, num_modes)
        self._forcings = (
            clock_forcings(self._seconds, grid.num_lat, np.asarray(grid.longitudes)) if with_forcings else None
        )
        self._statics = self._make_statics(rng) if with_statics else None

    # ------------------------------------------------------------------ field synthesis
    def _make_states(self, rng: np.random.Generator, num_modes: int) -> np.ndarray:
        grid = self.spec.grid
        lat = np.deg2rad(np.asarray(grid.latitudes))[None, :, None]
        lon = np.deg2rad(np.asarray(grid.longitudes))[None, None, :]
        steps = np.arange(self.num_times, dtype=np.float64)[:, None, None]

        states = np.zeros((self.num_times, len(self.spec.input_variables), grid.num_lat, grid.num_lon))
        for channel, var in enumerate(self.spec.input_variables):
            field = np.zeros((self.num_times, grid.num_lat, grid.num_lon))
            for mode in range(num_modes):
                zonal = int(rng.integers(1, 6))
                meridional = int(rng.integers(1, 4))
                omega = float(rng.normal(0.0, 0.35))
                phase = float(rng.uniform(0.0, 2 * np.pi))
                amplitude = float(rng.normal(0.0, 1.0)) / (mode + 1)
                field += amplitude * np.cos(zonal * lon + omega * steps + phase) * np.cos(meridional * lat)
            field /= max(field.std(), 1e-12)
            mean, scale = self._physical_scale(var)
            states[:, channel] = mean + scale * field
            if self.noise > 0:
                states[:, channel] += scale * self.noise * rng.standard_normal(field.shape)
        return states.astype(np.float32)

    @staticmethod
    def _physical_scale(var: Variable) -> tuple[float, float]:
        """Plausible mean and standard deviation, so normalisation is exercised realistically."""
        if var.name == "mean_sea_level_pressure":
            return 101_325.0, 1_000.0
        if var.name.endswith("temperature"):
            return 288.0, 15.0
        if var.name == "geopotential":
            return 5_500.0 * 9.81, 3_000.0
        if var.name == "specific_humidity":
            return 5e-3, 2e-3
        if "wind" in var.name or var.name in ("u_component_of_wind", "v_component_of_wind"):
            return 0.0, 8.0
        return 0.0, 1.0

    def _make_statics(self, rng: np.random.Generator) -> np.ndarray:
        grid = self.spec.grid
        lat = np.deg2rad(np.asarray(grid.latitudes))[:, None]
        lon = np.deg2rad(np.asarray(grid.longitudes))[None, :]
        orography = np.zeros((grid.num_lat, grid.num_lon))
        for mode in range(4):
            orography += rng.normal() * np.cos((mode + 1) * lon + rng.uniform(0, 2 * np.pi)) * np.cos((mode + 1) * lat)
        orography /= max(orography.std(), 1e-12)
        land_sea_mask = (orography > 0.2).astype(np.float64)
        statics = np.stack([orography, land_sea_mask])
        # Standardise per channel, as is customary for static conditioning fields.
        statics = (statics - statics.mean(axis=(-2, -1), keepdims=True)) / statics.std(
            axis=(-2, -1), keepdims=True
        ).clip(1e-6)
        return statics.astype(np.float32)

    # ------------------------------------------------------------------ Dataset protocol
    def __len__(self) -> int:
        return self.num_times - self.frames_per_item + 1

    def get_item(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        stop = index + self.frames_per_item
        item = {
            "dynamics": torch.from_numpy(self._states[index:stop].copy()),
            "time": torch.tensor(int(self._seconds[index + self.spec.window - 1]), dtype=torch.int64),
        }
        if self._forcings is not None:
            item["forcings"] = torch.from_numpy(self._forcings[index:stop].copy())
        if self._statics is not None:
            item["statics"] = torch.from_numpy(self._statics.copy())
        return item

    def initial_times(self) -> np.ndarray:
        """Initial-condition timestamps, one per item."""
        return self._times[self.spec.window - 1 : self.spec.window - 1 + len(self)]
