"""The dataset contract.

A dataset yields dictionaries of tensors; nothing downstream knows or cares where the data came from.
Implement :class:`ForecastDataset` for your own archive (E3SM history files, a regional subset, a
different reanalysis) and the model, training loop, metrics and CLI work unchanged.

Each item describes one forecast case:

==============  =========================  =========================================================
key             shape                      meaning
==============  =========================  =========================================================
``dynamics``    ``(T, C_in, H, W)``        state in physical units, ``T = window + rollout_steps``
``forcings``    ``(T, C_forcing, H, W)``   optional time-varying conditioning (clock, solar, ...)
``statics``     ``(C_static, H, W)``       optional time-invariant conditioning (orography, LSM)
``time``        scalar int64               seconds since epoch of the last input frame
==============  =========================  =========================================================

Frames ``[0, window)`` are inputs; frame ``window + k`` is the target for lead time ``k + 1``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from ..grid import Grid
from ..variables import VariableSet

__all__ = ["DatasetSpec", "ForecastDataset"]


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the model needs to know about a dataset to size itself.

    Args:
        input_variables: Channel order of ``dynamics``.
        output_variables: Channels the model predicts.  Usually equal to ``input_variables``; a subset
            is how you keep a prescribed field (SST, say) as an input without forecasting it.
        grid: The spatial grid.
        window: Number of input frames per forecast.
        num_forcing_channels / num_static_channels: Sizes of the optional conditioning tensors.
        step_hours: Hours between consecutive frames, used to label lead times.
    """

    input_variables: VariableSet
    output_variables: VariableSet
    grid: Grid
    window: int = 2
    num_forcing_channels: int = 0
    num_static_channels: int = 0
    step_hours: int = 12

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        missing = [v.key for v in self.output_variables if v.key not in self.input_variables]
        if missing:
            # Allowed, but the model then predicts these channels' absolute values rather than
            # increments, since there is no previous frame to add them to.
            pass

    @property
    def num_input_channels(self) -> int:
        """Channels the network sees: the flattened state window plus all conditioning."""
        return len(self.input_variables) * self.window + self.num_forcing_channels + self.num_static_channels

    @property
    def num_output_channels(self) -> int:
        return len(self.output_variables)

    @property
    def output_indices(self) -> list[int]:
        """Position of each output variable inside ``input_variables`` (``-1`` when absent)."""
        return self.output_variables.indices_in(self.input_variables, missing_ok=True)

    def replace(self, **kwargs) -> "DatasetSpec":
        import dataclasses

        return dataclasses.replace(self, **kwargs)

    def to_dict(self) -> dict:
        """Round-trippable description, stored inside checkpoints."""
        return {
            "input_variables": self.input_variables.to_list(),
            "output_variables": self.output_variables.to_list(),
            "grid": self.grid.to_dict(),
            "window": self.window,
            "num_forcing_channels": self.num_forcing_channels,
            "num_static_channels": self.num_static_channels,
            "step_hours": self.step_hours,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetSpec":
        grid = data["grid"]
        return cls(
            input_variables=VariableSet(data["input_variables"]),
            output_variables=VariableSet(data["output_variables"]),
            grid=Grid.from_arrays(
                grid["latitudes"], grid["longitudes"], periodic_longitude=grid.get("periodic_longitude", True)
            ),
            window=int(data["window"]),
            num_forcing_channels=int(data["num_forcing_channels"]),
            num_static_channels=int(data["num_static_channels"]),
            step_hours=int(data.get("step_hours", 12)),
        )


class ForecastDataset(Dataset, ABC):
    """Base class for U-Cast datasets.

    Subclasses set :attr:`spec` and :attr:`rollout_steps`, then implement :meth:`__len__` and
    :meth:`get_item`.  :meth:`__getitem__` validates shapes so mistakes surface at the dataset rather
    than deep inside a training step.
    """

    spec: DatasetSpec
    rollout_steps: int = 1

    @property
    def frames_per_item(self) -> int:
        return self.spec.window + self.rollout_steps

    @property
    def grid(self) -> Grid:
        return self.spec.grid

    @abstractmethod
    def __len__(self) -> int:  # pragma: no cover - abstract
        ...

    @abstractmethod
    def get_item(self, index: int) -> dict[str, torch.Tensor]:  # pragma: no cover - abstract
        """Return one item; see the module docstring for the expected keys and shapes."""

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.get_item(index)
        self.validate_item(item)
        return item

    def validate_item(self, item: dict[str, torch.Tensor]) -> None:
        spec = self.spec
        height, width = spec.grid.shape
        dynamics = item.get("dynamics")
        if dynamics is None:
            raise KeyError("dataset items must contain a 'dynamics' tensor")
        expected = (self.frames_per_item, len(spec.input_variables), height, width)
        if tuple(dynamics.shape) != expected:
            raise ValueError(f"'dynamics' should have shape {expected}, got {tuple(dynamics.shape)}")

        forcings = item.get("forcings")
        if spec.num_forcing_channels:
            if forcings is None:
                raise KeyError(f"spec declares {spec.num_forcing_channels} forcing channels but none were returned")
            expected_f = (self.frames_per_item, spec.num_forcing_channels, height, width)
            if tuple(forcings.shape) != expected_f:
                raise ValueError(f"'forcings' should have shape {expected_f}, got {tuple(forcings.shape)}")
        elif forcings is not None:
            raise ValueError("'forcings' was returned but the spec declares no forcing channels")

        statics = item.get("statics")
        if spec.num_static_channels:
            if statics is None:
                raise KeyError(f"spec declares {spec.num_static_channels} static channels but none were returned")
            expected_s = (spec.num_static_channels, height, width)
            if tuple(statics.shape) != expected_s:
                raise ValueError(f"'statics' should have shape {expected_s}, got {tuple(statics.shape)}")
        elif statics is not None:
            raise ValueError("'statics' was returned but the spec declares no static channels")

    def describe(self) -> str:
        spec = self.spec
        return (
            f"{type(self).__name__}: {len(self)} items, grid {spec.grid.shape}, "
            f"{len(spec.input_variables)} input / {spec.num_output_channels} output variables, "
            f"window={spec.window}, rollout_steps={self.rollout_steps}, "
            f"forcings={spec.num_forcing_channels}, statics={spec.num_static_channels}"
        )
