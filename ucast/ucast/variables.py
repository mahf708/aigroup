"""Variable bookkeeping: what the model ingests, what it predicts, and how channels are weighted.

A forecast state is a channel-stacked tensor ``(..., C, H, W)``.  :class:`VariableSet` is the single
source of truth for the meaning and ordering of ``C``, so datasets, normalizers, losses and metrics
never have to agree on a convention out of band.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

# Standard ERA5/WeatherBench-2 pressure levels (hPa).
WB2_LEVELS: tuple[int, ...] = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

# The 37 levels GraphCast normalizes its pressure weighting by.  Keeping the divisor tied to the full
# set (not to the levels actually used) means a run that drops levels keeps comparable loss scaling.
_GRAPHCAST_ALL_LEVELS: tuple[int, ...] = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000,
)  # fmt: skip
LEVEL_WEIGHT_DIVISOR: float = float(np.mean(_GRAPHCAST_ALL_LEVELS))

#: Loss weights for surface variables, following GraphCast / GenCast.
SURFACE_WEIGHTS: dict[str, float] = {
    "2m_temperature": 1.0,
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "mean_sea_level_pressure": 0.1,
    "sea_surface_temperature": 0.1,
    "total_precipitation": 0.1,
    "total_precipitation_6hr": 0.1,
    "total_precipitation_12hr": 0.1,
}


@dataclass(frozen=True)
class Variable:
    """One channel of the forecast state.

    Args:
        name: Variable name without a level suffix, e.g. ``"temperature"`` or ``"2m_temperature"``.
        level: Pressure level in hPa for atmospheric variables, or ``None`` for surface/single-level.
        weight: Explicit loss weight.  When ``None`` the weight is derived (see :meth:`loss_weight`).
    """

    name: str
    level: int | None = None
    weight: float | None = None

    @property
    def key(self) -> str:
        """Flat identifier, e.g. ``"temperature_850"``.  Matches WeatherBench-2 naming conventions."""
        return self.name if self.level is None else f"{self.name}_{self.level}"

    @property
    def is_surface(self) -> bool:
        return self.level is None

    @classmethod
    def parse(cls, spec: "str | Mapping[str, object] | Variable") -> "Variable":
        """Build a variable from a flat key (``"temperature_850"``) or a mapping."""
        if isinstance(spec, Variable):
            return spec
        if isinstance(spec, Mapping):
            level = spec.get("level")
            return cls(
                name=str(spec["name"]),
                level=None if level is None else int(level),  # type: ignore[arg-type]
                weight=None if spec.get("weight") is None else float(spec["weight"]),  # type: ignore[arg-type]
            )
        if not isinstance(spec, str):
            raise TypeError(f"Cannot parse variable from {spec!r} of type {type(spec).__name__}")
        head, _, tail = spec.rpartition("_")
        if head and tail.isdigit():
            return cls(name=head, level=int(tail))
        return cls(name=spec)

    def loss_weight(self, level_divisor: float = LEVEL_WEIGHT_DIVISOR) -> float:
        """Per-channel loss weight: explicit override, else pressure- or surface-based default."""
        if self.weight is not None:
            return self.weight
        if self.level is not None:
            return self.level / level_divisor
        return SURFACE_WEIGHTS.get(self.name, 1.0)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.key


class VariableSet(Sequence[Variable]):
    """An ordered, duplicate-free set of :class:`Variable` defining a channel axis."""

    def __init__(self, variables: Iterable["str | Mapping[str, object] | Variable"]):
        parsed: list[Variable] = []
        for spec in variables:
            # ``{name: temperature, levels: [...]}`` expands to one channel per level, which keeps
            # configs for the full 83-channel ERA5 state readable.
            if isinstance(spec, Mapping) and spec.get("levels") is not None:
                weight = spec.get("weight")
                parsed.extend(
                    Variable(
                        name=str(spec["name"]),
                        level=int(level),
                        weight=None if weight is None else float(weight),  # type: ignore[arg-type]
                    )
                    for level in spec["levels"]  # type: ignore[union-attr]
                )
            else:
                parsed.append(Variable.parse(spec))
        seen: dict[str, int] = {}
        for i, var in enumerate(parsed):
            if var.key in seen:
                raise ValueError(f"Duplicate variable {var.key!r} at positions {seen[var.key]} and {i}")
            seen[var.key] = i
        self._variables: tuple[Variable, ...] = tuple(parsed)
        self._index: dict[str, int] = seen

    # ------------------------------------------------------------------ Sequence protocol
    def __len__(self) -> int:
        return len(self._variables)

    def __getitem__(self, item):  # type: ignore[override]
        if isinstance(item, slice):
            return VariableSet(self._variables[item])
        if isinstance(item, str):
            return self._variables[self._index[item]]
        return self._variables[item]

    def __iter__(self) -> Iterator[Variable]:
        return iter(self._variables)

    def __contains__(self, item) -> bool:  # type: ignore[override]
        if isinstance(item, str):
            return item in self._index
        if isinstance(item, Variable):
            return item.key in self._index
        return False

    def __eq__(self, other) -> bool:
        return isinstance(other, VariableSet) and self._variables == other._variables

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"VariableSet({len(self)} channels: {', '.join(self.keys[:4])}{', ...' if len(self) > 4 else ''})"

    # ------------------------------------------------------------------ Accessors
    @property
    def keys(self) -> list[str]:
        return [v.key for v in self._variables]

    def index(self, key: "str | Variable") -> int:  # type: ignore[override]
        """Channel position of ``key``."""
        return self._index[key.key if isinstance(key, Variable) else key]

    def indices_in(self, other: "VariableSet", missing_ok: bool = False) -> list[int]:
        """Positions of this set's variables inside ``other`` (``-1`` for absent ones if allowed)."""
        out = []
        for var in self._variables:
            if var.key in other._index:
                out.append(other._index[var.key])
            elif missing_ok:
                out.append(-1)
            else:
                raise KeyError(f"{var.key!r} is not part of {other!r}")
        return out

    def subset(self, keys: Iterable[str]) -> "VariableSet":
        return VariableSet([self[k] for k in keys])

    def union(self, other: "VariableSet") -> "VariableSet":
        """This set followed by the variables of ``other`` that it does not already contain."""
        return VariableSet(list(self._variables) + [v for v in other if v.key not in self._index])

    def difference(self, other: "VariableSet") -> "VariableSet":
        return VariableSet([v for v in self._variables if v.key not in other._index])

    def with_weights(self, weights: Mapping[str, float]) -> "VariableSet":
        """Copy with explicit loss weights applied to the named variables."""
        unknown = set(weights) - set(self._index)
        if unknown:
            raise KeyError(f"Cannot weight unknown variables: {sorted(unknown)}")
        return VariableSet([replace(v, weight=weights.get(v.key, v.weight)) for v in self._variables])

    # ------------------------------------------------------------------ Derived quantities
    def loss_weights(self, level_divisor: float = LEVEL_WEIGHT_DIVISOR) -> np.ndarray:
        """Per-channel loss weights, shape ``(C,)``."""
        return np.asarray([v.loss_weight(level_divisor) for v in self._variables], dtype=np.float64)

    def levels(self) -> list[int]:
        """Sorted unique pressure levels present in the set."""
        return sorted({v.level for v in self._variables if v.level is not None})

    def surface_names(self) -> list[str]:
        return [v.name for v in self._variables if v.is_surface]

    def atmospheric_names(self) -> list[str]:
        seen: dict[str, None] = {}
        for v in self._variables:
            if v.level is not None:
                seen.setdefault(v.name, None)
        return list(seen)

    def to_list(self) -> list[dict[str, object]]:
        """Round-trippable plain-Python description (for checkpoints / configs)."""
        return [{"name": v.name, "level": v.level, "weight": v.weight} for v in self._variables]


def expand_levels(names: Iterable[str], levels: Iterable[int]) -> list[Variable]:
    """Cross-product of atmospheric ``names`` with pressure ``levels``."""
    levels = list(levels)
    return [Variable(name=n, level=lev) for n in names for lev in levels]


def era5_variable_set(
    surface: Sequence[str] = (
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "sea_surface_temperature",
    ),
    atmospheric: Sequence[str] = (
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
    ),
    levels: Sequence[int] = WB2_LEVELS,
) -> VariableSet:
    """The U-Cast paper's 83-channel ERA5 state (5 surface + 6 atmospheric x 13 levels)."""
    return VariableSet([Variable(name=n) for n in surface] + expand_levels(atmospheric, levels))
