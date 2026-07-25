"""Declarative description of the data an ATLAS model consumes and produces.

Everything downstream (normalisation, latent shape, patching, probes) is driven
by the objects in this module, so retargeting the model from ERA5 to an
E3SM-style dataset is a configuration change rather than a code change.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "Channel",
    "VariableSet",
    "GridSpec",
    "LatentConfig",
    "BackboneConfig",
    "ProjectorConfig",
    "EstimatorConfig",
    "ForcingConfig",
    "AtlasConfig",
]


# ---------------------------------------------------------------------------
# channels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Channel:
    """One prognostic 2-D field.

    Parameters
    ----------
    name
        Unique identifier, e.g. ``"z500"``, ``"T_lev30"``, ``"PRECT"``.
    group
        Family the channel belongs to, e.g. ``"z"``, ``"T"``, ``"surface"``.
        Probes and loss weights are frequently expressed per group.
    level
        Vertical coordinate value.  ``None`` for surface/single-level fields.
    level_kind
        ``"pressure"`` (hPa), ``"hybrid"`` (E3SM/CAM hybrid sigma-pressure
        index), ``"model"``, ``"height"`` or ``"surface"``.
    units, long_name
        Free-form metadata carried through to any xarray output.
    weight
        Per-channel multiplier used by the training losses.  Defaults to 1.
    positive
        If True the field is physically non-negative (precipitation, specific
        humidity, ...).  Used by optional output clamping.
    """

    name: str
    group: str = "misc"
    level: float | None = None
    level_kind: str = "surface"
    units: str | None = None
    long_name: str | None = None
    weight: float = 1.0
    positive: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _pressure_channels(
    prefix: str, levels: Sequence[float], group: str | None = None, **kw: Any
) -> list[Channel]:
    group = group or prefix
    return [
        Channel(f"{prefix}{int(p)}", group=group, level=float(p), level_kind="pressure", **kw)
        for p in levels
    ]


@dataclass
class VariableSet:
    """Ordered collection of :class:`Channel` objects.

    The order defines the channel axis of every tensor the model sees.
    """

    channels: list[Channel]

    def __post_init__(self) -> None:
        names = [c.name for c in self.channels]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate channel names: {dupes}")

    def __len__(self) -> int:
        return len(self.channels)

    def __iter__(self):
        return iter(self.channels)

    def __getitem__(self, key: int | str) -> Channel:
        if isinstance(key, str):
            return self.channels[self.index(key)]
        return self.channels[key]

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.channels]

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:  # pragma: no cover - trivial
            raise KeyError(f"no channel named {name!r}") from exc

    def indices(self, names: Iterable[str]) -> np.ndarray:
        return np.array([self.index(n) for n in names], dtype=np.int64)

    def group_indices(self, group: str) -> np.ndarray:
        return np.array(
            [i for i, c in enumerate(self.channels) if c.group == group], dtype=np.int64
        )

    @property
    def groups(self) -> list[str]:
        seen: list[str] = []
        for c in self.channels:
            if c.group not in seen:
                seen.append(c.group)
        return seen

    def weights(self) -> np.ndarray:
        """Per-channel loss weights, normalised to mean 1."""
        w = np.array([c.weight for c in self.channels], dtype=np.float64)
        if w.sum() <= 0:
            raise ValueError("channel weights must be positive")
        return (w / w.mean()).astype(np.float32)

    def subset(self, names: Iterable[str]) -> VariableSet:
        keep = list(names)
        return VariableSet([self[n] for n in keep])

    def to_dict(self) -> dict[str, Any]:
        """Serialise, collapsing to a preset name when one matches exactly.

        Writing 75 fully-specified channels into every config makes them
        unreadable, so the standard sets round-trip as ``{"preset": "..."}``.
        Custom sets are always written out in full.
        """
        for name, factory in _PRESETS.items():
            try:
                preset = factory()
            except Exception:  # pragma: no cover - defensive
                continue
            if preset.names == self.names:
                return {"preset": name}
            if set(self.names) <= set(preset.names) and all(
                preset[c.name] == c for c in self.channels
            ):
                return {"preset": name, "subset": self.names}
        return {"channels": [c.to_dict() for c in self.channels]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VariableSet:
        if "preset" in d:
            preset = d["preset"]
            if preset not in _PRESETS:
                raise KeyError(
                    f"unknown variable preset {preset!r}; known: {sorted(_PRESETS)}"
                )
            vs = _PRESETS[preset]()
            if "subset" in d:
                vs = vs.subset(d["subset"])
            return vs
        return cls([Channel(**c) for c in d["channels"]])

    # -- ready-made sets ---------------------------------------------------

    @classmethod
    def era5_atlas(cls) -> VariableSet:
        """The 75-channel ERA5 set used by ATLAS.

        Channel order matches the released NVIDIA checkpoint's variable list:
        eight surface fields, then ``u``/``v``/``z``/``t``/``q`` on thirteen
        pressure levels, then SST and total precipitation.  (Table 1 of the
        paper lists seven surface fields; surface pressure is the eighth,
        which is what makes the total 75.)
        """
        levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        surface = [
            Channel("u10m", "surface", units="m s-1", long_name="10 metre U wind"),
            Channel("v10m", "surface", units="m s-1", long_name="10 metre V wind"),
            Channel("u100m", "surface", units="m s-1", long_name="100 metre U wind"),
            Channel("v100m", "surface", units="m s-1", long_name="100 metre V wind"),
            Channel("t2m", "surface", units="K", long_name="2 metre temperature"),
            Channel("sp", "surface", units="Pa", long_name="Surface pressure"),
            Channel("msl", "surface", units="Pa", long_name="Mean sea level pressure"),
            Channel("tcwv", "surface", units="kg m-2", long_name="Total column water vapour"),
        ]
        atmos: list[Channel] = []
        for prefix, unit in (
            ("u", "m s-1"),
            ("v", "m s-1"),
            ("z", "m2 s-2"),
            ("t", "K"),
            ("q", "kg kg-1"),
        ):
            atmos += _pressure_channels(
                prefix, levels, units=unit, positive=(prefix == "q")
            )
        extra = [
            Channel("sst", "surface", units="K", long_name="Sea surface temperature"),
            Channel("tp", "surface", units="m", long_name="Total precipitation", positive=True),
        ]
        return cls(surface + atmos + extra)

    @classmethod
    def e3sm_eam(
        cls,
        n_levels: int = 24,
        prognostic: Sequence[str] = ("T", "U", "V", "Q", "Z3"),
        surface: Sequence[str] = ("PS", "TS", "PRECT", "TMQ", "U10", "FLUT"),
    ) -> VariableSet:
        """A generic E3SM/EAM channel set on hybrid sigma-pressure levels.

        ``n_levels`` is the number of *retained* model levels; EAMv2/v3 run on
        72 or 80 levels, but AI emulators usually train on a coarsened subset.
        Levels are indexed from the top of the atmosphere downwards, matching
        CAM's ``lev`` ordering.
        """
        units = {
            "T": "K",
            "U": "m s-1",
            "V": "m s-1",
            "Q": "kg kg-1",
            "Z3": "m",
            "CLDLIQ": "kg kg-1",
            "CLDICE": "kg kg-1",
            "OMEGA": "Pa s-1",
            "PS": "Pa",
            "TS": "K",
            "PRECT": "m s-1",
            "TMQ": "kg m-2",
            "U10": "m s-1",
            "FLUT": "W m-2",
        }
        pos = {"Q", "CLDLIQ", "CLDICE", "PRECT", "TMQ", "U10"}
        chans: list[Channel] = [
            Channel(v, "surface", units=units.get(v), positive=v in pos) for v in surface
        ]
        for v in prognostic:
            for k in range(n_levels):
                chans.append(
                    Channel(
                        f"{v}_lev{k:02d}",
                        group=v,
                        level=float(k),
                        level_kind="hybrid",
                        units=units.get(v),
                        positive=v in pos,
                    )
                )
        return cls(chans)


#: Named channel sets that round-trip through configs as ``{"preset": name}``.
_PRESETS: dict[str, Any] = {
    "era5_atlas": lambda: VariableSet.era5_atlas(),
    "e3sm_eam_24lev": lambda: VariableSet.e3sm_eam(n_levels=24),
    "e3sm_eam_72lev": lambda: VariableSet.e3sm_eam(n_levels=72),
}


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------


@dataclass
class GridSpec:
    """A structured latitude-longitude grid.

    ATLAS tokenises a rectangular field, so the native grid must be
    structured.  Unstructured E3SM output (``ne30pg2`` and friends) should be
    regridded first -- see ``docs/atlas.md`` for the recommended recipe.

    Attributes
    ----------
    lat, lon
        1-D coordinate arrays in degrees.  ``lat`` may be ascending
        (E3SM/CAM convention) or descending (ERA5 convention).
    periodic_lon
        Whether longitude wraps.  False for limited-area / regional domains,
        which switches circular padding to replication.
    poles
        Whether the grid includes both poles.  Enables pole-consistent
        padding in the local-attention projector.
    """

    lat: np.ndarray
    lon: np.ndarray
    periodic_lon: bool = True
    poles: bool = True
    name: str = "latlon"

    def __post_init__(self) -> None:
        self.lat = np.asarray(self.lat, dtype=np.float64)
        self.lon = np.asarray(self.lon, dtype=np.float64)
        if self.lat.ndim != 1 or self.lon.ndim != 1:
            raise ValueError("lat and lon must be 1-D")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.lat.size), int(self.lon.size)

    @property
    def nlat(self) -> int:
        return int(self.lat.size)

    @property
    def nlon(self) -> int:
        return int(self.lon.size)

    @property
    def descending_lat(self) -> bool:
        return bool(self.lat[0] > self.lat[-1])

    @classmethod
    def equiangular(
        cls, nlat: int, nlon: int, descending: bool = True, include_poles: bool = True
    ) -> GridSpec:
        """Regular lat-lon grid.

        With ``include_poles`` the latitudes span [-90, 90] inclusive (ERA5's
        721x1440); otherwise cell centres are used (CAM's 180x360).
        """
        if include_poles:
            lat = np.linspace(-90.0, 90.0, nlat)
        else:
            edges = np.linspace(-90.0, 90.0, nlat + 1)
            lat = 0.5 * (edges[:-1] + edges[1:])
        if descending:
            lat = lat[::-1].copy()
        lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
        return cls(lat=lat, lon=lon, poles=include_poles)

    def area_weights(self) -> np.ndarray:
        """Normalised (sum == 1) cell areas of shape ``(nlat, 1)``.

        Uses the exact spherical-cap difference ``sin(lat_u) - sin(lat_l)``
        as in the paper's metric definitions, so it is correct for both
        pole-inclusive and cell-centre grids.
        """
        lat = np.sort(self.lat.astype(np.float64))
        edges = np.empty(lat.size + 1)
        edges[1:-1] = 0.5 * (lat[:-1] + lat[1:])
        edges[0] = max(-90.0, lat[0] - 0.5 * (lat[1] - lat[0]))
        edges[-1] = min(90.0, lat[-1] + 0.5 * (lat[-1] - lat[-2]))
        w = np.abs(np.diff(np.sin(np.deg2rad(edges))))
        if self.descending_lat:
            w = w[::-1]
        w = w / w.sum()
        return w.reshape(-1, 1).astype(np.float32)

    def coarsen(self, factor: int | tuple[int, int]) -> GridSpec:
        """Grid obtained by reducing resolution by ``factor``.

        Latitude keeps the endpoints (so 721 -> 181 for factor 4, exactly the
        paper's 0.25 deg -> 1 deg choice); longitude stays equispaced.
        """
        fy, fx = (factor, factor) if isinstance(factor, int) else factor
        nlat, nlon = self.shape
        new_nlat = (nlat - 1) // fy + 1 if self.poles else nlat // fy
        new_nlon = nlon // fx
        lat = np.linspace(self.lat[0], self.lat[-1], new_nlat)
        lon = np.linspace(self.lon[0], self.lon[0] + 360.0, new_nlon, endpoint=False)
        return replace(self, lat=lat, lon=lon, name=f"{self.name}/coarse")

    def _is_regular(self) -> bool:
        """True when the grid is reproducible from :meth:`equiangular`."""
        nlat, nlon = self.shape
        if nlat < 2 or nlon < 2:
            return False
        ref = GridSpec.equiangular(
            nlat, nlon, descending=self.descending_lat, include_poles=self.poles
        )
        return bool(
            np.allclose(self.lat, ref.lat, atol=1e-6)
            and np.allclose(self.lon, ref.lon, atol=1e-6)
        )

    def to_dict(self) -> dict[str, Any]:
        """Compact form for regular grids, explicit coordinates otherwise."""
        base = {"periodic_lon": self.periodic_lon, "poles": self.poles, "name": self.name}
        if self._is_regular():
            return {
                "nlat": self.nlat,
                "nlon": self.nlon,
                "descending": self.descending_lat,
                **base,
            }
        return {"lat": self.lat.tolist(), "lon": self.lon.tolist(), **base}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GridSpec:
        if "nlat" in d:  # compact form
            g = cls.equiangular(
                d["nlat"],
                d["nlon"],
                descending=d.get("descending", True),
                include_poles=d.get("poles", True),
            )
            g.periodic_lon = d.get("periodic_lon", True)
            g.name = d.get("name", "latlon")
            return g
        return cls(
            lat=np.asarray(d["lat"]),
            lon=np.asarray(d["lon"]),
            periodic_lon=d.get("periodic_lon", True),
            poles=d.get("poles", True),
            name=d.get("name", "latlon"),
        )


# ---------------------------------------------------------------------------
# model configuration
# ---------------------------------------------------------------------------


@dataclass
class LatentConfig:
    """How the full-resolution state is reduced to the modelling space.

    ``mode`` selects the encoder:

    ``bilinear``
        The paper's choice: pure bilinear interpolation.  No parameters, and
        latent channel *c* stays physically identical to state channel *c*.
    ``area``
        Area-weighted average pooling.  Conserves the global mean, which
        matters for E3SM-style budget diagnostics.
    ``spectral``
        Spherical-harmonic truncation (requires ``torch-harmonics``).  Makes
        the latent an explicit set of retained modes.
    ``learned``
        A trainable convolutional/DiT encoder, for reproducing the paper's
        Section 5 ablation against a VAE-style latent.
    """

    shape: tuple[int, int] = (181, 360)
    mode: str = "bilinear"
    residual: bool = True
    history: int = 1
    learned_encoder_dim: int = 256
    learned_encoder_depth: int = 4

    def __post_init__(self) -> None:
        self.shape = tuple(int(s) for s in self.shape)  # type: ignore[assignment]
        if self.mode not in {"bilinear", "area", "spectral", "learned"}:
            raise ValueError(f"unknown latent mode {self.mode!r}")
        if self.history < 0:
            raise ValueError("history must be >= 0")


@dataclass
class BackboneConfig:
    """The global-attention DiT that models the latent conditional law."""

    embed_dim_state: int = 768
    embed_dim_history: int = 256
    combine: str = "concat"  # concat | add | mul
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    patch: tuple[int, int] = (2, 3)
    attention: str = "global"
    pos_embed: str = "sincos2d"  # sincos2d | latlon | learned | none
    qk_norm: bool = False
    grad_checkpoint_every: int | None = None
    noise_dim: int | None = None  # FiLM noise channel, used by the CRPS head

    def __post_init__(self) -> None:
        self.patch = tuple(int(p) for p in self.patch)  # type: ignore[assignment]

    @property
    def embed_dim(self) -> int:
        if self.combine == "concat":
            return self.embed_dim_state + self.embed_dim_history
        return self.embed_dim_state


@dataclass
class ProjectorConfig:
    """The history-conditioned local decoder ``D(r, x0) ~= x1 - x0``."""

    embed_dim_state: int = 512
    embed_dim_latent: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    attention: str = "neighborhood"  # neighborhood | window | global
    kernel: tuple[int, int] = (3, 3)
    pos_embed: str = "sincos2d"
    qk_norm: bool = False
    grad_checkpoint_every: int | None = None
    neighborhood_chunk: int = 4096

    def __post_init__(self) -> None:
        self.kernel = tuple(int(k) for k in self.kernel)  # type: ignore[assignment]

    @property
    def embed_dim(self) -> int:
        return self.embed_dim_state + self.embed_dim_latent


@dataclass
class EstimatorConfig:
    """Which probabilistic estimator sits on top of the shared backbone."""

    kind: str = "si"  # si | edm | crps
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForcingConfig:
    """Extra deterministic input channels appended to the projector input."""

    cos_zenith: bool = True
    time_of_year: bool = False
    time_of_day: bool = False
    static_fields: list[str] = field(default_factory=list)
    scalar_fields: list[str] = field(default_factory=list)

    def n_channels(self, grid_free: bool = False) -> int:
        n = int(self.cos_zenith) + len(self.static_fields) + len(self.scalar_fields)
        n += 2 * int(self.time_of_year) + 2 * int(self.time_of_day)
        return n


@dataclass
class AtlasConfig:
    """Everything needed to build an :class:`atlas.model.Atlas`."""

    variables: VariableSet
    grid: GridSpec
    latent: LatentConfig = field(default_factory=LatentConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    forcings: ForcingConfig = field(default_factory=ForcingConfig)
    dt_hours: float = 6.0
    name: str = "atlas"

    @property
    def n_channels(self) -> int:
        return len(self.variables)

    @property
    def latent_grid(self) -> GridSpec:
        """Grid the latent lives on, inheriting the parent grid's topology."""
        nlat, nlon = self.latent.shape
        lat = np.linspace(self.grid.lat[0], self.grid.lat[-1], nlat)
        lon = np.linspace(self.grid.lon[0], self.grid.lon[0] + 360.0, nlon, endpoint=False)
        if not self.grid.periodic_lon:
            lon = np.linspace(self.grid.lon[0], self.grid.lon[-1], nlon)
        return GridSpec(
            lat=lat,
            lon=lon,
            periodic_lon=self.grid.periodic_lon,
            poles=self.grid.poles,
            name=f"{self.grid.name}/latent",
        )

    @property
    def compression(self) -> float:
        hi = math.prod(self.grid.shape)
        lo = math.prod(self.latent.shape)
        return hi / lo

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dt_hours": self.dt_hours,
            "variables": self.variables.to_dict(),
            "grid": self.grid.to_dict(),
            "latent": asdict(self.latent),
            "backbone": asdict(self.backbone),
            "projector": asdict(self.projector),
            "estimator": asdict(self.estimator),
            "forcings": asdict(self.forcings),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AtlasConfig:
        return cls(
            variables=VariableSet.from_dict(d["variables"]),
            grid=GridSpec.from_dict(d["grid"]),
            latent=LatentConfig(**d.get("latent", {})),
            backbone=BackboneConfig(**d.get("backbone", {})),
            projector=ProjectorConfig(**d.get("projector", {})),
            estimator=EstimatorConfig(**d.get("estimator", {})),
            forcings=ForcingConfig(**d.get("forcings", {})),
            dt_hours=d.get("dt_hours", 6.0),
            name=d.get("name", "atlas"),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        text = _dump(self.to_dict(), path.suffix)
        path.write_text(text)

    @classmethod
    def load(cls, path: str | Path) -> AtlasConfig:
        path = Path(path)
        return cls.from_dict(_load(path.read_text(), path.suffix))


def _dump(d: dict[str, Any], suffix: str) -> str:
    if suffix in {".yaml", ".yml"}:
        import yaml  # noqa: PLC0415

        return yaml.safe_dump(d, sort_keys=False)
    return json.dumps(d, indent=2)


def _load(text: str, suffix: str) -> dict[str, Any]:
    if suffix in {".yaml", ".yml"}:
        import yaml  # noqa: PLC0415

        return yaml.safe_load(text)
    return json.loads(text)
