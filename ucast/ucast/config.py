"""Typed configuration: nested dataclasses, YAML files with inheritance, and CLI overrides.

Everything a run needs lives in one :class:`ExperimentConfig`, which is stored inside every checkpoint
(as JSON) so a checkpoint alone is enough to rebuild the model.

The YAML loader supports a ``base:`` key -- one path or a list of them, relative to the file being
loaded -- so the Stage-2 config only has to state what differs from Stage 1, exactly like the reference
implementation's Hydra composition, but without the Hydra dependency.
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

__all__ = [
    "GridConfig",
    "ModelConfig",
    "DataConfig",
    "OptimizerConfig",
    "MuonConfig",
    "ScheduleConfig",
    "TrainConfig",
    "EvalConfig",
    "ExperimentConfig",
    "load_config",
    "apply_overrides",
    "deep_merge",
    "to_dict",
    "from_dict",
]


# ---------------------------------------------------------------------------------- dataclass <-> dict
def to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses (and containers of them) to plain Python."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _is_optional(annotation) -> tuple[bool, Any]:
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return len(args) < len(typing.get_args(annotation)), args[0] if len(args) == 1 else annotation
    return False, annotation


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    optional, annotation = _is_optional(annotation)
    if value is None:
        if not optional:
            # Permit None for un-annotated optionals rather than fighting the user's YAML.
            return None
        return None
    if is_dataclass(annotation) and isinstance(annotation, type):
        if isinstance(value, annotation):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"{path}: expected a mapping for {annotation.__name__}, got {type(value).__name__}")
        return from_dict(annotation, value, path=path)

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (list, Sequence, typing.Sequence):
        inner = args[0] if args else Any
        return [_coerce(v, inner, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], f"{path}[{i}]") for i, v in enumerate(value))
        return tuple(_coerce(v, a, f"{path}[{i}]") for i, (v, a) in enumerate(zip(value, args)))
    if origin in (dict, Mapping, typing.Mapping):
        val_type = args[1] if len(args) == 2 else Any
        return {str(k): _coerce(v, val_type, f"{path}.{k}") for k, v in value.items()}
    if annotation in (int, float, str, bool):
        if annotation is bool and isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return annotation(value)
    return value


def from_dict(cls: type, data: Mapping[str, Any], path: str = "") -> Any:
    """Instantiate a (possibly nested) dataclass from a mapping, coercing field types."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        prefix = f"{path}: " if path else ""
        raise ValueError(f"{prefix}unknown config keys {sorted(unknown)}; valid keys are {sorted(known)}")
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        sub_path = f"{path}.{f.name}" if path else f.name
        kwargs[f.name] = _coerce(data[f.name], hints[f.name], sub_path)
    return cls(**kwargs)


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; ``update`` wins, nested mappings are merged rather than replaced."""
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``dotted.key=yaml_value`` overrides, e.g. ``train.max_epochs=8``."""
    out = dict(data)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override {override!r} must look like key.subkey=value")
        key, _, raw = override.partition("=")
        value = yaml.safe_load(raw) if raw != "" else None
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value
    return out


# ---------------------------------------------------------------------------------- config schema
@dataclass
class GridConfig:
    """Spatial grid.  ``latitudes``/``longitudes`` override the equiangular defaults when given."""

    num_lat: int = 121
    num_lon: int = 240
    with_poles: bool = True
    periodic_longitude: bool = True
    latitudes: list[float] | None = None
    longitudes: list[float] | None = None


@dataclass
class ModelConfig:
    """U-Net backbone.  The defaults are the paper's 895M-parameter configuration."""

    model_channels: int = 320
    channel_mult: tuple[int, ...] = (1, 2, 3, 4)
    num_blocks: int = 4
    attn_levels: tuple[int, ...] = (-2, -1)
    channels_per_head: int = 64
    num_heads: int | None = None
    dropout: float = 0.1
    latitude_padding: str = "zeros"
    skip_scale: float = 1.0
    pool_ceil_mode: bool = True


@dataclass
class DataConfig:
    """How to build the datasets and dataloaders.

    ``builder`` is either a short registry key (see :mod:`ucast.data`) or a dotted path
    ``"my_package.my_module:MyDataset"``, so plugging in E3SM output means writing one dataset class
    and pointing the config at it -- no changes anywhere else.

    ``common`` is merged into both ``train`` and ``val`` kwargs; the split-specific dicts win.
    """

    builder: str = "synthetic"
    common: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    val: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 48
    batch_size_per_device: int = 8
    eval_batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    drop_last: bool = True
    statistics: str | None = None
    compute_statistics_samples: int | None = None

    def split_kwargs(self, split: str) -> dict[str, Any]:
        specific = {"train": self.train, "val": self.val}[split]
        return deep_merge(self.common, specific)


@dataclass
class MuonConfig:
    """Muon settings.  ``lr=None`` disables Muon and puts every parameter on AdamW."""

    lr: float | None = 3.0e-3
    weight_decay: float = 0.03
    momentum: float = 0.95
    momentum_min_offset: float = 0.1
    ns_steps: int = 5
    exclude_patterns: tuple[str, ...] = ("out_conv", "stem")


@dataclass
class ScheduleConfig:
    name: str = "cosine"
    warmup_steps: int = 1500
    start_lr_ratio: float = 0.0
    min_lr_ratio: float = 0.0


@dataclass
class OptimizerConfig:
    """AdamW settings for the non-Muon parameters, plus the nested Muon and schedule settings."""

    lr: float = 3.0e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1.0e-8
    muon: MuonConfig = field(default_factory=MuonConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)


@dataclass
class TrainConfig:
    """Training loop.

    Args:
        stage: ``"deterministic"`` (Stage 1, weighted MAE, one member) or ``"probabilistic"``
            (Stage 2, weighted CRPS, ``ensemble_members`` members).  It sets the loss and ensemble
            size unless those are given explicitly.
        loss / ensemble_members: Explicit overrides of the stage defaults.
        init_from: Checkpoint to warm-start weights from -- how Stage 2 (and each deep-ensemble
            member) picks up the Stage-1 backbone.
        max_epochs / max_steps: Stop condition; whichever is reached first.
        precision: ``"fp32"``, ``"bf16"`` or ``"fp16"`` autocast for the forward/backward pass.
    """

    stage: str = "deterministic"
    loss: str | None = None
    ensemble_members: int | None = None
    max_epochs: int = 100
    max_steps: int | None = None
    grad_clip_norm: float | None = 1.0
    precision: str = "bf16"
    ema_decay: float = 0.9999
    seed: int = 11
    init_from: str | None = None
    init_from_strict: bool = False
    resume_from: str | None = None
    output_dir: str = "runs/ucast"
    log_every: int = 50
    validate_every_epochs: int = 1
    save_every_epochs: int = 1
    keep_last_checkpoints: int = 2
    monitor: str = "val/avg/crps/avg"
    monitor_mode: str = "min"
    channels_last: bool = False
    compile: bool = False
    deterministic: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None

    def resolved_loss(self) -> str:
        if self.loss is not None:
            return self.loss
        return {"deterministic": "wmae", "probabilistic": "wcrps"}[self.validated_stage()]

    def resolved_ensemble_members(self) -> int:
        if self.ensemble_members is not None:
            return self.ensemble_members
        return {"deterministic": 1, "probabilistic": 2}[self.validated_stage()]

    def validated_stage(self) -> str:
        if self.stage not in ("deterministic", "probabilistic"):
            raise ValueError(f"stage must be 'deterministic' or 'probabilistic', got {self.stage!r}")
        return self.stage


@dataclass
class EvalConfig:
    """Validation / inference rollout.

    Args:
        rollout_steps: Autoregressive steps to take (60 x 6h = the standard 15-day forecast).
        ensemble_size: MC-dropout members per checkpoint at evaluation time.
        members_per_forward: Cap on how many members share one batched forward pass (memory knob).
        lead_times: Which lead times to score, in steps.  ``None`` scores every step.
        max_batches: Cap on validation batches during training (full evaluation on ``ucast score``).
    """

    rollout_steps: int = 4
    ensemble_size: int = 4
    members_per_forward: int | None = None
    lead_times: list[int] | None = None
    max_batches: int | None = None
    use_ema: bool = True
    mc_dropout: bool = True


@dataclass
class ExperimentConfig:
    """The whole run."""

    name: str = "ucast"
    grid: GridConfig = field(default_factory=GridConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    #: Variables the model ingests; flat keys (``"temperature_850"``) or mappings.
    input_variables: list[Any] = field(default_factory=list)
    #: Variables the model predicts.  Defaults to ``input_variables`` when left empty.
    output_variables: list[Any] = field(default_factory=list)
    #: Number of past frames fed to the model (2 in the paper).
    window: int = 2
    #: Hours between consecutive frames; used only to label lead times.
    step_hours: int = 12

    def to_dict(self) -> dict[str, Any]:
        return to_dict(self)

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        return from_dict(cls, data)

    @classmethod
    def from_json(cls, text: str) -> "ExperimentConfig":
        return cls.from_dict(json.loads(text))

    def replace(self, **overrides) -> "ExperimentConfig":
        return dataclasses.replace(self, **overrides)


# ---------------------------------------------------------------------------------- YAML loading
def _load_yaml_with_bases(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path in seen:
        raise ValueError(f"circular config inheritance: {' -> '.join(str(p) for p in (*seen, path))}")
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a mapping at the top level")
    bases = data.pop("base", None)
    if bases is None:
        return data
    if isinstance(bases, str):
        bases = [bases]
    merged: dict[str, Any] = {}
    for base in bases:
        base_path = (path.parent / base).resolve()
        merged = deep_merge(merged, _load_yaml_with_bases(base_path, seen + (path,)))
    return deep_merge(merged, data)


def load_config(path: str | Path | None = None, overrides: Sequence[str] = ()) -> ExperimentConfig:
    """Load a YAML config (resolving ``base:`` inheritance) and apply CLI overrides.

    Args:
        path: YAML file; ``None`` starts from the dataclass defaults.
        overrides: ``dotted.key=value`` strings, applied after inheritance.
    """
    data = _load_yaml_with_bases(Path(path)) if path is not None else {}
    if overrides:
        data = apply_overrides(data, overrides)
    return ExperimentConfig.from_dict(data)
