"""Checkpoint format.

A checkpoint is self-describing: it carries the experiment config (as JSON), the dataset spec and the
normalisation statistics alongside the weights, so :func:`load_forecaster` can rebuild a working model
from the file alone -- no config file, no dataset access, no guessing which variables the channels were.

That is what makes Stage 2, deep ensembling and standalone inference simple: they all just load a file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .config import ExperimentConfig
from .data.base import DatasetSpec
from .normalization import Normalizer
from .utils import get_logger

__all__ = ["save_checkpoint", "load_checkpoint", "load_forecaster", "load_weights_into"]

log = get_logger(__name__)

FORMAT_VERSION = 1


def _unwrap(module: nn.Module) -> nn.Module:
    """Strip DDP / ``torch.compile`` wrappers so state dict keys stay stable across launch modes."""
    for attribute in ("module", "_orig_mod"):
        inner = getattr(module, attribute, None)
        if isinstance(inner, nn.Module):
            return _unwrap(inner)
    return module


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: ExperimentConfig,
    spec: DatasetSpec,
    normalizer: Normalizer,
    ema=None,
    optimizer=None,
    schedule=None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: Mapping[str, float] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a checkpoint.  Optimizer/EMA state is included when given, so runs resume exactly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "config_json": config.to_json(),
        "spec_json": json.dumps(spec.to_dict()),
        "model": _unwrap(model).state_dict(),
        "normalizer": {
            "keys": spec.input_variables.keys,
            "mean": normalizer.mean.detach().flatten().cpu(),
            "std": normalizer.std.detach().flatten().cpu(),
            "residual_std": normalizer.residual_std.detach().flatten().cpu(),
            "has_residual_std": normalizer.has_residual_std,
        },
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": dict(metrics or {}),
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if schedule is not None:
        payload["schedule"] = schedule.state_dict()
    if extra:
        payload["extra"] = dict(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Read a checkpoint written by :func:`save_checkpoint`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        log.warning("Checkpoint %s has format_version=%s (expected %s); loading anyway.", path, version, FORMAT_VERSION)
    return payload


def _normalizer_from_payload(payload: Mapping[str, Any], spec: DatasetSpec) -> Normalizer:
    stored = payload.get("normalizer")
    if stored is None:
        raise KeyError("checkpoint holds no normalisation statistics")
    return Normalizer(
        spec.input_variables,
        mean=stored["mean"],
        std=stored["std"],
        residual_std=stored["residual_std"] if stored.get("has_residual_std", True) else None,
    )


def load_forecaster(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    use_ema: bool = True,
    spec: DatasetSpec | None = None,
):
    """Rebuild a :class:`~ucast.forecaster.UCast` from a checkpoint.

    Args:
        path: Checkpoint file.
        map_location: Where to load tensors.
        use_ema: Load the EMA weights (what the reported skill refers to) instead of the raw ones.
        spec: Override the stored dataset spec -- e.g. to forecast on a different grid than the one
            trained on, which works as long as the channel layout is unchanged.

    Returns:
        ``(model, config, spec)``.
    """
    from .forecaster import build_forecaster  # imported here to avoid a circular import at module load

    payload = load_checkpoint(path, map_location=map_location)
    config = ExperimentConfig.from_json(payload["config_json"])
    stored_spec = DatasetSpec.from_dict(json.loads(payload["spec_json"]))
    spec = spec or stored_spec
    normalizer = _normalizer_from_payload(payload, stored_spec)
    if spec is not stored_spec:
        normalizer = normalizer.subset(spec.input_variables)
    model = build_forecaster(config, spec, normalizer)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if missing or unexpected:
        log.warning("Loaded %s with missing=%s unexpected=%s", Path(path).name, list(missing)[:5], list(unexpected)[:5])
    if use_ema and "ema" in payload:
        from .ema import ExponentialMovingAverage

        ema = ExponentialMovingAverage(model, decay=1.0)
        ema.load_state_dict(payload["ema"], strict=False)
        ema.copy_to(model)
        log.info("Applied EMA weights from %s (%d updates).", Path(path).name, ema.num_updates)
    elif use_ema:
        log.warning("%s holds no EMA weights; using the raw weights.", Path(path).name)
    return model, config, spec


def load_weights_into(
    model: nn.Module,
    path: str | Path,
    strict: bool = False,
    use_ema: bool = True,
    map_location: str | torch.device = "cpu",
) -> None:
    """Warm-start ``model`` from a checkpoint's weights, ignoring everything else.

    This is the Stage 1 -> Stage 2 handoff (and how each deep-ensemble member starts from the same
    deterministic backbone).  ``strict=False`` lets the architecture differ slightly, e.g. a changed
    dropout rate or a re-sized conditioning stack, while still transferring everything that matches.
    """
    payload = load_checkpoint(path, map_location=map_location)
    state = dict(payload["model"])
    if use_ema and "ema" in payload:
        shadow = payload["ema"].get("shadow", {})
        for name, tensor in shadow.items():
            if name in state:
                state[name] = tensor.to(state[name].dtype)
        log.info("Warm-starting from the EMA weights in %s.", Path(path).name)
    target = _unwrap(model)
    missing, unexpected = target.load_state_dict(state, strict=strict)
    kept = len(state) - len(unexpected)
    log.info(
        "Warm-started %d/%d tensors from %s (missing=%d, unexpected=%d).",
        kept,
        len(target.state_dict()),
        Path(path).name,
        len(missing),
        len(unexpected),
    )
