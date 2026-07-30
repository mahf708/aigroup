"""Assemble the AdamW + Muon optimizer pair and its LR schedule from a config."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from ..config import OptimizerConfig
from ..utils import get_logger
from .hybrid import HybridOptimizer
from .muon import Muon
from .schedules import build_schedule

__all__ = ["split_parameters", "build_optimizer", "set_muon_momentum"]

log = get_logger(__name__)


def _is_no_decay(name: str, param: nn.Parameter) -> bool:
    """Biases, norm gains and any other 1-D parameter are excluded from weight decay."""
    return param.ndim < 2 or name.endswith(".bias")


def split_parameters(
    model: nn.Module,
    muon_enabled: bool,
    exclude_patterns: Iterable[str] = (),
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    """Split trainable parameters into ``(muon, adamw_decay, adamw_no_decay)``.

    Muon takes the hidden weight matrices only.  The stem, the output projection and every 1-D
    parameter stay on AdamW, as recommended for Muon: those layers face the data distribution directly
    and do not behave like the interior of the network.
    """
    exclude = tuple(exclude_patterns)
    muon_params: list[nn.Parameter] = []
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_no_decay(name, param):
            no_decay.append(param)
        elif muon_enabled and not any(pattern in name for pattern in exclude):
            muon_params.append(param)
        else:
            decay.append(param)
    return muon_params, decay, no_decay


def build_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
    total_steps: int,
) -> tuple[HybridOptimizer, object]:
    """Build the optimizer and per-step LR schedule for ``model``.

    Returns:
        ``(optimizer, schedule)``.  Call ``schedule.step()`` once per optimizer step.
    """
    muon_enabled = config.muon.lr is not None
    muon_params, decay, no_decay = split_parameters(model, muon_enabled, config.muon.exclude_patterns)

    adamw_groups = []
    if decay:
        adamw_groups.append({"params": decay, "lr": config.lr, "weight_decay": config.weight_decay})
    if no_decay:
        adamw_groups.append({"params": no_decay, "lr": config.lr, "weight_decay": 0.0})
    if not adamw_groups:
        raise ValueError("no parameters were routed to AdamW; check the Muon exclude patterns")
    adamw = torch.optim.AdamW(adamw_groups, lr=config.lr, betas=tuple(config.betas), eps=config.eps)

    optimizers: list[torch.optim.Optimizer] = [adamw]
    if muon_params:
        optimizers.append(
            Muon(
                muon_params,
                lr=config.muon.lr,
                weight_decay=config.muon.weight_decay,
                momentum=config.muon.momentum,
                ns_steps=config.muon.ns_steps,
            )
        )
    elif muon_enabled:
        log.warning("Muon is enabled but no parameters matched it; falling back to AdamW only.")

    optimizer = HybridOptimizer(optimizers)
    schedule = build_schedule(
        config.schedule.name,
        optimizer,
        total_steps=total_steps,
        warmup_steps=config.schedule.warmup_steps,
        start_lr_ratio=config.schedule.start_lr_ratio,
        min_lr_ratio=config.schedule.min_lr_ratio,
    )
    counts = (
        sum(p.numel() for p in muon_params),
        sum(p.numel() for p in decay),
        sum(p.numel() for p in no_decay),
    )
    log.info("Optimizer parameters: muon=%d, adamw(decay)=%d, adamw(no-decay)=%d", *counts)
    return optimizer, schedule


def set_muon_momentum(optimizer: HybridOptimizer, momentum: float) -> None:
    """Update the momentum of every Muon group (the schedule ramps it during training)."""
    for child in optimizer.optimizers:
        if isinstance(child, Muon):
            for group in child.param_groups:
                group["momentum"] = momentum
