"""Small shared helpers: logging, seeding, dotted-path imports and distributed bookkeeping."""

from __future__ import annotations

import importlib
import logging
import os
import random
from typing import Any

import numpy as np
import torch

__all__ = [
    "get_logger",
    "setup_logging",
    "seed_everything",
    "import_object",
    "count_parameters",
    "format_count",
    "resolve_device",
    "DistributedInfo",
    "distributed_info",
    "barrier",
]


def setup_logging(level: int | str = logging.INFO, rank: int = 0) -> None:
    """Configure root logging once; non-zero ranks are quietened to warnings."""
    logging.basicConfig(
        level=logging.WARNING if rank != 0 else level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed Python, NumPy and torch (all devices).  Returns the seed for convenient logging."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def import_object(path: str) -> Any:
    """Import ``"package.module:attribute"`` (or ``"package.module.attribute"``)."""
    if ":" in path:
        module_name, _, attr = path.partition(":")
    else:
        module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"{path!r} is not a valid dotted import path")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"{module_name!r} has no attribute {attr!r}") from exc


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def format_count(n: int) -> str:
    """``895_000_000 -> '895.0M'``."""
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= threshold:
            return f"{n / threshold:.1f}{suffix}"
    return str(n)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """``"auto"``/``None`` picks CUDA, then MPS, then CPU."""
    if device is not None and str(device) != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DistributedInfo:
    """Rank/world-size view of the current process, whether or not ``torchrun`` launched it."""

    def __init__(self, rank: int, local_rank: int, world_size: int, enabled: bool):
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.enabled = enabled

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"DistributedInfo(rank={self.rank}/{self.world_size}, local_rank={self.local_rank})"


def distributed_info() -> DistributedInfo:
    """Read rank/world size from the environment (no side effects, no process-group setup)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistributedInfo(rank=rank, local_rank=local_rank, world_size=world_size, enabled=world_size > 1)


def barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
