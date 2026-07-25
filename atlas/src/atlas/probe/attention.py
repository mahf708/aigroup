"""Reading the backbone's attention as geography.

Because the predictive backbone uses *global* attention over a 91x120 token
grid, each attention row is a map: "when updating the atmosphere here, where
does the model look?".  That is the closest thing the architecture has to an
explicit teleconnection operator, and it is cheap to extract for a handful of
query points even though the full matrix is far too large to store.
"""

from __future__ import annotations

import numpy as np
import torch

from ..backbones import token_coords
from ..spec import GridSpec

__all__ = [
    "nearest_token",
    "teleconnection_map",
    "head_locality",
    "attention_entropy",
    "attention_rollout",
    "great_circle_matrix",
]


def nearest_token(
    grid: GridSpec, grid_hw: tuple[int, int], lat: float, lon: float
) -> int:
    """Flat token index closest to a geographic point."""
    tlat, tlon = token_coords(grid, grid_hw)
    dlat = np.abs(tlat - lat)
    dlon = np.abs((tlon - np.mod(lon, 360.0) + 180.0) % 360.0 - 180.0)
    return int(np.argmin(dlat)) * grid_hw[1] + int(np.argmin(dlon))


def teleconnection_map(
    weights: torch.Tensor,
    grid_hw: tuple[int, int],
    head: int | None = None,
    query: int = 0,
    batch: int = 0,
) -> np.ndarray:
    """One attention row reshaped to a ``(gh, gw)`` map.

    ``weights`` is the ``(B, H, n_query, N)`` tensor produced by
    :meth:`atlas.probe.record.Recorder.capture_attention` with explicit
    ``query_indices``.  ``head=None`` averages over heads.
    """
    w = weights[batch]
    row = w.mean(0)[query] if head is None else w[head, query]
    return row.reshape(grid_hw).cpu().numpy()


def great_circle_matrix(grid: GridSpec, grid_hw: tuple[int, int]) -> torch.Tensor:
    """``(N, N)`` great-circle distances between token centres, in km."""
    tlat, tlon = token_coords(grid, grid_hw)
    la, lo = np.meshgrid(np.deg2rad(tlat), np.deg2rad(tlon), indexing="ij")
    la, lo = la.reshape(-1), lo.reshape(-1)
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)
    cos = np.clip(xyz @ xyz.T, -1.0, 1.0)
    return torch.from_numpy(6371.0 * np.arccos(cos)).float()


def head_locality(
    weights: torch.Tensor, grid: GridSpec, grid_hw: tuple[int, int], query_indices: torch.Tensor
) -> torch.Tensor:
    """Mean attended great-circle distance per head, in km.

    A head whose mass sits within a few hundred kilometres is doing local
    advection/diffusion; a head with a mean attended distance of several
    thousand kilometres is a candidate teleconnection channel.  Sorting heads
    by this number is a fast way to find the small subset worth looking at in
    detail.
    """
    dist = great_circle_matrix(grid, grid_hw).to(weights.device)
    rows = dist[query_indices.to(dist.device)]  # (n_query, N)
    return (weights * rows.unsqueeze(0).unsqueeze(0)).sum(-1).mean(-1)


def attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of each attention row, averaged over queries."""
    w = weights.clamp_min(1e-12)
    return (-(w * w.log()).sum(-1)).mean(-1)


def attention_rollout(
    per_layer: list[torch.Tensor], residual_weight: float = 0.5
) -> torch.Tensor:
    """Compose per-layer attention into an input-to-output influence matrix.

    Requires full ``(B, H, N, N)`` matrices, so this is only tractable on small
    token grids -- build a reduced-resolution configuration when you want it.
    Heads are averaged and a residual connection is mixed in at each layer, per
    the usual rollout construction.
    """
    out = None
    for a in per_layer:
        m = a.mean(1)
        n = m.shape[-1]
        eye = torch.eye(n, device=m.device, dtype=m.dtype).expand_as(m)
        m = residual_weight * eye + (1 - residual_weight) * m
        m = m / m.sum(-1, keepdim=True).clamp_min(1e-12)
        out = m if out is None else torch.bmm(m, out)
    if out is None:
        raise ValueError("no attention matrices given")
    return out
