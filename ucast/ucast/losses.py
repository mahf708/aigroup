"""Training objectives: area/variable-weighted MAE, MSE and the fair ensemble CRPS.

Stage 1 of the curriculum minimises the weighted MAE, Stage 2 the weighted CRPS.  MAE is used rather
than MSE precisely because for a single-member forecast the MAE *is* the CRPS, so the two stages
optimise the same functional and Stage 2 starts from a well-aligned loss landscape.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["crps_ensemble", "WeightedLoss", "WeightedMAE", "WeightedMSE", "WeightedCRPS", "build_loss"]


def crps_ensemble(predictions: Tensor, targets: Tensor, ensemble_dim: int = 0, fair: bool = True) -> Tensor:
    """Pointwise ensemble CRPS, keeping every non-ensemble dimension.

    Uses the kernel identity ``CRPS(F, y) = E|X - y| - 1/2 E|X - X'|``, estimated from the ensemble as
    ``skill - spread / 2``.  With ``fair=True`` the spread term is divided by ``M(M - 1)`` instead of
    ``M^2`` (Zamo & Naveau, 2018), which removes the bias that otherwise makes small training
    ensembles systematically under-dispersive -- essential here, since U-Cast trains with ``M = 2``.

    Args:
        predictions: Ensemble forecasts with an ensemble axis at ``ensemble_dim``.
        targets: Ground truth, shaped like ``predictions`` with the ensemble axis removed.
        ensemble_dim: Position of the ensemble axis in ``predictions``.
        fair: Use the unbiased (fair) estimator.

    Returns:
        CRPS with the same shape as ``targets``.
    """
    preds = predictions.movedim(ensemble_dim, 0)
    if preds.shape[1:] != targets.shape:
        raise ValueError(f"target shape {tuple(targets.shape)} does not match members {tuple(preds.shape[1:])}")
    num_members = preds.shape[0]
    skill = (preds - targets).abs().mean(dim=0)
    if num_members == 1:
        return skill  # a single member reduces exactly to the absolute error
    # Accumulate the |x_m - x_n| pairs one at a time: O(M^2) work but O(1) extra memory, which matters
    # because these tensors are (B, C, H, W) at full model width.
    spread = torch.zeros_like(skill)
    for m in range(num_members - 1):
        spread = spread + (preds[m + 1 :] - preds[m]).abs().sum(dim=0)
    # spread now holds sum_{m<n} |x_m - x_n|; the CRPS spread term is half of the mean over ordered
    # pairs, i.e. sum_{m<n} / denominator.
    denominator = num_members * (num_members - 1) if fair else num_members**2
    return skill - spread / denominator


class WeightedLoss(nn.Module):
    """Base class holding broadcastable per-channel / per-latitude weights.

    Args:
        weights: Tensor broadcastable against ``(..., C, H, W)``, typically ``(C, H, 1)`` from
            :meth:`ucast.grid.Grid.channel_area_weights`.  ``None`` means unweighted.
    """

    def __init__(self, weights: Tensor | None = None):
        super().__init__()
        self.register_buffer("weights", None if weights is None else torch.as_tensor(weights).clone(), persistent=False)

    def reduce(self, pointwise: Tensor) -> Tensor:
        if self.weights is not None:
            pointwise = pointwise * self.weights.to(dtype=pointwise.dtype, device=pointwise.device)
        return pointwise.mean()

    def pointwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.reduce(self.pointwise(predictions, targets))


class WeightedMAE(WeightedLoss):
    """Weighted mean absolute error; the Stage-1 (deterministic pre-training) objective."""

    needs_ensemble = False

    def pointwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return (predictions - targets).abs()


class WeightedMSE(WeightedLoss):
    """Weighted mean squared error, provided for comparison/ablation."""

    needs_ensemble = False

    def pointwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return (predictions - targets) ** 2


class WeightedCRPS(WeightedLoss):
    """Weighted fair ensemble CRPS; the Stage-2 (probabilistic fine-tuning) objective.

    ``predictions`` must carry a leading ensemble axis, i.e. ``(M, B, C, H, W)``.
    """

    needs_ensemble = True

    def __init__(self, weights: Tensor | None = None, fair: bool = True):
        super().__init__(weights)
        self.fair = fair

    def pointwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return crps_ensemble(predictions, targets, ensemble_dim=0, fair=self.fair)


_ALIASES: dict[str, type[WeightedLoss]] = {
    "mae": WeightedMAE,
    "wmae": WeightedMAE,
    "weighted_mae": WeightedMAE,
    "l1": WeightedMAE,
    "mse": WeightedMSE,
    "wmse": WeightedMSE,
    "weighted_mse": WeightedMSE,
    "l2": WeightedMSE,
    "crps": WeightedCRPS,
    "wcrps": WeightedCRPS,
    "weighted_crps": WeightedCRPS,
}


def build_loss(name: str, weights: Tensor | None = None, **kwargs) -> WeightedLoss:
    """Look up a loss by name (``"wmae"``, ``"wcrps"``, ``"wmse"``, ...)."""
    key = name.strip().lower().replace("-", "_")
    if key not in _ALIASES:
        raise ValueError(f"Unknown loss {name!r}; available: {sorted(set(_ALIASES))}")
    return _ALIASES[key](weights=weights, **kwargs)
