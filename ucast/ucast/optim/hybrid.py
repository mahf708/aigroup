"""Drive several optimizers as one.

U-Cast optimises hidden weight matrices with Muon and everything else (biases, norm gains, the output
projection) with AdamW.  Wrapping both in a single object keeps the training loop and the LR schedule
oblivious to the split.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

__all__ = ["HybridOptimizer"]


class HybridOptimizer(torch.optim.Optimizer):
    """Composite optimizer whose ``param_groups`` are the concatenation of its children's.

    Because the group dicts are shared (not copied), anything that mutates ``param_groups`` -- an LR
    schedule, for instance -- transparently reaches the underlying optimizers.
    """

    def __init__(self, optimizers: Sequence[torch.optim.Optimizer]):
        optimizers = list(optimizers)
        if not optimizers:
            raise ValueError("HybridOptimizer needs at least one optimizer")
        self.optimizers = optimizers
        self.defaults = dict(optimizers[0].defaults)
        # Deliberately skips Optimizer.__init__: state and param_groups are delegated to the children.

    # ------------------------------------------------------------------ delegation
    @property
    def param_groups(self) -> list[dict]:  # type: ignore[override]
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    @param_groups.setter
    def param_groups(self, value) -> None:  # pragma: no cover - present so torch internals can assign
        raise AttributeError("HybridOptimizer.param_groups is derived from its child optimizers")

    @property
    def state(self) -> dict:  # type: ignore[override]
        merged: dict = {}
        for optimizer in self.optimizers:
            merged.update(optimizer.state)
        return merged

    def add_param_group(self, param_group: dict) -> None:  # pragma: no cover - not meaningful here
        raise NotImplementedError("Add parameter groups to the child optimizers instead")

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def state_dict(self) -> dict:  # type: ignore[override]
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict: dict) -> None:  # type: ignore[override]
        states = state_dict["optimizers"]
        if len(states) != len(self.optimizers):
            raise ValueError(f"checkpoint holds {len(states)} optimizers but {len(self.optimizers)} were built")
        for optimizer, state in zip(self.optimizers, states):
            optimizer.load_state_dict(state)

    def named_children(self) -> Iterable[tuple[str, torch.optim.Optimizer]]:
        for optimizer in self.optimizers:
            yield type(optimizer).__name__, optimizer

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        names = ", ".join(type(o).__name__ for o in self.optimizers)
        return f"HybridOptimizer({names})"
