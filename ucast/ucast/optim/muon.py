"""Muon: momentum orthogonalised by Newton-Schulz.

Adapted from Keller Jordan's reference implementation (https://kellerjordan.github.io/posts/muon/,
MIT licensed).  U-Cast finds Muon both converges faster and reaches a better optimum than AdamW here,
most visibly during the short CRPS fine-tuning stage, which is what makes an 8-epoch Stage 2 enough.

Muon only applies to >=2D hidden weights.  Biases, norm gains and the output projection stay on AdamW;
:func:`ucast.optim.build.build_optimizer` does that split and wires both into a
:class:`~ucast.optim.hybrid.HybridOptimizer`.
"""

from __future__ import annotations

import torch

__all__ = ["Muon", "muon_momentum_schedule", "orthogonalize_via_newton_schulz"]


def orthogonalize_via_newton_schulz(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonal polar factor of ``matrix`` with a quintic Newton-Schulz iteration.

    The coefficients maximise the slope at zero rather than converging exactly to the singular-value-1
    limit; the result behaves like ``U S' V^T`` with ``S'`` close to (but not exactly) the identity,
    which empirically costs nothing and keeps the iteration stable in bfloat16.
    """
    if matrix.ndim < 2:
        raise ValueError(f"expected at least a 2-D tensor, got shape {tuple(matrix.shape)}")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.bfloat16()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)  # spectral norm <= 1
    for _ in range(steps):
        a_mat = x @ x.mT
        x = a * x + (b * a_mat + c * a_mat @ a_mat) @ x
    return x.mT if transposed else x


def _muon_update(grad: torch.Tensor, momentum_buffer: torch.Tensor, beta: float, ns_steps: int, nesterov: bool):
    momentum_buffer.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum_buffer, beta) if nesterov else momentum_buffer.clone()
    if update.ndim > 2:  # convolution kernels are flattened to (out, in*kh*kw)
        update = update.reshape(update.shape[0], -1)
    update = orthogonalize_via_newton_schulz(update, steps=ns_steps)
    # Match the RMS of an Adam-style update so a single lr works across differently shaped layers.
    return update * max(1.0, update.shape[-2] / update.shape[-1]) ** 0.5


class Muon(torch.optim.Optimizer):
    """Muon optimizer for >=2D hidden weights.

    Args:
        params: Parameters to optimise; every one must have ``ndim >= 2``.
        lr: Learning rate, in units of spectral norm per update (much larger than an AdamW lr; the
            paper uses 3e-3 for Stage 1 and 7e-3 for Stage 2).
        weight_decay: Decoupled (AdamW-style) weight decay.
        momentum: Momentum coefficient; see :func:`muon_momentum_schedule` for the usual warmup.
        nesterov: Use the Nesterov-style lookahead on the momentum buffer.
        ns_steps: Newton-Schulz iterations per step.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim < 2:
                    raise ValueError(
                        f"Muon requires parameters with ndim >= 2, got {p.ndim}. "
                        "Route biases and norm gains to AdamW instead."
                    )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                update = _muon_update(
                    p.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    ns_steps=group["ns_steps"],
                    nesterov=group["nesterov"],
                )
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape).to(p.dtype), alpha=-group["lr"])
        return loss


def muon_momentum_schedule(
    step: int,
    total_steps: int,
    warmup_steps: int = 300,
    cooldown_steps: int = 50,
    momentum_min: float = 0.85,
    momentum_max: float = 0.95,
) -> float:
    """Ramp momentum up at the start of training and back down at the end."""
    warmup_steps = max(1, warmup_steps)
    cooldown_steps = max(1, cooldown_steps)
    cooldown_start = max(warmup_steps, total_steps - cooldown_steps)
    if step < warmup_steps:
        return momentum_min + (step / warmup_steps) * (momentum_max - momentum_min)
    if step > cooldown_start:
        fraction = min(1.0, (step - cooldown_start) / cooldown_steps)
        return momentum_max - fraction * (momentum_max - momentum_min)
    return momentum_max
