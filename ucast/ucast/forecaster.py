"""The U-Cast forecaster: backbone + normalisation + residual bookkeeping + rollout.

This is the object the training loop and inference both talk to.  It owns the conventions that make the
backbone a *forecaster* rather than a generic image-to-image network:

* the input is a flattened window of past states plus conditioning (clock forcings, statics),
* the output is the normalised **increment** from the last input frame, scaled by ``std/residual_std``
  so all channels are order 1,
* an ensemble is produced by running the same input several times with different MC-dropout masks --
  batched into a single forward pass, so ``M`` members cost one wide forward, not ``M`` narrow ones,
* rolling out autoregressively feeds predictions back in, optionally taking prescribed (non-predicted)
  channels from ground truth.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping

import torch
from torch import Tensor, nn

from .config import ExperimentConfig, ModelConfig
from .data.base import DatasetSpec
from .grid import Grid
from .losses import WeightedLoss
from .nn.unet import UCastUNet
from .normalization import Normalizer
from .utils import get_logger
from .variables import VariableSet

__all__ = ["UCast", "mc_dropout", "build_forecaster"]

log = get_logger(__name__)


@contextmanager
def mc_dropout(module: nn.Module, enabled: bool = True) -> Iterator[None]:
    """Put every dropout layer in training mode while leaving the rest of the model in eval mode.

    This is what turns a deterministic network into an ensemble generator at inference time.  Norm
    layers keep their eval behaviour, so only the dropout masks vary between members.
    """
    if not enabled:
        yield
        return
    dropouts = [m for m in module.modules() if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d))]
    previous = [m.training for m in dropouts]
    for m in dropouts:
        m.train(True)
    try:
        yield
    finally:
        for m, was_training in zip(dropouts, previous):
            m.train(was_training)


class UCast(nn.Module):
    """A probabilistic single-step forecaster that can be rolled out to arbitrary lead times.

    Args:
        net: The backbone; must map ``(B, spec.num_input_channels, H, W)`` to
            ``(B, spec.num_output_channels, H, W)``.
        normalizer: Statistics for ``spec.input_variables``.
        spec: Dataset description (variables, grid, window, conditioning sizes).
        predict_residual: Predict the increment from the last input frame (the paper's setup) rather
            than the next state outright.
        prescribed_policy: For input variables the model does not predict, how to advance them during a
            rollout -- ``"prescribed"`` takes them from the ground-truth frames when the batch provides
            them (right for AMIP-style prescribed SST), ``"persist"`` holds the last known value.
    """

    def __init__(
        self,
        net: nn.Module,
        normalizer: Normalizer,
        spec: DatasetSpec,
        predict_residual: bool = True,
        prescribed_policy: str = "prescribed",
    ):
        super().__init__()
        if prescribed_policy not in ("prescribed", "persist"):
            raise ValueError(f"prescribed_policy must be 'prescribed' or 'persist', got {prescribed_policy!r}")
        missing = [v.key for v in spec.output_variables if v.key not in spec.input_variables]
        if missing:
            raise ValueError(
                "output_variables must be a subset of input_variables (the model forecasts the state it "
                f"is given); these are missing from the inputs: {missing[:8]}"
            )
        if normalizer.variables != spec.input_variables:
            normalizer = normalizer.subset(spec.input_variables)

        self.net = net
        self.normalizer = normalizer
        self.spec = spec
        self.predict_residual = predict_residual
        self.prescribed_policy = prescribed_policy

        output_indices = spec.output_indices  # position of each output channel within the input state
        self.register_buffer("output_indices", torch.as_tensor(output_indices, dtype=torch.long), persistent=False)
        prescribed = sorted(set(range(len(spec.input_variables))) - set(output_indices))
        self.register_buffer("prescribed_indices", torch.as_tensor(prescribed, dtype=torch.long), persistent=False)

    # ------------------------------------------------------------------ shorthands
    @property
    def grid(self) -> Grid:
        return self.spec.grid

    @property
    def window(self) -> int:
        return self.spec.window

    @property
    def input_variables(self) -> VariableSet:
        return self.spec.input_variables

    @property
    def output_variables(self) -> VariableSet:
        return self.spec.output_variables

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    # ------------------------------------------------------------------ tensor plumbing
    def compose_input(self, window_norm: Tensor, forcings: Tensor | None, statics: Tensor | None) -> Tensor:
        """Flatten a normalised state window and concatenate the conditioning channels.

        Args:
            window_norm: ``(B, window, C_in, H, W)`` normalised states.
            forcings: ``(B, C_forcing, H, W)`` forcings **valid at the target time**.
            statics: ``(B, C_static, H, W)`` time-invariant fields.
        """
        if window_norm.ndim != 5:
            raise ValueError(f"expected (B, window, C, H, W), got {tuple(window_norm.shape)}")
        if window_norm.shape[1] != self.window:
            raise ValueError(f"expected {self.window} input frames, got {window_norm.shape[1]}")
        parts = [window_norm.flatten(1, 2)]
        if self.spec.num_forcing_channels:
            if forcings is None:
                raise ValueError("this model was configured with forcings, but none were provided")
            parts.append(forcings)
        elif forcings is not None:
            raise ValueError("forcings were provided but the model has no forcing channels")
        if self.spec.num_static_channels:
            if statics is None:
                raise ValueError("this model was configured with statics, but none were provided")
            parts.append(statics)
        return torch.cat(parts, dim=1)

    def forward(self, window_norm: Tensor, forcings: Tensor | None = None, statics: Tensor | None = None) -> Tensor:
        """Raw network output: the residual-scaled increment for the output channels."""
        return self.net(self.compose_input(window_norm, forcings, statics))

    def normalize_states(self, states: Tensor) -> Tensor:
        """Physical units -> normalised, for a ``(..., C_in, H, W)`` tensor."""
        return self.normalizer.normalize(states)

    def denormalize_predictions(self, predictions_norm: Tensor) -> Tensor:
        """Normalised -> physical units, for a ``(..., C_out, H, W)`` tensor."""
        return self.normalizer.denormalize(predictions_norm, channels=self.output_indices.tolist())

    def target_from_states(self, window_norm: Tensor, target_norm: Tensor) -> Tensor:
        """Training target in the network's own output space.

        With ``predict_residual`` this is ``(x_{t+1} - x_t)`` in normalised units, rescaled by
        ``std/residual_std``; otherwise it is simply the normalised next state.
        """
        channels = self.output_indices
        target = target_norm.index_select(-3, channels)
        if not self.predict_residual:
            return target
        previous = window_norm[:, -1].index_select(-3, channels)
        return self.normalizer.to_residual_scale(target - previous, channels=channels.tolist())

    def prediction_from_output(self, output: Tensor, window_norm: Tensor) -> Tensor:
        """Network output -> normalised next state for the output channels."""
        if not self.predict_residual:
            return output
        increment = self.normalizer.from_residual_scale(output, channels=self.output_indices.tolist())
        previous = window_norm[:, -1].index_select(-3, self.output_indices)
        return previous + increment

    # ------------------------------------------------------------------ ensembles
    @staticmethod
    def _tile(tensor: Tensor | None, members: int) -> Tensor | None:
        if tensor is None or members == 1:
            return tensor
        return tensor.unsqueeze(0).expand(members, *tensor.shape).reshape(-1, *tensor.shape[1:])

    def ensemble_forward(
        self,
        window_norm: Tensor,
        forcings: Tensor | None = None,
        statics: Tensor | None = None,
        members: int = 1,
    ) -> Tensor:
        """Run ``members`` MC-dropout members in one batched pass; returns ``(M, B, C_out, H, W)``.

        The batch is tiled ``M`` times, so each copy draws its own dropout masks.  This is why CRPS
        training with ``M = 2`` costs only twice a deterministic step -- and why the paper's Stage 2 is
        affordable at all.
        """
        if members < 1:
            raise ValueError(f"members must be >= 1, got {members}")
        batch = window_norm.shape[0]
        output = self.forward(
            self._tile(window_norm, members), self._tile(forcings, members), self._tile(statics, members)
        )
        return output.reshape(members, batch, *output.shape[1:])

    # ------------------------------------------------------------------ losses
    def compute_loss(
        self,
        batch: Mapping[str, Tensor],
        loss_fn: WeightedLoss,
        members: int = 1,
    ) -> tuple[Tensor, dict[str, float]]:
        """Single-step training loss for one batch.

        Args:
            batch: Dataset batch with ``dynamics`` ``(B, T, C_in, H, W)`` and optional
                ``forcings``/``statics``.
            loss_fn: A :class:`~ucast.losses.WeightedLoss`; CRPS variants need ``members >= 2``.
            members: Ensemble members per sample.

        Returns:
            ``(loss, extra_metrics)``.
        """
        needs_ensemble = getattr(loss_fn, "needs_ensemble", False)
        if needs_ensemble and members < 2:
            raise ValueError(f"{type(loss_fn).__name__} needs at least 2 ensemble members, got {members}")

        states_norm = self.normalize_states(batch["dynamics"])
        window_norm = states_norm[:, : self.window]
        target_norm = states_norm[:, self.window]
        forcings = self._forcings_at(batch, self.window)
        statics = batch.get("statics")
        target = self.target_from_states(window_norm, target_norm)

        if needs_ensemble or members > 1:
            predictions = self.ensemble_forward(window_norm, forcings, statics, members=members)
            loss = loss_fn(predictions if needs_ensemble else predictions.mean(dim=0), target)
        else:
            predictions = self.forward(window_norm, forcings, statics)
            loss = loss_fn(predictions, target)
        return loss, {"members": float(members)}

    def _forcings_at(self, batch: Mapping[str, Tensor], frame: int) -> Tensor | None:
        """Forcings valid at frame ``frame`` (the frame being predicted), or None."""
        forcings = batch.get("forcings")
        if forcings is None:
            return None
        if forcings.ndim != 5:
            raise ValueError(f"'forcings' should be (B, T, C, H, W), got {tuple(forcings.shape)}")
        index = min(frame, forcings.shape[1] - 1)
        return forcings[:, index]

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def rollout(
        self,
        batch: Mapping[str, Tensor],
        steps: int,
        ensemble_size: int = 1,
        members_per_forward: int | None = None,
        use_mc_dropout: bool = True,
        denormalize: bool = True,
    ) -> Tensor:
        """Autoregressive forecast.

        Args:
            batch: Batch whose ``dynamics`` holds at least ``window`` input frames.  Frames beyond the
                window are used only for prescribed (non-predicted) channels.
            steps: Number of autoregressive steps.
            ensemble_size: MC-dropout members.
            members_per_forward: Cap on members per batched forward pass; lets a large ensemble run
                within a fixed memory budget.  ``None`` runs them all at once.
            use_mc_dropout: Enable dropout during the rollout (turn off for a deterministic forecast).
            denormalize: Return physical units instead of normalised values.

        Returns:
            ``(ensemble_size, B, steps, C_out, H, W)``.
        """
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        chunk = ensemble_size if members_per_forward is None else min(members_per_forward, ensemble_size)
        was_training = self.training
        self.eval()
        try:
            with mc_dropout(self, enabled=use_mc_dropout):
                chunks = []
                remaining = ensemble_size
                while remaining > 0:
                    members = min(chunk, remaining)
                    chunks.append(self._rollout_chunk(batch, steps, members, denormalize=denormalize))
                    remaining -= members
            return torch.cat(chunks, dim=0)
        finally:
            self.train(was_training)

    def _rollout_chunk(self, batch: Mapping[str, Tensor], steps: int, members: int, denormalize: bool) -> Tensor:
        states = batch["dynamics"]
        statics = batch.get("statics")
        forcings = batch.get("forcings")
        states_norm = self.normalize_states(states)

        window = self._tile(states_norm[:, : self.window].contiguous(), members)
        statics_tiled = self._tile(statics, members)
        batch_size = states.shape[0]
        outputs: list[Tensor] = []

        for step in range(steps):
            frame = self.window + step
            forcing_frame = None
            if forcings is not None:
                index = min(frame, forcings.shape[1] - 1)
                forcing_frame = self._tile(forcings[:, index], members)
            output = self.forward(window, forcing_frame, statics_tiled)
            prediction_norm = self.prediction_from_output(output, window)
            outputs.append(prediction_norm)

            if step + 1 < steps:
                next_state = self._advance_state(window[:, -1], prediction_norm, states_norm, frame, members)
                window = torch.cat([window[:, 1:], next_state.unsqueeze(1)], dim=1)

        stacked = torch.stack(outputs, dim=1)  # (M*B, steps, C_out, H, W)
        stacked = stacked.reshape(members, batch_size, *stacked.shape[1:])
        return self.denormalize_predictions(stacked) if denormalize else stacked

    def _advance_state(
        self,
        previous: Tensor,
        prediction_norm: Tensor,
        truth_norm: Tensor,
        frame: int,
        members: int,
    ) -> Tensor:
        """Build the next full input state from the prediction plus any prescribed channels."""
        next_state = previous.clone()  # (M*B, C_in, H, W)
        next_state.index_copy_(1, self.output_indices, prediction_norm)
        if self.prescribed_indices.numel() == 0:
            return next_state
        if self.prescribed_policy == "prescribed" and frame < truth_norm.shape[1]:
            prescribed = truth_norm[:, frame].index_select(1, self.prescribed_indices)
            next_state.index_copy_(1, self.prescribed_indices, self._tile(prescribed, members))
        # "persist" (and the fall-back when truth runs out) keeps the values already copied from
        # `previous`, i.e. holds the prescribed channels fixed.
        return next_state

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"window={self.window}, in={self.spec.num_input_channels}, out={self.spec.num_output_channels}, "
            f"grid={self.grid.shape}, predict_residual={self.predict_residual}"
        )


def build_backbone(config: ModelConfig, spec: DatasetSpec) -> UCastUNet:
    """Instantiate the U-Net sized for ``spec``."""
    return UCastUNet(
        in_channels=spec.num_input_channels,
        out_channels=spec.num_output_channels,
        spatial_shape=spec.grid.shape,
        model_channels=config.model_channels,
        channel_mult=tuple(config.channel_mult),
        num_blocks=config.num_blocks,
        attn_levels=tuple(config.attn_levels),
        channels_per_head=config.channels_per_head,
        num_heads=config.num_heads,
        dropout=config.dropout,
        periodic_longitude=spec.grid.periodic_longitude,
        latitude_padding=config.latitude_padding,
        skip_scale=config.skip_scale,
        pool_ceil_mode=config.pool_ceil_mode,
    )


def build_forecaster(config: ExperimentConfig, spec: DatasetSpec, normalizer: Normalizer) -> UCast:
    """Build the full forecaster from an experiment config, a dataset spec and statistics."""
    net = build_backbone(config.model, spec)
    model = UCast(net=net, normalizer=normalizer, spec=spec)
    log.info(
        "U-Cast backbone: %d input / %d output channels on %s, %.1fM parameters",
        spec.num_input_channels,
        spec.num_output_channels,
        spec.grid.shape,
        model.num_parameters / 1e6,
    )
    return model
