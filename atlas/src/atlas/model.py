"""The ATLAS model: encode, predict in latent space, project back.

Two states are carried through a rollout and they are deliberately *not* the
same object:

* a coarse **latent state** ``z``, which the probabilistic backbone advances;
* the full-resolution **fine state** ``x``, which conditions the projector.

Re-encoding the decoded fine state at every step would feed the projector's
own super-resolution artefacts back into the dynamics.  Instead the latent is
advanced in latent space (``z_{j+1} = z_j + r_{j+1}``) while the fine state is
advanced by the projected residual, exactly as in the reference inference
loop.  That split is also why the model stays stable over 60 autoregressive
steps without autoregressive fine-tuning.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .backbones import LatentDiT, LocalProjector
from .estimators import Estimator, build_estimator
from .forcings import ForcingProvider
from .grids import Resampler, SphericalTransform
from .losses import Weights, l1_loss
from .normalize import NormalizerPair
from .spec import AtlasConfig

__all__ = ["Atlas", "StepOutput", "RolloutState"]


@dataclass
class StepOutput:
    """Everything one 6-hour step produced, including the latent internals."""

    state: torch.Tensor
    latent_state: torch.Tensor
    latent_residual: torch.Tensor
    fine_residual: torch.Tensor
    trajectory: torch.Tensor | None = None
    valid_time: np.ndarray | None = None


@dataclass
class RolloutState:
    """Mutable carrier threaded through :meth:`Atlas.rollout`."""

    fine: torch.Tensor
    latents: list[torch.Tensor] = field(default_factory=list)
    time: np.ndarray | None = None

    def history_tensor(self, n: int) -> torch.Tensor | None:
        """Oldest-to-newest history *excluding* the current latent."""
        if n == 0:
            return None
        past = self.latents[:-1][-n:]
        while len(past) < n:
            past = [past[0] if past else self.latents[-1]] + past
        return torch.cat(past, dim=1)


class Atlas(nn.Module):
    """Latent probabilistic forecaster.

    Parameters
    ----------
    cfg
        Full model description; see :class:`atlas.spec.AtlasConfig`.
    estimator
        Optional pre-built estimator.  By default one is constructed from
        ``cfg.estimator``, so switching between the stochastic-interpolant,
        EDM and CRPS variants is a one-line config change with an otherwise
        identical network.
    """

    def __init__(self, cfg: AtlasConfig, estimator: Estimator | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.n_channels

        self.resampler = Resampler(
            cfg.grid,
            cfg.latent.shape,
            cfg.latent.mode,
            n_channels=c,
            hidden_dim=cfg.latent.learned_encoder_dim,
            depth=cfg.latent.learned_encoder_depth,
        )
        self.normalizers = NormalizerPair(c)
        self.forcings = ForcingProvider(cfg.forcings, cfg.grid)

        self.backbone = LatentDiT(
            c, cfg.backbone, cfg.latent.shape, cfg.latent_grid, cfg.latent.history
        )
        self.projector = LocalProjector(
            c, c + self.forcings.n_channels, cfg.projector, cfg.grid, cfg.latent.shape
        )

        opts = dict(cfg.estimator.options)
        if cfg.estimator.kind == "crps" and cfg.backbone.noise_dim:
            opts.setdefault("noise_dim", cfg.backbone.noise_dim)
        self.estimator = estimator or build_estimator(cfg.estimator.kind, **opts)
        self.estimator.attach_sphere(SphericalTransform(cfg.latent_grid))

        aw = torch.from_numpy(cfg.latent_grid.area_weights()).squeeze(-1)
        self.register_buffer("latent_area", aw, persistent=False)
        aw_fine = torch.from_numpy(cfg.grid.area_weights()).squeeze(-1)
        self.register_buffer("fine_area", aw_fine, persistent=False)
        self.register_buffer(
            "channel_weights", torch.from_numpy(cfg.variables.weights()), persistent=False
        )

    # -- weights -----------------------------------------------------------

    @property
    def latent_weights(self) -> Weights:
        return Weights(self.latent_area, self.channel_weights)

    @property
    def fine_weights(self) -> Weights:
        return Weights(self.fine_area, self.channel_weights)

    def parameter_counts(self) -> dict[str, int]:
        return {
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
            "projector": sum(p.numel() for p in self.projector.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }

    # -- encoding ----------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Physical full-resolution state -> normalised latent state."""
        return self.resampler.encode(self.normalizers.state.normalize(x))

    def encode_residual(self, x_next: torch.Tensor, x_cur: torch.Tensor) -> torch.Tensor:
        """Physical states -> normalised latent residual (the model's target)."""
        return self.resampler.encode(self.normalizers.residual.normalize(x_next - x_cur))

    def latent_advance(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """``z_{j+1}`` from a latent state and a latent residual, both normalised."""
        phys = self.normalizers.state.denormalize(z) + self.normalizers.residual.denormalize(r)
        return self.normalizers.state.normalize(phys)

    def forcing_channels(
        self,
        times: np.ndarray | None,
        batch: int,
        device: torch.device | str,
        scalars: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        return self.forcings(times, batch, device=device, scalars=scalars)

    # -- prediction --------------------------------------------------------

    def predict_latent(
        self,
        z0: torch.Tensor,
        history: torch.Tensor | None = None,
        n_ensemble: int = 1,
        generator: torch.Generator | None = None,
        record_trajectory: bool = False,
        steps: int | None = None,
    ):
        """Draw ``n_ensemble`` latent residuals from ``rho_c(r_1 | z_0, z_-1)``."""
        return self.estimator.sample(
            self.backbone,
            z0,
            history,
            n_samples=n_ensemble,
            generator=generator,
            record_trajectory=record_trajectory,
            steps=steps,
        )

    def decode(
        self,
        latent_residual: torch.Tensor,
        x_fine: torch.Tensor,
        times: np.ndarray | None = None,
        scalars: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Project a latent residual to a full-resolution *physical* increment."""
        xn = self.normalizers.state.normalize(x_fine)
        forc = self.forcing_channels(times, xn.shape[0], xn.device, scalars)
        state_in = torch.cat([xn, forc], dim=1) if forc.shape[1] else xn
        out = self.projector(latent_residual, state_in)
        return self.normalizers.residual.denormalize(out)

    # -- one step ----------------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        x_fine: torch.Tensor,
        z0: torch.Tensor,
        history: torch.Tensor | None = None,
        times: np.ndarray | None = None,
        n_ensemble: int = 1,
        generator: torch.Generator | None = None,
        scalars: dict[str, Any] | None = None,
        record_trajectory: bool = False,
        latent_residual_override: torch.Tensor | None = None,
    ) -> StepOutput:
        """Advance the coupled (fine, latent) state by one model timestep.

        ``latent_residual_override`` replaces the sampled latent residual and
        is the cleanest causal-intervention point in the whole model: because
        the latent is a named, physically-scaled field, editing it asks a
        well-posed question ("what does the projector do if the coarse
        increment of q850 is doubled?").
        """
        if latent_residual_override is not None:
            r = latent_residual_override
            traj = None
            m = r.shape[0] // z0.shape[0]
        else:
            res = self.predict_latent(
                z0, history, n_ensemble, generator, record_trajectory
            )
            r, traj = res.sample, res.trajectory
            m = n_ensemble

        z_tiled = z0.repeat_interleave(m, dim=0) if m > 1 else z0
        x_tiled = x_fine.repeat_interleave(m, dim=0) if m > 1 else x_fine
        t_tiled = None
        if times is not None:
            t_tiled = np.repeat(np.atleast_1d(times), m) if m > 1 else np.atleast_1d(times)

        dx = self.decode(r, x_tiled, t_tiled, scalars)
        return StepOutput(
            state=x_tiled + dx,
            latent_state=self.latent_advance(z_tiled, r),
            latent_residual=r,
            fine_residual=dx,
            trajectory=traj,
            valid_time=t_tiled,
        )

    # -- rollout -----------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        x_window: torch.Tensor,
        times: np.ndarray,
        steps: int,
        n_ensemble: int = 1,
        generator: torch.Generator | None = None,
        scalars: dict[str, Any] | None = None,
        yield_initial: bool = True,
    ) -> Iterator[StepOutput]:
        """Autoregressive forecast.

        Parameters
        ----------
        x_window
            ``(B, K, C, nlat, nlon)`` physical states, oldest first, where
            ``K = cfg.latent.history + 1``.
        times
            ``(B,)`` ``datetime64`` valid times of the *last* state in the
            window.
        steps
            Number of model timesteps to integrate.
        n_ensemble
            Members drawn per initial condition.  Members are stacked into the
            batch dimension, so outputs have leading size ``B * n_ensemble``
            ordered as ``(b0m0, b0m1, ..., b1m0, ...)``.
        """
        k = self.cfg.latent.history + 1
        if x_window.shape[1] < k:
            raise ValueError(f"need at least {k} states in the window, got {x_window.shape[1]}")
        x_window = x_window[:, -k:]
        m = n_ensemble

        latents = [self.encode(x_window[:, i]) for i in range(k)]
        state = RolloutState(fine=x_window[:, -1], latents=latents, time=np.atleast_1d(times))

        if m > 1:
            state.fine = state.fine.repeat_interleave(m, dim=0)
            state.latents = [z.repeat_interleave(m, dim=0) for z in state.latents]
            state.time = np.repeat(state.time, m)

        if yield_initial:
            yield StepOutput(
                state=state.fine,
                latent_state=state.latents[-1],
                latent_residual=torch.zeros_like(state.latents[-1]),
                fine_residual=torch.zeros_like(state.fine),
                valid_time=state.time,
            )

        dt = np.timedelta64(int(round(self.cfg.dt_hours * 60)), "m")
        for _ in range(steps):
            out = self.step(
                state.fine,
                state.latents[-1],
                state.history_tensor(self.cfg.latent.history),
                times=state.time,
                n_ensemble=1,  # members already live in the batch dimension
                generator=generator,
                scalars=scalars,
            )
            state.fine = out.state
            state.latents = (state.latents + [out.latent_state])[-(k):]
            state.time = state.time + dt
            out.valid_time = state.time
            yield out

    # -- training ----------------------------------------------------------

    def training_losses(
        self,
        x_window: torch.Tensor,
        x_next: torch.Tensor,
        times: np.ndarray | None = None,
        scalars: dict[str, Any] | None = None,
        parts: Sequence[str] = ("latent", "projector", "encoder"),
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Joint objective for one training window.

        ``x_window`` is ``(B, K, C, H, W)`` ending at time ``j``; ``x_next`` is
        the state at ``j + 1``.  The two heads are independent -- the projector
        is trained on the *true* latent residual, never on a sampled one -- so
        they can also be optimised in separate runs by passing a single part,
        which is what the paper does (one decoder shared by all three
        estimators).
        """
        k = self.cfg.latent.history + 1
        x_window = x_window[:, -k:]
        x_cur = x_window[:, -1]

        z_all = [self.encode(x_window[:, i]) for i in range(k)]
        z0 = z_all[-1]
        history = torch.cat(z_all[:-1], dim=1) if self.cfg.latent.history else None
        r_target = self.encode_residual(x_next, x_cur)

        total = torch.zeros((), device=x_cur.device, dtype=x_cur.dtype)
        logs: dict[str, float] = {}

        if "latent" in parts:
            # A learnable encoder must not be optimised through the estimator:
            # the probabilistic head would be chasing a target free to move.
            # It is trained only through the projector's reconstruction loss.
            def cut(t: torch.Tensor | None) -> torch.Tensor | None:
                if t is None or not self.resampler.is_learned:
                    return t
                return t.detach()

            loss, info = self.estimator.loss(
                self.backbone, cut(z0), cut(history), cut(r_target), self.latent_weights
            )
            total = total + loss
            logs |= info

        if "projector" in parts:
            xn = self.normalizers.state.normalize(x_cur)
            forc = self.forcing_channels(times, xn.shape[0], xn.device, scalars)
            state_in = torch.cat([xn, forc], dim=1) if forc.shape[1] else xn
            # keep the graph through r_target when the encoder is learnable, so
            # encoder and projector train together as an autoencoder
            latent_in = r_target if self.resampler.is_learned else r_target.detach()
            pred = self.projector(latent_in, state_in)
            target = self.normalizers.residual.normalize(x_next - x_cur)
            ploss = l1_loss(pred, target, self.fine_weights)
            total = total + ploss
            logs["projector_l1"] = float(ploss.detach())

        logs["loss"] = float(total.detach())
        return total, logs

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.cfg.to_dict(), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: Any = "cpu", strict: bool = True) -> Atlas:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(AtlasConfig.from_dict(ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"], strict=strict)
        return model

    @classmethod
    def from_config(cls, path: str | Path) -> Atlas:
        return cls(AtlasConfig.load(path))
