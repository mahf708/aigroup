"""The two networks of ATLAS.

``LatentDiT``
    Global-attention transformer that models the latent conditional law
    ``rho_c(r_1 | z_0, z_-1)``.  Shared, unchanged, by all three probabilistic
    estimators -- that interchangeability is the paper's central claim.

``LocalProjector``
    Deterministic local-attention transformer implementing the decoder
    ``D(r_1, x_0) ~= x_1 - x_0``.  It never sees the future high-resolution
    state, only a coarse residual plus the current fine state, so its job is
    strictly "resolve the small scales consistent with this coarse increment".

Both expose their token grid and a ``token_coords`` helper so activations can
be mapped back to latitude/longitude for the diagnostics in
:mod:`atlas.probe`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from .layers import DiTBlock, FinalLayer, FourierEmbedder, PatchEmbed, make_pos_embed, unpatchify
from .spec import BackboneConfig, GridSpec, ProjectorConfig

__all__ = ["LatentDiT", "LocalProjector", "build_blocks", "run_blocks", "token_coords"]


def token_coords(grid: GridSpec, grid_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Latitude/longitude of each token centre for a ``grid_hw`` token grid."""
    gh, gw = grid_hw
    lat = np.interp(
        (np.arange(gh) + 0.5) * grid.nlat / gh, np.arange(grid.nlat), grid.lat
    )
    lon0 = float(grid.lon[0])
    lon = lon0 + (np.arange(gw) + 0.5) * 360.0 / gw
    return lat, np.mod(lon, 360.0)


def build_blocks(
    dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float,
    attention: str,
    grid_hw: tuple[int, int],
    kernel: tuple[int, int],
    qk_norm: bool,
    periodic_lon: bool,
    chunk: int = 4096,
) -> nn.ModuleList:
    """A stack of DiT blocks, registered directly on the owning network.

    Kept flat (``backbone.blocks.7.attn`` rather than nested behind a trunk
    module) because every probe addresses sites by module path, and the path
    people guess is the one that should work.
    """
    return nn.ModuleList(
        [
            DiTBlock(
                dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                attention=attention,
                grid_hw=grid_hw,
                kernel=kernel,
                qk_norm=qk_norm,
                periodic_lon=periodic_lon,
                chunk=chunk,
            )
            for _ in range(depth)
        ]
    )


def run_blocks(
    blocks: nn.ModuleList,
    x: torch.Tensor,
    c: torch.Tensor,
    checkpoint_every: int | None,
    training: bool,
) -> torch.Tensor:
    for i, block in enumerate(blocks):
        if checkpoint_every and training and (i + 1) % checkpoint_every == 0:
            x = ckpt.checkpoint(block, x, c, use_reentrant=False)
        else:
            x = block(x, c)
    return x


class LatentDiT(nn.Module):
    """Predictive backbone operating entirely on the coarse latent grid.

    Two token streams are formed and fused:

    * **state** -- the noisy/interpolated latent residual concatenated with the
      current latent state ``z_0``;
    * **history** -- the previous ``history`` latent states.

    Both are patched onto the *same* token grid so that the fused embedding
    dimension is ``embed_dim_state + embed_dim_history`` (the paper's
    2496 + 832 = 3328).  Fusing by feature concatenation keeps the two streams
    linearly separable in the residual stream, which makes it possible to ask
    later how much of a token's representation came from the history.
    """

    def __init__(
        self,
        n_channels: int,
        cfg: BackboneConfig,
        latent_shape: tuple[int, int],
        latent_grid: GridSpec | None = None,
        history: int = 1,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_channels = n_channels
        self.latent_shape = tuple(latent_shape)
        self.history = history
        periodic = latent_grid.periodic_lon if latent_grid is not None else True

        self.state_embed = PatchEmbed(
            2 * n_channels, cfg.embed_dim_state, cfg.patch, self.latent_shape, periodic
        )
        self.grid_hw = self.state_embed.grid_hw
        self.pos_state = make_pos_embed(
            cfg.pos_embed, cfg.embed_dim_state, self.grid_hw, latent_grid
        )

        if history > 0:
            self.history_embed = PatchEmbed(
                history * n_channels,
                cfg.embed_dim_history,
                cfg.patch,
                self.latent_shape,
                periodic,
            )
            self.pos_history = make_pos_embed(
                cfg.pos_embed, cfg.embed_dim_history, self.grid_hw, latent_grid
            )
        else:
            self.history_embed = None
            self.pos_history = None

        dim = cfg.embed_dim if history > 0 else cfg.embed_dim_state
        self.embed_dim = dim
        self.t_embed = FourierEmbedder(dim)
        if cfg.noise_dim:
            self.noise_embed = nn.Sequential(
                nn.Linear(cfg.noise_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
            )
        else:
            self.noise_embed = None

        self.blocks = build_blocks(
            dim,
            cfg.depth,
            cfg.num_heads,
            cfg.mlp_ratio,
            cfg.attention,
            self.grid_hw,
            (3, 3),
            cfg.qk_norm,
            periodic,
        )
        self.final = FinalLayer(dim, cfg.patch[0] * cfg.patch[1] * n_channels)

    def token_coords(self, latent_grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
        return token_coords(latent_grid, self.grid_hw)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z0: torch.Tensor,
        history: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Raw network output on the latent grid.

        Preconditioning (the ``c_in`` / ``c_out`` / ``c_skip`` scalings) lives
        in the estimator, not here, so the same weights serve the stochastic
        interpolant, EDM and CRPS objectives.
        """
        tok = self.state_embed(torch.cat([x, z0], dim=1)) + self.pos_state
        if self.history_embed is not None:
            if history is None:
                raise ValueError("model was built with history > 0 but none was given")
            h = self.history_embed(history) + self.pos_history
            if self.cfg.combine == "concat":
                tok = torch.cat([tok, h], dim=-1)
            elif self.cfg.combine == "add":
                tok = tok + h
            elif self.cfg.combine == "mul":
                tok = tok * h
            else:
                raise ValueError(f"unknown combine {self.cfg.combine!r}")

        if t.ndim == 0:
            t = t.expand(x.shape[0])
        c = self.t_embed(t.reshape(-1))
        if self.noise_embed is not None and noise is not None:
            c = c + self.noise_embed(noise)

        tok = run_blocks(self.blocks, tok, c, self.cfg.grad_checkpoint_every, self.training)
        out = self.final(tok, c)
        return unpatchify(
            out, self.grid_hw, self.cfg.patch, self.n_channels, self.latent_shape
        )


class LocalProjector(nn.Module):
    """Decoder ``D(r, x_0)``: coarse residual + fine state -> fine residual.

    The fine state is tokenised with a strided convolution whose stride equals
    the compression ratio, so both inputs land on the *same* token grid as the
    latent.  Attention is local because the task -- inventing scales below the
    latent cutoff -- is local; the paper measures this to be an order of
    magnitude cheaper than global attention at 181x360 and no less accurate.
    """

    def __init__(
        self,
        n_channels: int,
        n_input_channels: int,
        cfg: ProjectorConfig,
        grid: GridSpec,
        latent_shape: tuple[int, int],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_channels = n_channels
        self.grid_shape = grid.shape
        self.latent_shape = tuple(latent_shape)

        patch = (
            -(-self.grid_shape[0] // self.latent_shape[0]),
            -(-self.grid_shape[1] // self.latent_shape[1]),
        )
        self.patch = patch
        self.state_embed = PatchEmbed(
            n_input_channels, cfg.embed_dim_state, patch, self.grid_shape, grid.periodic_lon
        )
        self.grid_hw = self.state_embed.grid_hw
        if self.grid_hw != self.latent_shape:
            raise ValueError(
                f"projector token grid {self.grid_hw} must equal the latent grid "
                f"{self.latent_shape}; adjust LatentConfig.shape or the data grid"
            )
        self.latent_embed = nn.Conv2d(n_channels, cfg.embed_dim_latent, kernel_size=1)

        dim = cfg.embed_dim
        self.embed_dim = dim
        self.pos = make_pos_embed(cfg.pos_embed, dim, self.grid_hw, grid)
        self.t_embed = FourierEmbedder(dim)
        self.blocks = build_blocks(
            dim,
            cfg.depth,
            cfg.num_heads,
            cfg.mlp_ratio,
            cfg.attention,
            self.grid_hw,
            cfg.kernel,
            cfg.qk_norm,
            grid.periodic_lon,
            chunk=cfg.neighborhood_chunk,
        )
        self.final = FinalLayer(dim, patch[0] * patch[1] * n_channels)

    def token_coords(self, grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
        return token_coords(grid, self.grid_hw)

    def forward(
        self, residual: torch.Tensor, state: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Fine-grid residual prediction.

        ``state`` carries the prognostic channels *plus* any forcing channels
        (cosine zenith angle, orography, land/sea mask, ...).
        """
        b = state.shape[0]
        tok_state = self.state_embed(state)
        tok_latent = self.latent_embed(residual).flatten(2).transpose(1, 2)
        tok = torch.cat([tok_state, tok_latent], dim=-1) + self.pos

        if t is None:
            t = torch.ones(b, device=state.device, dtype=torch.float32)
        c = self.t_embed(t.reshape(-1).to(state.dtype))
        tok = run_blocks(self.blocks, tok, c, self.cfg.grad_checkpoint_every, self.training)
        out = self.final(tok, c)
        return unpatchify(out, self.grid_hw, self.patch, self.n_channels, self.grid_shape)
