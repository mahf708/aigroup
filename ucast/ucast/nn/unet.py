"""The U-Cast backbone: a plain U-Net with bottleneck self-attention.

Deliberately unremarkable.  The point of the paper is that this architecture -- no graph, no spherical
harmonics, no diffusion sampler -- reaches frontier probabilistic skill once it is scaled up and
trained with the MAE -> CRPS curriculum.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .layers import ResBlock, SphereConv2d, upsample

__all__ = ["UCastUNet"]


def _level_sizes(shape: tuple[int, int], num_levels: int, ceil_mode: bool = True) -> list[tuple[int, int]]:
    """Spatial extent at each U-Net level, mirroring how :func:`~ucast.nn.layers.downsample` shrinks."""
    sizes = [tuple(int(s) for s in shape)]
    for _ in range(num_levels - 1):
        h, w = sizes[-1]
        if ceil_mode:
            sizes.append((-(-h // 2), -(-w // 2)))
        else:
            sizes.append((h // 2, w // 2))
    return sizes  # type: ignore[return-value]


class UCastUNet(nn.Module):
    """U-Net mapping a stack of input channels to a single-step forecast increment.

    Args:
        in_channels: Total input channels (state window + forcings + statics, already concatenated).
        out_channels: Predicted channels (one forecast state).
        spatial_shape: ``(H, W)`` of the input grid.  Any size works; odd extents such as 121 are
            handled by resampling to explicitly recorded per-level sizes.
        model_channels: Width of the finest level (the paper uses 320, giving ~900M parameters).
        channel_mult: Width multiplier per level; its length is the number of levels.
        num_blocks: Residual blocks per level and per encoder/decoder side.
        attn_levels: Levels (0 = finest) that get self-attention.  Negative indices count from the
            coarsest level, so ``(-2, -1)`` always means "the two coarsest levels".
        channels_per_head / num_heads: Attention head sizing.
        dropout: MC-dropout rate; the sole source of ensemble spread at inference.
        periodic_longitude: Wrap convolutions around the longitude axis (set False for regional grids).
        latitude_padding: ``"zeros"`` or ``"replicate"`` padding across the poles.
        skip_scale: Residual-branch scaling (see :class:`~ucast.nn.layers.ResBlock`).
        pool_ceil_mode: Keep (rather than drop) the trailing row/column when halving an odd extent.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_shape: Sequence[int],
        model_channels: int = 320,
        channel_mult: Sequence[int] = (1, 2, 3, 4),
        num_blocks: int = 4,
        attn_levels: Sequence[int] = (-2, -1),
        channels_per_head: int = 64,
        num_heads: int | None = None,
        dropout: float = 0.1,
        periodic_longitude: bool = True,
        latitude_padding: str = "zeros",
        skip_scale: float = 1.0,
        pool_ceil_mode: bool = True,
    ):
        super().__init__()
        spatial_shape = tuple(int(s) for s in spatial_shape)
        if len(spatial_shape) != 2:
            raise ValueError(f"spatial_shape must be (H, W), got {spatial_shape}")
        channel_mult = tuple(int(m) for m in channel_mult)
        num_levels = len(channel_mult)
        if num_levels < 1:
            raise ValueError("channel_mult must not be empty")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_shape = spatial_shape
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.level_sizes = _level_sizes(spatial_shape, num_levels, ceil_mode=pool_ceil_mode)
        self.attn_levels = tuple(sorted({lvl % num_levels for lvl in attn_levels}))

        conv_kwargs = dict(periodic_longitude=periodic_longitude, latitude_padding=latitude_padding)
        block_kwargs = dict(
            channels_per_head=channels_per_head,
            num_heads=num_heads,
            dropout=dropout,
            skip_scale=skip_scale,
            pool_ceil_mode=pool_ceil_mode,
            conv_kwargs=conv_kwargs,
        )

        # ---------------------------------------------------------------- Encoder
        self.stem = SphereConv2d(in_channels, model_channels * channel_mult[0], kernel=3, **conv_kwargs)
        self.enc_down = nn.ModuleList()  # one entry per level; None-equivalent at the finest level
        self.enc_blocks = nn.ModuleList()
        skip_channels: list[int] = [model_channels * channel_mult[0]]  # the stem output is a skip too
        channels = model_channels * channel_mult[0]
        for level, mult in enumerate(channel_mult):
            if level == 0:
                self.enc_down.append(nn.Identity())
            else:
                self.enc_down.append(ResBlock(channels, channels, mode="down", **block_kwargs))
                skip_channels.append(channels)
            blocks = nn.ModuleList()
            for _ in range(num_blocks):
                blocks.append(
                    ResBlock(channels, model_channels * mult, attention=level in self.attn_levels, **block_kwargs)
                )
                channels = model_channels * mult
                skip_channels.append(channels)
            self.enc_blocks.append(blocks)

        # ---------------------------------------------------------------- Bottleneck
        self.mid_block1 = ResBlock(channels, channels, attention=True, **block_kwargs)
        self.mid_block2 = ResBlock(channels, channels, **block_kwargs)

        # ---------------------------------------------------------------- Decoder
        self.dec_up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for level in reversed(range(num_levels)):
            self.dec_up.append(
                nn.Identity() if level == num_levels - 1 else ResBlock(channels, channels, mode="up", **block_kwargs)
            )
            blocks = nn.ModuleList()
            for _ in range(num_blocks + 1):
                cin = channels + skip_channels.pop()
                blocks.append(
                    ResBlock(
                        cin,
                        model_channels * channel_mult[level],
                        attention=level in self.attn_levels,
                        **block_kwargs,
                    )
                )
                channels = model_channels * channel_mult[level]
            self.dec_blocks.append(blocks)
        if skip_channels:  # pragma: no cover - guards a construction bug, not user input
            raise AssertionError(f"{len(skip_channels)} skip connections were left unconsumed")

        self.out_norm = nn.GroupNorm(num_groups=min(32, channels // 4), num_channels=channels)
        self.out_conv = SphereConv2d(
            channels, out_channels, kernel=3, init_weight=0.0, init_bias=0.0, **conv_kwargs
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(B, in_channels, H, W)`` to ``(B, out_channels, H, W)``."""
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {x.shape[1]}")
        input_size = tuple(x.shape[-2:])

        skips: list[Tensor] = []
        x = self.stem(x)
        skips.append(x)
        for level, blocks in enumerate(self.enc_blocks):
            if level > 0:
                x = self.enc_down[level](x)
                skips.append(x)
            for block in blocks:
                x = block(x)
                skips.append(x)

        x = self.mid_block2(self.mid_block1(x))

        num_levels = len(self.channel_mult)
        for i, blocks in enumerate(self.dec_blocks):
            level = num_levels - 1 - i
            if level != num_levels - 1:
                x = self.dec_up[i](x, self.level_sizes[level])
            for block in blocks:
                skip = skips.pop()
                if skip.shape[-2:] != x.shape[-2:]:  # only possible if level_sizes drifted
                    x = upsample(x, tuple(skip.shape[-2:]))
                x = block(torch.cat([x, skip], dim=1))

        x = self.out_conv(torch.nn.functional.silu(self.out_norm(x)))
        if tuple(x.shape[-2:]) != input_size:  # pragma: no cover - defensive
            x = upsample(x, input_size)
        return x
