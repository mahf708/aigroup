"""Building blocks of the U-Cast backbone.

The layers follow the ADM / "DhariwalUNet" design used by EDM (Dhariwal & Nichol 2021; Karras et al.
2022) with the modifications U-Cast makes for weather:

* convolutions wrap around the longitude axis (:class:`SphereConv2d`),
* the adaptive-LayerNorm timestep conditioning is gone -- there is no diffusion step to condition on,
  and MC dropout replaces noise injection, so blocks are plain residual blocks,
* resampling takes an explicit target size, so grids whose extent is not a power of two (121x240)
  round-trip exactly through the encoder/decoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["SphereConv2d", "GroupNorm", "SelfAttention2d", "ResBlock", "downsample", "upsample"]


def _init_scale(mode: str, fan_in: int, fan_out: int) -> float:
    """Standard deviation / half-width used by EDM's magnitude-preserving initialisation."""
    if mode == "xavier_uniform":
        return math.sqrt(6 / (fan_in + fan_out))
    if mode == "xavier_normal":
        return math.sqrt(2 / (fan_in + fan_out))
    if mode == "kaiming_uniform":
        return math.sqrt(3 / fan_in)
    if mode == "kaiming_normal":
        return math.sqrt(1 / fan_in)
    raise ValueError(f"Unknown init mode {mode!r}")


def _init_tensor(shape: tuple[int, ...], mode: str, fan_in: int, fan_out: int, gain: float) -> Tensor:
    if gain == 0.0:
        return torch.zeros(shape)
    scale = _init_scale(mode, fan_in, fan_out) * gain
    if mode.endswith("uniform"):
        return scale * (torch.rand(shape) * 2 - 1)
    return scale * torch.randn(shape)


def downsample(x: Tensor, ceil_mode: bool = True) -> Tensor:
    """Halve the spatial extent by 2x2 average pooling.

    ``ceil_mode=True`` keeps the trailing row/column of odd-sized grids (e.g. the south pole of a
    121-point latitude axis) instead of discarding it.
    """
    return F.avg_pool2d(x, kernel_size=2, ceil_mode=ceil_mode)


def upsample(x: Tensor, size: tuple[int, int] | None = None) -> Tensor:
    """Nearest-neighbour upsampling to an explicit ``(H, W)``, or 2x when ``size`` is None."""
    if size is None:
        return F.interpolate(x, scale_factor=2, mode="nearest")
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=tuple(size), mode="nearest")


class SphereConv2d(nn.Module):
    """2-D convolution with periodic padding in longitude and configurable padding in latitude.

    Args:
        in_channels / out_channels: Channel counts.
        kernel: Square kernel size.  ``kernel=0`` makes the layer an identity (used for skip
            connections that only need resampling), in which case the channel counts must match.
        periodic_longitude: Wrap the last (longitude) axis instead of padding it.
        latitude_padding: ``"zeros"`` (as in the reference implementation) or ``"replicate"``, which
            avoids injecting artificial zeros next to the poles.
        init_mode / init_weight / init_bias: Initialisation, following EDM.  ``init_weight=0`` gives a
            zero-initialised layer, which keeps residual branches at identity at the start of training.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 3,
        bias: bool = True,
        periodic_longitude: bool = True,
        latitude_padding: str = "zeros",
        init_mode: str = "kaiming_uniform",
        init_weight: float = 1.0,
        init_bias: float = 0.0,
    ):
        super().__init__()
        if latitude_padding not in ("zeros", "replicate", "circular"):
            raise ValueError(f"latitude_padding must be zeros/replicate/circular, got {latitude_padding!r}")
        if kernel == 0 and in_channels != out_channels:
            raise ValueError(f"kernel=0 requires in_channels == out_channels ({in_channels} != {out_channels})")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.periodic_longitude = periodic_longitude
        self.latitude_padding = latitude_padding

        if kernel:
            fan_in, fan_out = in_channels * kernel**2, out_channels * kernel**2
            self.weight = nn.Parameter(
                _init_tensor((out_channels, in_channels, kernel, kernel), init_mode, fan_in, fan_out, init_weight)
            )
            self.bias = (
                nn.Parameter(_init_tensor((out_channels,), init_mode, fan_in, fan_out, init_bias)) if bias else None
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def pad(self, x: Tensor) -> Tensor:
        p = self.kernel // 2
        if p == 0:
            return x
        lon_mode = "circular" if self.periodic_longitude else "constant"
        x = F.pad(x, (p, p, 0, 0), mode=lon_mode)
        lat_mode = {"zeros": "constant", "replicate": "replicate", "circular": "circular"}[self.latitude_padding]
        return F.pad(x, (0, 0, p, p), mode=lat_mode)

    def forward(self, x: Tensor) -> Tensor:
        if self.weight is None:
            return x
        w = self.weight.to(x.dtype)
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.conv2d(self.pad(x), w, b)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.in_channels}, {self.out_channels}, kernel={self.kernel}, "
            f"periodic_longitude={self.periodic_longitude}, latitude_padding={self.latitude_padding!r}"
        )


class GroupNorm(nn.GroupNorm):
    """GroupNorm that picks the group count from the channel count, as in ADM/EDM."""

    def __init__(self, num_channels: int, num_groups: int = 32, min_channels_per_group: int = 4, eps: float = 1e-5):
        groups = max(1, min(num_groups, num_channels // min_channels_per_group))
        while num_channels % groups != 0:
            groups -= 1
        super().__init__(num_groups=groups, num_channels=num_channels, eps=eps)


class SelfAttention2d(nn.Module):
    """Multi-head self-attention over all grid points, used only at the coarse U-Net levels.

    This is what lets an otherwise local convolutional model represent non-local interactions
    (teleconnections, planetary waves) without any spherical or graph machinery.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int | None = None,
        channels_per_head: int = 64,
        eps: float = 1e-5,
        conv_kwargs: dict | None = None,
    ):
        super().__init__()
        conv_kwargs = dict(conv_kwargs or {})
        heads = num_heads if num_heads is not None else max(1, channels // channels_per_head)
        while channels % heads != 0:
            heads -= 1
        self.num_heads = heads
        self.norm = GroupNorm(channels, eps=eps)
        self.qkv = SphereConv2d(channels, channels * 3, kernel=1, **conv_kwargs)
        self.proj = SphereConv2d(channels, channels, kernel=1, **{**conv_kwargs, "init_weight": 0.0, "init_bias": 0.0})

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, h * w).transpose(-2, -1).unbind(1)
        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, HW, head_dim)
        out = out.transpose(-2, -1).reshape(b, c, h, w)
        return self.proj(out)


class ResBlock(nn.Module):
    """Pre-activation residual block with optional resampling and self-attention.

    Args:
        in_channels / out_channels: Channel counts.
        mode: ``"same"``, ``"down"`` (2x2 average pool) or ``"up"`` (nearest upsampling).
        attention: Append a self-attention block after the residual branch.
        dropout: MC-dropout rate.  Active in training *and*, deliberately, at inference time: the
            sampled dropout masks are U-Cast's only source of ensemble spread.
        skip_scale: Multiplier applied after each residual addition (``1/sqrt(2)`` is a common
            variance-preserving choice; the reference uses 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str = "same",
        attention: bool = False,
        num_heads: int | None = None,
        channels_per_head: int = 64,
        dropout: float = 0.0,
        eps: float = 1e-5,
        skip_scale: float = 1.0,
        pool_ceil_mode: bool = True,
        conv_kwargs: dict | None = None,
    ):
        super().__init__()
        if mode not in ("same", "up", "down"):
            raise ValueError(f"mode must be same/up/down, got {mode!r}")
        conv_kwargs = dict(conv_kwargs or {})
        zero_kwargs = {**conv_kwargs, "init_weight": 0.0, "init_bias": 0.0}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode
        self.skip_scale = skip_scale
        self.pool_ceil_mode = pool_ceil_mode

        self.norm0 = GroupNorm(in_channels, eps=eps)
        self.conv0 = SphereConv2d(in_channels, out_channels, kernel=3, **conv_kwargs)
        self.norm1 = GroupNorm(out_channels, eps=eps)
        self.dropout = nn.Dropout(p=dropout)
        self.conv1 = SphereConv2d(out_channels, out_channels, kernel=3, **zero_kwargs)
        self.skip = (
            SphereConv2d(in_channels, out_channels, kernel=1, **conv_kwargs)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.attention = (
            SelfAttention2d(
                out_channels,
                num_heads=num_heads,
                channels_per_head=channels_per_head,
                eps=eps,
                conv_kwargs=conv_kwargs,
            )
            if attention
            else None
        )

    def resample(self, x: Tensor, size: tuple[int, int] | None = None) -> Tensor:
        if self.mode == "down":
            return downsample(x, ceil_mode=self.pool_ceil_mode)
        if self.mode == "up":
            return upsample(x, size)
        return x

    def forward(self, x: Tensor, size: tuple[int, int] | None = None) -> Tensor:
        x = self.resample(x, size)
        h = self.conv0(F.silu(self.norm0(x)))
        h = self.conv1(self.dropout(F.silu(self.norm1(h))))
        x = (h + self.skip(x)) * self.skip_scale
        if self.attention is not None:
            x = (x + self.attention(x)) * self.skip_scale
        return x
