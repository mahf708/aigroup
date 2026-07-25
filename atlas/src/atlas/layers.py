"""Transformer building blocks.

Self-contained reimplementation of the DiT pieces ATLAS relies on: no
``timm``, ``einops``, ``natten`` or ``physicsnemo`` required, so the model can
be built, trained and inspected anywhere PyTorch runs.

Every attention module can be asked to expose its attention weights without
changing the values it returns, which is what :mod:`atlas.probe` builds on.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grids import SphericalPad, build_pos_embedding
from .spec import GridSpec

__all__ = [
    "AttentionCapture",
    "MultiheadAttention",
    "DiTBlock",
    "FinalLayer",
    "FourierEmbedder",
    "PatchEmbed",
    "modulate",
    "unpatchify",
]


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int, act: Callable[[], nn.Module] | None = None) -> None:
        super().__init__()
        act = act or (lambda: nn.GELU(approximate="tanh"))
        self.fc1 = nn.Linear(dim, hidden)
        self.act = act()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------


@dataclass
class AttentionCapture:
    """Request for attention diagnostics from one attention module.

    Attributes
    ----------
    query_indices
        Flattened token indices whose attention rows should be stored.  Storing
        the full ``N x N`` matrix of a global block is usually impossible
        (10,920 tokens x 13 heads), so probing a handful of query locations is
        the practical way to read off teleconnections.  ``None`` stores the
        full matrix -- only do that on small grids.
    heads
        Which heads to keep; ``None`` keeps all.
    stats
        Also accumulate per-head summary statistics (entropy and mean attended
        great-circle distance) over *all* queries, computed in chunks.
    """

    query_indices: torch.Tensor | None = None
    heads: torch.Tensor | None = None
    stats: bool = False
    chunk: int = 1024
    weights: torch.Tensor | None = field(default=None, repr=False)
    entropy: torch.Tensor | None = field(default=None, repr=False)
    distance: torch.Tensor | None = field(default=None, repr=False)

    def reset(self) -> None:
        self.weights = None
        self.entropy = None
        self.distance = None


class MultiheadAttention(nn.Module):
    """Self attention with global, neighborhood or windowed receptive field.

    ``mode="global"``
        Dense attention over all tokens -- the paper's predictive backbone,
        motivated by long-range atmospheric correlations.
    ``mode="neighborhood"``
        Exact ``kh x kw`` neighborhood attention on the token grid, with
        sphere-consistent halos.  This is the paper's projector setting (3x3)
        and is implemented by gathering keys/values, chunked to bound memory.
    ``mode="window"``
        Non-overlapping window attention.  Cheaper than neighborhood attention
        for large kernels; useful when experimenting on high-resolution grids.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mode: str = "global",
        grid_hw: tuple[int, int] | None = None,
        kernel: tuple[int, int] = (3, 3),
        qkv_bias: bool = True,
        qk_norm: bool = False,
        periodic_lon: bool = True,
        pad_mode: str = "pole",
        chunk: int = 4096,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.mode = mode
        self.grid_hw = tuple(grid_hw) if grid_hw is not None else None
        self.kernel = tuple(kernel)
        self.chunk = chunk

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

        if mode in {"neighborhood", "window"}:
            if self.grid_hw is None:
                raise ValueError(f"mode={mode!r} requires grid_hw")
            self.pad = SphericalPad(
                pad_lat=self.kernel[0] // 2,
                pad_lon=self.kernel[1] // 2,
                mode=pad_mode,
                periodic_lon=periodic_lon,
            )
        else:
            self.pad = None

        #: set by :class:`atlas.probe.record.Recorder`; ``None`` disables capture
        self.capture: AttentionCapture | None = None

    # -- helpers -----------------------------------------------------------

    def _qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, H, N, hd)
        return self.q_norm(q), self.k_norm(k), v

    def _gather_neighbors(self, t: torch.Tensor) -> torch.Tensor:
        """(B, H, N, hd) -> (B, H, N, kh*kw, hd) over the token grid."""
        b, h, n, d = t.shape
        gh, gw = self.grid_hw  # type: ignore[misc]
        kh, kw = self.kernel
        x = t.permute(0, 1, 3, 2).reshape(b, h * d, gh, gw)
        x = self.pad(x)  # type: ignore[misc]
        x = F.unfold(x, kernel_size=(kh, kw))  # (B, h*d*kh*kw, N)
        x = x.reshape(b, h, d, kh * kw, n)
        return x.permute(0, 1, 4, 3, 2).contiguous()

    def _neighborhood_attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        kk = self._gather_neighbors(k)
        vv = self._gather_neighbors(v)
        out = torch.empty_like(q)
        n = q.shape[2]
        step = max(1, self.chunk)
        for s in range(0, n, step):
            e = min(n, s + step)
            qs = q[:, :, s:e].unsqueeze(-2)  # (B, H, n, 1, hd)
            logits = (qs * kk[:, :, s:e]).sum(-1) * self.scale  # (B, H, n, k)
            w = logits.softmax(dim=-1)
            out[:, :, s:e] = torch.einsum("bhnk,bhnkd->bhnd", w, vv[:, :, s:e])
            if self.capture is not None and self.capture.query_indices is None:
                self._stash_neighborhood(w, s)
        return out

    def _stash_neighborhood(self, w: torch.Tensor, offset: int) -> None:
        cap = self.capture
        assert cap is not None
        buf = cap.weights
        if buf is None:
            b, h, _, k = w.shape
            n = self.grid_hw[0] * self.grid_hw[1]  # type: ignore[index]
            buf = torch.zeros(b, h, n, k, device="cpu", dtype=w.dtype)
            cap.weights = buf
        buf[:, :, offset : offset + w.shape[2]] = w.detach().to("cpu")

    def _window_attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        b, h, n, d = q.shape
        gh, gw = self.grid_hw  # type: ignore[misc]
        wh, ww = self.kernel
        ph, pw = (-gh) % wh, (-gw) % ww

        def to_windows(t: torch.Tensor) -> torch.Tensor:
            t = t.permute(0, 1, 3, 2).reshape(b, h * d, gh, gw)
            if pw:
                t = F.pad(t, (0, pw, 0, 0), mode="circular")
            if ph:
                t = F.pad(t, (0, 0, 0, ph), mode="replicate")
            H, W = t.shape[-2:]
            t = t.reshape(b, h, d, H // wh, wh, W // ww, ww)
            t = t.permute(0, 1, 3, 5, 4, 6, 2)
            return t.reshape(b, h * (H // wh) * (W // ww), wh * ww, d)

        qs, ks, vs = to_windows(q), to_windows(k), to_windows(v)
        o = F.scaled_dot_product_attention(qs, ks, vs)
        H, W = gh + ph, gw + pw
        o = o.reshape(b, h, H // wh, W // ww, wh, ww, d)
        o = o.permute(0, 1, 6, 2, 4, 3, 5).reshape(b, h, d, H, W)
        o = o[..., :gh, :gw].reshape(b, h, d, n).permute(0, 1, 3, 2)
        return o.contiguous()

    def _capture_global(self, q: torch.Tensor, k: torch.Tensor) -> None:
        cap = self.capture
        assert cap is not None
        heads = cap.heads
        qh = q if heads is None else q[:, heads.to(q.device)]
        kh = k if heads is None else k[:, heads.to(k.device)]
        if cap.query_indices is not None:
            idx = cap.query_indices.to(q.device)
            logits = torch.einsum("bhqd,bhkd->bhqk", qh[:, :, idx], kh) * self.scale
            cap.weights = logits.softmax(-1).detach().to("cpu")
        if cap.stats:
            n = q.shape[2]
            ent = torch.zeros(qh.shape[0], qh.shape[1], device=q.device)
            for s in range(0, n, cap.chunk):
                e = min(n, s + cap.chunk)
                w = (
                    torch.einsum("bhqd,bhkd->bhqk", qh[:, :, s:e], kh) * self.scale
                ).softmax(-1)
                ent += -(w * (w + 1e-12).log()).sum(-1).sum(-1)
            cap.entropy = (ent / n).detach().to("cpu")

    # -- forward -----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._qkv(x)
        if self.mode == "global":
            if self.capture is not None:
                self._capture_global(q, k)
            o = F.scaled_dot_product_attention(q, k, v)
        elif self.mode == "neighborhood":
            o = self._neighborhood_attend(q, k, v)
        elif self.mode == "window":
            o = self._window_attend(q, k, v)
        else:
            raise ValueError(f"unknown attention mode {self.mode!r}")
        b, h, n, d = o.shape
        o = o.transpose(1, 2).reshape(b, n, h * d)
        return self.proj(o)


# ---------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------


class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning.

    The conditioning vector ``c`` carries the diffusion/interpolant time and,
    for the CRPS estimator, the FiLM-projected noise vector.  In the projector
    it is held at ``t = 1``: the paper keeps the modulation path rather than
    stripping it, and reports that this trains better.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attention: str = "global",
        grid_hw: tuple[int, int] | None = None,
        kernel: tuple[int, int] = (3, 3),
        qk_norm: bool = False,
        periodic_lon: bool = True,
        chunk: int = 4096,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiheadAttention(
            dim,
            num_heads,
            mode=attention,
            grid_hw=grid_hw,
            kernel=kernel,
            qk_norm=qk_norm,
            periodic_lon=periodic_lon,
            chunk=chunk,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.zeros_(self.ada_ln[1].weight)
        nn.init.zeros_(self.ada_ln[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada_ln(c).chunk(6, dim=1)
        x = x + gate_a.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_m.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_dim)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        nn.init.zeros_(self.ada_ln[1].weight)
        nn.init.zeros_(self.ada_ln[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class FourierEmbedder(nn.Module):
    """Sinusoidal embedding of a scalar (or short vector) into ``out_dim``."""

    def __init__(
        self,
        out_dim: int,
        freq_dim: int = 256,
        input_multiplier: float = 1000.0,
        max_period: float = 10000.0,
    ) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.input_multiplier = input_multiplier
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(1)
        t = t * self.input_multiplier
        half = (self.freq_dim // t.shape[-1]) // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.unsqueeze(-1).float() * freqs.view(1, 1, -1)
        emb = torch.cat([args.cos(), args.sin()], dim=-1).reshape(t.shape[0], -1)
        if emb.shape[-1] < self.freq_dim:
            emb = F.pad(emb, (0, self.freq_dim - emb.shape[-1]))
        return self.mlp(emb[:, : self.freq_dim])


class PatchEmbed(nn.Module):
    """Strided-convolution tokeniser with sphere-aware padding to a fixed grid."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch: tuple[int, int],
        input_hw: tuple[int, int],
        periodic_lon: bool = True,
    ) -> None:
        super().__init__()
        self.patch = tuple(patch)
        self.input_hw = tuple(input_hw)
        self.periodic_lon = periodic_lon
        self.pad_h = (-self.input_hw[0]) % self.patch[0]
        self.pad_w = (-self.input_hw[1]) % self.patch[1]
        self.grid_hw = (
            (self.input_hw[0] + self.pad_h) // self.patch[0],
            (self.input_hw[1] + self.pad_w) // self.patch[1],
        )
        self.num_patches = self.grid_hw[0] * self.grid_hw[1]
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch, stride=self.patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_w:
            x = F.pad(
                x, (0, self.pad_w, 0, 0),
                mode="circular" if self.periodic_lon else "replicate",
            )
        if self.pad_h:
            x = F.pad(x, (0, 0, 0, self.pad_h), mode="replicate")
        return self.proj(x).flatten(2).transpose(1, 2)


def unpatchify(
    tokens: torch.Tensor,
    grid_hw: tuple[int, int],
    patch: tuple[int, int],
    channels: int,
    crop_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    """``(B, N, patch_h * patch_w * C)`` -> ``(B, C, H, W)``."""
    b = tokens.shape[0]
    gh, gw = grid_hw
    ph, pw = patch
    x = tokens.reshape(b, gh, gw, ph, pw, channels)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, channels, gh * ph, gw * pw)
    if crop_hw is not None:
        x = x[..., : crop_hw[0], : crop_hw[1]]
    return x


def make_pos_embed(
    kind: str, dim: int, grid_hw: tuple[int, int], grid: GridSpec | None
) -> nn.Parameter:
    return build_pos_embedding(kind, dim, grid_hw[0], grid_hw[1], grid)
