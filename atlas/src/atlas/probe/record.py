"""Capture what the model computes, without changing what it computes.

Two things are worth being explicit about before using any of this.

1. ATLAS's *latent* is not learned.  It is a bilinear downsampling, so latent
   channel ``c`` is literally a coarse-grained version of physical variable
   ``c``.  Questions of the form "what does the latent encode?" have a trivial
   answer; the interesting object is the **token residual stream** inside the
   DiT, which is learned, high-dimensional, and spatially localisable.
2. The residual stream lives on a coarse token grid (91x120 for the paper's
   configuration), so every activation can be given a latitude and longitude.
   That is the property this module exploits: activations come back as fields,
   not as anonymous vectors.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..layers import AttentionCapture, MultiheadAttention

__all__ = ["Recorder", "RecordedSite", "match_module", "token_field", "to_xarray"]


def match_module(name: str, pattern: str) -> bool:
    """Glob a module path, treating ``.`` as a path separator.

    ``*`` matches within one path segment and ``**`` matches any number of
    segments, so ``"backbone.blocks.*"`` selects the twelve DiT blocks and not
    the hundred-odd leaf modules inside them.  Plain :mod:`fnmatch` would match
    both, which is almost never what you want when recording a residual stream.
    """
    np_, pp = name.split("."), pattern.split(".")

    def rec(i: int, j: int) -> bool:
        if j == len(pp):
            return i == len(np_)
        if pp[j] == "**":
            return any(rec(k, j + 1) for k in range(i, len(np_) + 1))
        if i == len(np_):
            return False
        return fnmatch.fnmatch(np_[i], pp[j]) and rec(i + 1, j + 1)

    return rec(0, 0)


@dataclass
class RecordedSite:
    """Activations captured at one module."""

    name: str
    values: list[torch.Tensor] = field(default_factory=list)
    mode: str = "last"

    def tensor(self) -> torch.Tensor:
        """Stack according to ``mode``.

        ``last``
            The final call (a plain forward pass, or the last solver step).
        ``all``
            ``(n_calls, ...)`` -- for the SDE samplers this is the activation
            trajectory *through* the generative process, which is often more
            informative than any single snapshot.
        ``mean``
            Average over calls.
        """
        if not self.values:
            raise KeyError(f"nothing recorded at {self.name!r}")
        if self.mode == "last":
            return self.values[-1]
        if self.mode == "all":
            return torch.stack(self.values)
        if self.mode == "mean":
            return torch.stack(self.values).mean(0)
        raise ValueError(f"unknown mode {self.mode!r}")

    @property
    def n_calls(self) -> int:
        return len(self.values)


class Recorder:
    """Context manager that hooks named submodules of an :class:`~atlas.model.Atlas`.

    Examples
    --------
    >>> rec = Recorder(model, sites=["backbone.blocks.*"], mode="last")
    >>> with rec:
    ...     out = model.step(x, z0, history)
    >>> rec.field("backbone.blocks.11").shape   # (B, E, 91, 120)

    Parameters
    ----------
    sites
        Glob patterns matched against ``model.named_modules()``.  ``"*"``
        wildcards are supported, so ``"backbone.blocks.*"`` grabs the whole
        residual stream and ``"projector.blocks.0"`` grabs one decoder layer.
    mode
        How repeated calls are combined; see :class:`RecordedSite`.
    to_cpu, detach
        Keep memory bounded during long rollouts.  Detaching is on by default;
        turn it off when you need gradients through the recorded tensors (for
        attribution or integrated-gradients style analyses).
    """

    def __init__(
        self,
        model: nn.Module,
        sites: Sequence[str] = ("backbone.blocks.*",),
        mode: str = "last",
        to_cpu: bool = True,
        detach: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.patterns = list(sites)
        self.mode = mode
        self.to_cpu = to_cpu
        self.detach = detach
        self.dtype = dtype
        self.sites: dict[str, RecordedSite] = {}
        self._handles: list[Any] = []
        self._captures: dict[str, AttentionCapture] = {}

    # -- selection ---------------------------------------------------------

    def matched(self) -> list[tuple[str, nn.Module]]:
        out = []
        for name, mod in self.model.named_modules():
            if not name:
                continue
            if any(match_module(name, p) for p in self.patterns):
                out.append((name, mod))
        return out

    def capture_attention(
        self,
        patterns: str | Sequence[str] = "backbone.blocks.*.attn",
        query_indices: torch.Tensor | Sequence[int] | None = None,
        heads: Sequence[int] | None = None,
        stats: bool = False,
    ) -> Recorder:
        """Also collect attention weights from the matching attention modules.

        Storing a full global attention matrix is usually impossible (10,920
        tokens squared, per head, per layer), so pass ``query_indices`` to keep
        only the rows you care about -- that is the practical route to
        teleconnection maps -- or ``stats=True`` for per-head summaries
        accumulated in chunks.
        """
        pats = [patterns] if isinstance(patterns, str) else list(patterns)
        qi = None
        if query_indices is not None:
            qi = torch.as_tensor(list(query_indices), dtype=torch.long)
        hd = torch.as_tensor(list(heads), dtype=torch.long) if heads is not None else None
        for name, mod in self.model.named_modules():
            if isinstance(mod, MultiheadAttention) and any(
                match_module(name, p) for p in pats
            ):
                self._captures[name] = AttentionCapture(
                    query_indices=qi, heads=hd, stats=stats
                )
        if not self._captures:
            raise ValueError(f"no attention modules matched {pats}")
        return self

    # -- context management -------------------------------------------------

    def __enter__(self) -> Recorder:
        for name, mod in self.matched():
            self.sites.setdefault(name, RecordedSite(name, mode=self.mode))
            self._handles.append(mod.register_forward_hook(self._make_hook(name)))
        for name, cap in self._captures.items():
            cap.reset()
            dict(self.model.named_modules())[name].capture = cap
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for name in self._captures:
            dict(self.model.named_modules())[name].capture = None

    def _make_hook(self, name: str):
        def hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(t):
                return
            if self.detach:
                t = t.detach()
            if self.dtype is not None:
                t = t.to(self.dtype)
            if self.to_cpu:
                t = t.to("cpu")
            site = self.sites[name]
            if site.mode == "last":
                site.values = [t]
            else:
                site.values.append(t)

        return hook

    def clear(self) -> None:
        for site in self.sites.values():
            site.values.clear()
        for cap in self._captures.values():
            cap.reset()

    # -- access -------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self.sites

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.sites[name].tensor()

    def keys(self) -> list[str]:
        """Recorded site names in **layer order**.

        Plain string sorting would put ``blocks.10`` before ``blocks.2``, which
        silently scrambles any depth sweep on a model with ten or more blocks.
        """
        return sorted(self.sites, key=_layer_sort_key)

    def attention(self, name: str) -> AttentionCapture:
        key = name if name in self._captures else f"{name}.attn"
        return self._captures[key]

    def stack(self, pattern: str = "backbone.blocks.*") -> torch.Tensor:
        """Residual stream across depth: ``(n_layers, ...)`` in layer order."""
        names = [n for n in self.keys() if match_module(n, pattern)]
        names.sort(key=_layer_sort_key)
        if not names:
            raise KeyError(f"no recorded site matches {pattern!r}")
        return torch.stack([self[n] for n in names])

    def field(self, name: str, grid_hw: tuple[int, int] | None = None) -> torch.Tensor:
        """Reshape ``(B, N, E)`` tokens into ``(B, E, gh, gw)`` maps."""
        t = self[name]
        if grid_hw is None:
            grid_hw = self._infer_grid(name)
        return token_field(t, grid_hw)

    def _infer_grid(self, name: str) -> tuple[int, int]:
        root = name.split(".")[0]
        mod = getattr(self.model, root, None)
        gh = getattr(mod, "grid_hw", None)
        if gh is None:
            raise ValueError(f"cannot infer token grid for {name!r}; pass grid_hw")
        return tuple(gh)  # type: ignore[return-value]

    def token_latlon(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Token-centre coordinates for a recorded site."""
        root = name.split(".")[0]
        mod = getattr(self.model, root)
        grid = (
            self.model.cfg.latent_grid if root == "backbone" else self.model.cfg.grid
        )
        return mod.token_coords(grid)


def _layer_sort_key(name: str) -> tuple:
    parts = name.split(".")
    return tuple(int(p) if p.isdigit() else p for p in parts)


def token_field(tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    """``(..., N, E)`` -> ``(..., E, gh, gw)``."""
    gh, gw = grid_hw
    if tokens.shape[-2] != gh * gw:
        raise ValueError(
            f"token count {tokens.shape[-2]} does not match grid {grid_hw}"
        )
    lead = tokens.shape[:-2]
    x = tokens.reshape(*lead, gh, gw, tokens.shape[-1])
    return x.movedim(-1, -3).contiguous()


def to_xarray(
    values: torch.Tensor | np.ndarray,
    dims: Iterable[str],
    coords: dict[str, Any] | None = None,
    name: str | None = None,
    attrs: dict[str, Any] | None = None,
):
    """Wrap an array as an ``xarray.DataArray`` (import is optional)."""
    try:
        import xarray as xr  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "to_xarray needs xarray installed: pip install 'e3sm-atlas[analysis]'"
        ) from exc
    arr = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    return xr.DataArray(arr, dims=list(dims), coords=coords or {}, name=name, attrs=attrs or {})
