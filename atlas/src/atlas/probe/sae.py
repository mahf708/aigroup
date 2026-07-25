"""Sparse dictionary learning on the residual stream.

A linear probe answers "is X in here?" for an X you thought of in advance.  A
sparse autoencoder answers the complementary question -- "what is in here?" --
by decomposing token activations into a sparse combination of learned
directions.  For a weather model the payoff is that each dictionary element
can be rendered as a **map**: the set of token positions and forecast times at
which it fires.  Features that light up along storm tracks, over convective
regions, or at the tropopause are legible in a way a raw activation is not.

The implementation is a top-k SAE, which avoids the L1 shrinkage bias and
makes sparsity an explicit knob rather than the outcome of a coefficient
search.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TopKSAE", "SAETrainConfig", "train_sae", "feature_maps", "feature_summary"]


class TopKSAE(nn.Module):
    """Top-k sparse autoencoder over ``d``-dimensional activations.

    Parameters
    ----------
    d_in
        Activation width (the DiT embedding dimension).
    n_features
        Dictionary size; ``8 * d_in`` to ``32 * d_in`` is the usual range.
    k
        Number of active features per token.
    """

    def __init__(self, d_in: int, n_features: int, k: int = 32, tied_init: bool = True) -> None:
        super().__init__()
        self.d_in = d_in
        self.n_features = n_features
        self.k = k
        self.encoder = nn.Linear(d_in, n_features, bias=True)
        self.decoder = nn.Linear(n_features, d_in, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_in))
        if tied_init:
            with torch.no_grad():
                self.decoder.weight.copy_(self.encoder.weight.T)
        self._normalize_decoder()
        self.register_buffer("fire_count", torch.zeros(n_features))
        self.register_buffer("n_seen", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(sparse_codes, dense_preactivations)``."""
        pre = self.encoder(x - self.pre_bias)
        vals, idx = pre.topk(self.k, dim=-1)
        codes = torch.zeros_like(pre).scatter_(-1, idx, F.relu(vals))
        return codes, pre

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(codes) + self.pre_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        codes, _ = self.encode(x)
        return self.decode(codes), codes

    @property
    def dead_features(self) -> torch.Tensor:
        return (self.fire_count == 0).nonzero(as_tuple=True)[0]

    def explained_variance(self, x: torch.Tensor) -> float:
        recon, _ = self(x)
        num = (x - recon).pow(2).sum()
        den = (x - x.mean(0, keepdim=True)).pow(2).sum()
        return float(1.0 - num / den.clamp_min(1e-12))


@dataclass
class SAETrainConfig:
    n_features: int = 4096
    k: int = 32
    lr: float = 1e-3
    epochs: int = 4
    batch_size: int = 4096
    aux_alpha: float = 1.0 / 32
    resample_dead_every: int | None = 2
    device: str = "cpu"
    seed: int = 0


def _batches(x: torch.Tensor, bs: int, gen: torch.Generator) -> Iterator[torch.Tensor]:
    perm = torch.randperm(x.shape[0], generator=gen)
    for s in range(0, x.shape[0], bs):
        yield x[perm[s : s + bs]]


def train_sae(
    activations: torch.Tensor, cfg: SAETrainConfig | None = None
) -> tuple[TopKSAE, dict[str, list[float]]]:
    """Fit a top-k SAE to a ``(n_tokens, d)`` activation matrix.

    Flatten batch and token dimensions before calling: every token is an
    independent sample, so a single forecast at 91x120 already provides ~11k
    rows, and a few dozen forecasts is usually enough to get stable features.
    """
    cfg = cfg or SAETrainConfig()
    gen = torch.Generator().manual_seed(cfg.seed)
    x = activations.reshape(-1, activations.shape[-1]).float()
    mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-6)
    xn = ((x - mu) / sd).to(cfg.device)

    sae = TopKSAE(xn.shape[1], cfg.n_features, cfg.k).to(cfg.device)
    sae.register_buffer("input_mean", mu.to(cfg.device))
    sae.register_buffer("input_std", sd.to(cfg.device))
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    history: dict[str, list[float]] = {"loss": [], "explained_variance": [], "dead": []}

    for _epoch in range(cfg.epochs):
        sae.fire_count.zero_()
        running = 0.0
        nb = 0
        for batch in _batches(xn, cfg.batch_size, gen):
            recon, codes = sae(batch)
            loss = F.mse_loss(recon, batch)
            if cfg.aux_alpha:
                # keep dead features alive by asking them to explain the residual
                dead = sae.dead_features
                if dead.numel():
                    resid = (batch - recon).detach()
                    pre = sae.encoder(batch - sae.pre_bias)[:, dead]
                    kk = min(sae.k, dead.numel())
                    vals, idx = pre.topk(kk, dim=-1)
                    aux = torch.zeros_like(pre).scatter_(-1, idx, F.relu(vals))
                    aux_recon = aux @ sae.decoder.weight[:, dead].T
                    loss = loss + cfg.aux_alpha * F.mse_loss(aux_recon, resid)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sae._normalize_decoder()
            with torch.no_grad():
                sae.fire_count += (codes > 0).float().sum(0)
                sae.n_seen += batch.shape[0]
            running += float(loss.detach())
            nb += 1
        history["loss"].append(running / max(nb, 1))
        with torch.no_grad():
            history["explained_variance"].append(sae.explained_variance(xn[: cfg.batch_size]))
            history["dead"].append(float(sae.dead_features.numel()))
    return sae, history


@torch.no_grad()
def feature_maps(
    sae: TopKSAE,
    activations: torch.Tensor,
    grid_hw: tuple[int, int],
    features: Iterable[int] | None = None,
) -> torch.Tensor:
    """Firing strength of selected features as ``(n_features, B, gh, gw)`` maps."""
    b, n, d = activations.shape
    x = activations.reshape(-1, d).to(sae.pre_bias.device)
    if hasattr(sae, "input_mean"):
        x = (x - sae.input_mean) / sae.input_std
    codes, _ = sae.encode(x)
    idx = torch.as_tensor(list(features), dtype=torch.long) if features is not None else None
    codes = codes if idx is None else codes[:, idx.to(codes.device)]
    return codes.reshape(b, grid_hw[0], grid_hw[1], -1).permute(3, 0, 1, 2).contiguous()


@torch.no_grad()
def feature_summary(
    sae: TopKSAE, activations: torch.Tensor, grid_hw: tuple[int, int], top: int = 20
) -> list[dict[str, float]]:
    """Rank features by activation mass and report where each one fires.

    ``concentration`` is the fraction of a feature's total firing mass held by
    its top 1% of token locations: values near 1 mean a geographically sharp
    feature (a jet exit region, a coastline), values near 0.01 mean it is
    diffuse.
    """
    b, n, d = activations.shape
    x = activations.reshape(-1, d).to(sae.pre_bias.device)
    if hasattr(sae, "input_mean"):
        x = (x - sae.input_mean) / sae.input_std
    codes, _ = sae.encode(x)
    mass = codes.sum(0)
    order = mass.argsort(descending=True)[:top]
    per_loc = codes.reshape(b, n, -1).sum(0)
    out = []
    n_top = max(1, n // 100)
    for f in order.tolist():
        col = per_loc[:, f]
        tot = col.sum().clamp_min(1e-12)
        best = int(col.argmax())
        out.append(
            {
                "feature": f,
                "mass": float(mass[f]),
                "fire_rate": float((codes[:, f] > 0).float().mean()),
                "concentration": float(col.topk(n_top).values.sum() / tot),
                "peak_row": best // grid_hw[1],
                "peak_col": best % grid_hw[1],
            }
        )
    return out
