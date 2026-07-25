"""Command line entry points: ``atlas info | demo | train | stats``."""

from __future__ import annotations

import argparse

import torch

from .data import SyntheticStore, WindowDataset, collate
from .model import Atlas
from .spec import AtlasConfig

__all__ = ["main"]


def _load_config(path: str) -> AtlasConfig:
    return AtlasConfig.load(path)


def _store_from_args(cfg: AtlasConfig, args: argparse.Namespace):
    if args.data in (None, "synthetic"):
        return SyntheticStore(cfg.variables, cfg.grid, n_times=args.n_times)
    import xarray as xr  # noqa: PLC0415

    from .data import XarrayStore  # noqa: PLC0415

    ds = xr.open_zarr(args.data) if args.data.endswith(".zarr") else xr.open_dataset(args.data)
    return XarrayStore(ds, cfg.variables, cfg.grid, level_dim=args.level_dim)


# ---------------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    model = Atlas(cfg)
    counts = model.parameter_counts()
    print(f"config      : {cfg.name}")
    print(f"channels    : {cfg.n_channels}")
    print(f"grid        : {cfg.grid.shape}  latent {cfg.latent.shape} ({cfg.latent.mode})")
    print(f"compression : {cfg.compression:.1f}x")
    print(f"token grid  : backbone {model.backbone.grid_hw}  projector {model.projector.grid_hw}")
    print(
        f"embed dim   : backbone {model.backbone.embed_dim}"
        f"  projector {model.projector.embed_dim}"
    )
    print(f"estimator   : {cfg.estimator.kind} {cfg.estimator.options}")
    print(f"forcings    : {model.forcings.n_channels} channels")
    for k, v in counts.items():
        print(f"params/{k:<10}: {v/1e6:.1f}M")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from .normalize import NormalizerPair, compute_statistics  # noqa: PLC0415

    cfg = _load_config(args.config)
    store = _store_from_args(cfg, args)
    ds = WindowDataset(store, history=cfg.latent.history)
    stats = compute_statistics(ds.stat_windows(stride=args.stride), max_samples=args.max_samples)
    norm = NormalizerPair(cfg.n_channels)
    norm.state.set_stats(stats["state_mean"], stats["state_std"])
    norm.residual.set_stats(stats["residual_mean"], stats["residual_std"])
    norm.save(args.out, cfg.variables)
    print(f"wrote {args.out}")
    for i, n in enumerate(cfg.variables.names[: args.show]):
        print(
            f"  {n:<10} state mu={stats['state_mean'][i]:+.4g} sd={stats['state_std'][i]:.4g}"
            f"   residual sd={stats['residual_std'][i]:.4g}"
        )
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """End-to-end sanity run on synthetic data: fit stats, train, roll out, score."""
    from .losses import Weights, ensemble_rmse, fair_crps, spread_skill_ratio  # noqa: PLC0415
    from .train import TrainConfig, Trainer, fit_normalizers  # noqa: PLC0415

    cfg = _load_config(args.config) if args.config else _demo_config()
    model = Atlas(cfg)
    store = SyntheticStore(cfg.variables, cfg.grid, n_times=args.n_times)
    ds = WindowDataset(store, history=cfg.latent.history)
    fit_normalizers(model, ds, max_samples=64)

    tcfg = TrainConfig(
        lr=3e-4,
        warmup_steps=10,
        cycle_steps=max(args.steps, 1),
        cycles=1,
        batch_size=args.batch_size,
        log_every=max(args.steps // 5, 1),
        checkpoint_every=0,
        out_dir=args.out_dir,
        device=args.device,
        amp_dtype=None,
    )
    trainer = Trainer(model, tcfg)
    trainer.fit(ds, max_steps=args.steps)

    model.eval()
    batch = collate([ds[i] for i in range(2)])
    x = batch["window"].to(args.device)
    truth = batch["next"].to(args.device)
    members = []
    rollout = model.rollout(
        x, batch["time"], steps=1, n_ensemble=args.members, yield_initial=False
    )
    for out in rollout:
        members.append(out.state)
    ens = members[-1].reshape(x.shape[0], args.members, *truth.shape[1:]).transpose(0, 1)
    w = Weights(model.fine_area, None)
    print(
        f"\nrmse={float(ensemble_rmse(ens, truth, w).mean()):.4f}  "
        f"crps={float(fair_crps(ens, truth, w).mean()):.4f}  "
        f"ssr={float(spread_skill_ratio(ens, truth, w).mean()):.3f}"
    )
    return 0


def _demo_config() -> AtlasConfig:
    from .spec import (  # noqa: PLC0415
        BackboneConfig,
        EstimatorConfig,
        ForcingConfig,
        GridSpec,
        LatentConfig,
        ProjectorConfig,
        VariableSet,
    )

    vs = VariableSet.era5_atlas()
    sub = VariableSet([vs[n] for n in ("u10m", "v10m", "t2m", "z500", "t850", "q850")])
    return AtlasConfig(
        name="atlas-demo",
        variables=sub,
        grid=GridSpec.equiangular(49, 96),
        latent=LatentConfig(shape=(13, 24), history=1),
        backbone=BackboneConfig(
            embed_dim_state=96, embed_dim_history=32, depth=4, num_heads=4, patch=(2, 3)
        ),
        projector=ProjectorConfig(
            embed_dim_state=64, embed_dim_latent=32, depth=3, num_heads=4
        ),
        estimator=EstimatorConfig(kind="si", options={"steps": 8}),
        forcings=ForcingConfig(cos_zenith=True, time_of_year=True),
    )


def cmd_train(args: argparse.Namespace) -> int:
    from .train import TrainConfig, Trainer, fit_normalizers  # noqa: PLC0415

    cfg = _load_config(args.config)
    model = Atlas(cfg)
    store = _store_from_args(cfg, args)
    ds = WindowDataset(store, history=cfg.latent.history, stride=args.stride)

    if args.stats:
        model.normalizers.load(args.stats, cfg.variables)
    else:
        print("no --stats given; estimating normalisation from the first windows")
        fit_normalizers(model, ds, max_samples=args.stat_samples)

    tcfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        cycle_steps=args.cycle_steps,
        cycles=args.cycles,
        parts=tuple(args.parts),
        out_dir=args.out_dir,
        device=args.device,
        num_workers=args.num_workers,
    )
    Trainer(model, tcfg).fit(ds, max_steps=args.max_steps)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("atlas", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="summarise a config")
    pi.add_argument("config")
    pi.set_defaults(func=cmd_info)

    ps = sub.add_parser("stats", help="compute normalisation statistics")
    ps.add_argument("config")
    ps.add_argument("--data", default="synthetic")
    ps.add_argument("--level-dim", default=None)
    ps.add_argument("--n-times", type=int, default=512)
    ps.add_argument("--stride", type=int, default=1)
    ps.add_argument("--max-samples", type=int, default=512)
    ps.add_argument("--show", type=int, default=8)
    ps.add_argument("--out", default="stats.npz")
    ps.set_defaults(func=cmd_stats)

    pd = sub.add_parser("demo", help="train and score a tiny model on synthetic data")
    pd.add_argument("--config", default=None)
    pd.add_argument("--steps", type=int, default=50)
    pd.add_argument("--batch-size", type=int, default=4)
    pd.add_argument("--members", type=int, default=4)
    pd.add_argument("--n-times", type=int, default=128)
    pd.add_argument("--device", default="cpu")
    pd.add_argument("--out-dir", default="runs/demo")
    pd.set_defaults(func=cmd_demo)

    pt = sub.add_parser("train", help="train from a config")
    pt.add_argument("config")
    pt.add_argument("--data", default="synthetic")
    pt.add_argument("--level-dim", default=None)
    pt.add_argument("--n-times", type=int, default=2048)
    pt.add_argument("--stats", default=None)
    pt.add_argument("--stat-samples", type=int, default=256)
    pt.add_argument("--stride", type=int, default=1)
    pt.add_argument("--lr", type=float, default=1.28e-4)
    pt.add_argument("--batch-size", type=int, default=8)
    pt.add_argument("--cycle-steps", type=int, default=100_000)
    pt.add_argument("--cycles", type=int, default=3)
    pt.add_argument("--max-steps", type=int, default=None)
    pt.add_argument(
        "--parts",
        nargs="+",
        default=["latent", "projector"],
        choices=["latent", "projector", "encoder"],
        help='heads to optimise; "encoder" applies only to latent.mode=learned',
    )
    pt.add_argument("--num-workers", type=int, default=0)
    pt.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pt.add_argument("--out-dir", default="runs/atlas")
    pt.set_defaults(func=cmd_train)

    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
