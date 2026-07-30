"""Command line interface.

.. code-block:: bash

    ucast summary  --config configs/synthetic_smoke.yaml       # shapes and parameter counts
    ucast stats    --config configs/era5_1p5deg_stage1.yaml -o stats.npz
    ucast train    --config configs/era5_1p5deg_stage1.yaml    # Stage 1 (deterministic, MAE)
    ucast train    --config configs/era5_1p5deg_stage2.yaml \\
                   train.init_from=runs/stage1/last.ckpt       # Stage 2 (probabilistic, CRPS)
    ucast score    --checkpoint runs/stage2_seed*/best.ckpt --ensemble-size 8
    ucast forecast --checkpoint runs/stage2/best.ckpt --items 0 1 -o forecast.nc

Any config field can be overridden inline with ``dotted.key=value`` arguments, which are parsed as YAML
(so ``model.channel_mult=[1,2,2]`` and ``train.compile=true`` both work).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import ExperimentConfig, load_config
from .utils import format_count, get_logger, resolve_device, setup_logging

__all__ = ["main"]

log = get_logger("ucast")


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", "-c", type=Path, default=None, help="YAML config file")
    parser.add_argument("overrides", nargs="*", help="config overrides as dotted.key=value")


def _load(args: argparse.Namespace) -> ExperimentConfig:
    return load_config(args.config, args.overrides)


# ---------------------------------------------------------------------------------- commands
def cmd_train(args: argparse.Namespace) -> int:
    from .train import Trainer

    config = _load(args)
    scores = Trainer(config, device=args.device).fit()
    if scores:
        monitor = config.train.monitor
        log.info("Finished. %s = %s", monitor, scores.get(monitor, "n/a"))
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """Build the model (and one batch) without training, to check shapes and cost."""
    import torch

    from .data import build_datasets, build_normalizer
    from .forecaster import build_forecaster

    config = _load(args)
    datasets = build_datasets(config, splits=("train",))
    dataset = datasets["train"]
    spec = dataset.spec
    normalizer = build_normalizer(
        spec.input_variables, config.data.statistics or "identity", dataset=dataset, max_samples=8
    )
    model = build_forecaster(config, spec, normalizer)

    print(f"dataset       : {dataset.describe()}")
    print(f"grid          : {spec.grid.shape} (periodic longitude: {spec.grid.periodic_longitude})")
    print(f"level sizes   : {model.net.level_sizes}")
    print(f"input channels: {spec.num_input_channels} = {len(spec.input_variables)} x {spec.window} window"
          f" + {spec.num_forcing_channels} forcings + {spec.num_static_channels} statics")
    print(f"output channels: {spec.num_output_channels}")
    print(f"parameters    : {format_count(model.num_parameters)}")
    print(f"loss / members: {config.train.resolved_loss()} / {config.train.resolved_ensemble_members()}")

    if not args.no_forward:
        device = resolve_device(args.device)
        model = model.to(device)
        batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in dataset[0].items()}
        with torch.no_grad():
            forecast = model.rollout(batch, steps=min(2, dataset.rollout_steps), ensemble_size=2)
        print(f"rollout shape : {tuple(forecast.shape)} (members, batch, steps, channels, lat, lon)")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Estimate normalisation statistics from the training split and save them to a .npz file."""
    from .data import build_datasets
    from .normalization import compute_statistics

    config = _load(args)
    dataset = build_datasets(config, splits=("train",))["train"]
    normalizer = compute_statistics(
        dataset, dataset.spec.input_variables, max_samples=args.max_samples
    )
    normalizer.save(args.output)
    log.info("Wrote statistics for %d channels to %s", len(dataset.spec.input_variables), args.output)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from .data import build_datasets
    from .inference import DeepEnsemble, evaluate

    paths = _expand_checkpoints(args.checkpoint)
    ensemble, config = DeepEnsemble.from_checkpoints(
        paths,
        device=args.device,
        use_ema=not args.no_ema,
        members_per_model=args.ensemble_size,
        members_per_forward=args.members_per_forward,
        use_mc_dropout=not args.deterministic,
    )
    if args.config is not None or args.overrides:
        config = _load(args)
    if args.steps is not None:
        config.eval.rollout_steps = args.steps
    dataset = build_datasets(config, splits=("val",))["val"]

    scores = evaluate(
        ensemble,
        dataset,
        steps=args.steps or config.eval.rollout_steps,
        lead_times=args.lead_times,
        batch_size=args.batch_size,
        num_workers=config.data.num_workers,
        max_batches=args.max_batches,
        device=args.device,
    )
    _report(scores, args.output)
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    import torch

    from .data import build_datasets
    from .inference import DeepEnsemble, save_forecast

    paths = _expand_checkpoints(args.checkpoint)
    ensemble, config = DeepEnsemble.from_checkpoints(
        paths,
        device=args.device,
        use_ema=not args.no_ema,
        members_per_model=args.ensemble_size,
        members_per_forward=args.members_per_forward,
        use_mc_dropout=not args.deterministic,
    )
    if args.config is not None or args.overrides:
        config = _load(args)
    dataset = build_datasets(config, splits=("val",))["val"]
    steps = min(args.steps or config.eval.rollout_steps, dataset.rollout_steps)

    items = args.items or [0]
    batch_items = [dataset[i] for i in items]
    batch = {
        key: torch.stack([item[key] for item in batch_items]).to(resolve_device(args.device))
        for key in batch_items[0]
    }
    forecast = ensemble.rollout(batch, steps=steps)
    times = dataset.initial_times()[items] if hasattr(dataset, "initial_times") else list(range(len(items)))
    save_forecast(args.output, forecast, ensemble.spec, times)
    return 0


# ---------------------------------------------------------------------------------- helpers
def _expand_checkpoints(patterns: Sequence[str]) -> list[Path]:
    """Expand shell-style globs so ``--checkpoint 'runs/seed*/best.ckpt'`` builds a deep ensemble."""
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path().glob(pattern)) if any(c in pattern for c in "*?[") else [Path(pattern)]
        if not matches:
            raise FileNotFoundError(f"no checkpoints matched {pattern!r}")
        paths.extend(matches)
    return paths


def _report(scores: dict, output: Path | None) -> None:
    summary = {k: v for k, v in scores.items() if k.startswith("avg/") or "/" not in k}
    width = max((len(k) for k in summary), default=0)
    for key, value in summary.items():
        print(f"{key:<{width}}  {value:.6g}")
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(scores, indent=2))
        log.info("Wrote %d metrics to %s", len(scores), output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ucast", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default=None, help="cuda, cuda:1, cpu, mps (default: auto)")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train one curriculum stage")
    _add_config_arguments(train_parser)
    train_parser.set_defaults(func=cmd_train)

    summary_parser = subparsers.add_parser("summary", help="print model/dataset shapes and parameter counts")
    _add_config_arguments(summary_parser)
    summary_parser.add_argument("--no-forward", action="store_true", help="skip the test rollout")
    summary_parser.set_defaults(func=cmd_summary)

    stats_parser = subparsers.add_parser("stats", help="compute normalisation statistics")
    _add_config_arguments(stats_parser)
    stats_parser.add_argument("--output", "-o", type=Path, default=Path("statistics.npz"))
    stats_parser.add_argument("--max-samples", type=int, default=512)
    stats_parser.set_defaults(func=cmd_stats)

    score_parser = subparsers.add_parser("score", help="score checkpoints on the validation split")
    _add_config_arguments(score_parser)
    score_parser.add_argument("--checkpoint", "-k", nargs="+", required=True, help="checkpoint paths or globs")
    score_parser.add_argument("--ensemble-size", type=int, default=4, help="MC-dropout members per checkpoint")
    score_parser.add_argument("--members-per-forward", type=int, default=None)
    score_parser.add_argument("--steps", type=int, default=None, help="autoregressive steps")
    score_parser.add_argument("--lead-times", type=int, nargs="*", default=None, help="steps to score (1-based)")
    score_parser.add_argument("--batch-size", type=int, default=1)
    score_parser.add_argument("--max-batches", type=int, default=None)
    score_parser.add_argument("--no-ema", action="store_true")
    score_parser.add_argument("--deterministic", action="store_true", help="disable MC dropout")
    score_parser.add_argument("--output", "-o", type=Path, default=None, help="write all metrics as JSON")
    score_parser.set_defaults(func=cmd_score)

    forecast_parser = subparsers.add_parser("forecast", help="write forecasts to NetCDF/zarr")
    _add_config_arguments(forecast_parser)
    forecast_parser.add_argument("--checkpoint", "-k", nargs="+", required=True)
    forecast_parser.add_argument("--ensemble-size", type=int, default=4)
    forecast_parser.add_argument("--members-per-forward", type=int, default=None)
    forecast_parser.add_argument("--steps", type=int, default=None)
    forecast_parser.add_argument("--items", type=int, nargs="*", default=None, help="dataset item indices")
    forecast_parser.add_argument("--no-ema", action="store_true")
    forecast_parser.add_argument("--deterministic", action="store_true")
    forecast_parser.add_argument("--output", "-o", type=Path, default=Path("forecast.nc"))
    forecast_parser.set_defaults(func=cmd_forecast)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level.upper())
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
