"""End-to-end tests: both curriculum stages, checkpoint handoff, deep ensembling and the CLI."""

import json

import pytest
import torch

from ucast.checkpoint import load_forecaster, save_checkpoint
from ucast.cli import main
from ucast.config import ExperimentConfig
from ucast.data import SyntheticForecastDataset
from ucast.inference import DeepEnsemble, evaluate
from ucast.train import Trainer


def make_config(tmp_path, **train_overrides) -> ExperimentConfig:
    data = {
        "name": "test",
        "model": {
            "model_channels": 16,
            "channel_mult": [1, 2],
            "num_blocks": 1,
            "attn_levels": [-1],
            "channels_per_head": 8,
            "dropout": 0.1,
        },
        "data": {
            "statistics": "compute",
            "compute_statistics_samples": 8,
            "batch_size": 4,
            "batch_size_per_device": 2,
            "eval_batch_size": 2,
            "num_workers": 0,
        },
        "optimizer": {"lr": 1e-3, "muon": {"lr": 5e-3}, "schedule": {"warmup_steps": 2}},
        "train": {
            "max_epochs": 1,
            "precision": "fp32",
            "ema_decay": 0.9,
            "output_dir": str(tmp_path / "run"),
            "log_every": 1_000,
            "monitor": "val/avg/rmse/avg",
            **train_overrides,
        },
        "eval": {"rollout_steps": 2, "ensemble_size": 2, "max_batches": 2},
    }
    return ExperimentConfig.from_dict(data)


def make_datasets(rollout_steps: int = 2) -> dict:
    return {
        "train": SyntheticForecastDataset(num_lat=16, num_lon=32, num_times=48, rollout_steps=1, seed=0),
        "val": SyntheticForecastDataset(num_lat=16, num_lon=32, num_times=24, rollout_steps=rollout_steps, seed=1),
    }


def test_deterministic_stage_trains_validates_and_checkpoints(tmp_path):
    trainer = Trainer(make_config(tmp_path), datasets=make_datasets(), device="cpu")
    assert trainer.accumulation == 2  # effective batch 4 = 2 per device x 2
    scores = trainer.fit()

    assert trainer.global_step > 0
    assert scores["val/avg/rmse/avg"] > 0
    assert "val/t12/rmse/2m_temperature" in scores
    assert "val/t24/crps/avg" in scores
    assert (tmp_path / "run" / "last.ckpt").exists()
    assert (tmp_path / "run" / "best.ckpt").exists()
    records = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert any("train/epoch_loss" in record for record in records)


def test_training_reduces_the_loss_on_a_predictable_signal(tmp_path):
    """Synthetic fields are genuinely predictable, so a few dozen steps must beat persistence."""
    config = make_config(tmp_path, max_epochs=3, ema_decay=0.0)
    trainer = Trainer(config, datasets=make_datasets(), device="cpu")
    first = trainer.train_epoch()["train/epoch_loss"]
    trainer.epoch += 1
    trainer.train_epoch()
    trainer.epoch += 1
    last = trainer.train_epoch()["train/epoch_loss"]
    assert last < first


def test_probabilistic_stage_warm_starts_from_stage_one(tmp_path):
    stage1 = Trainer(make_config(tmp_path), datasets=make_datasets(), device="cpu")
    stage1.fit()
    stage1_weights = stage1.model.net.enc_blocks[0][0].conv0.weight.detach().clone()

    config = make_config(
        tmp_path,
        stage="probabilistic",
        init_from=str(tmp_path / "run" / "last.ckpt"),
        output_dir=str(tmp_path / "run2"),
        monitor="val/avg/crps/avg",
    )
    stage2 = Trainer(config, datasets=make_datasets(), device="cpu")
    assert stage2.loss_fn.needs_ensemble  # CRPS
    assert stage2.members == 2
    # Warm start transferred the backbone (the EMA copy, so allow a small difference).
    assert torch.allclose(stage2.model.net.enc_blocks[0][0].conv0.weight, stage1_weights, atol=0.05)
    scores = stage2.fit()
    assert scores["val/avg/crps/avg"] > 0
    assert scores["val/avg/ssr/avg"] > 0  # spread exists, so the ensemble is not degenerate


def test_checkpoint_roundtrip_reproduces_the_forecast(tmp_path):
    trainer = Trainer(make_config(tmp_path), datasets=make_datasets(), device="cpu")
    trainer.fit()
    dataset = trainer.datasets["val"]
    batch = {key: value.unsqueeze(0) for key, value in dataset[0].items()}

    with trainer.ema.average_parameters(trainer.model):
        expected = trainer.model.rollout(batch, steps=2, ensemble_size=1, use_mc_dropout=False)

    model, config, spec = load_forecaster(tmp_path / "run" / "last.ckpt", use_ema=True)
    assert spec.input_variables == dataset.spec.input_variables
    assert config.model.model_channels == 16
    restored = model.rollout(batch, steps=2, ensemble_size=1, use_mc_dropout=False)
    assert torch.allclose(expected, restored, atol=1e-4)


def test_resume_continues_from_the_saved_step(tmp_path):
    trainer = Trainer(make_config(tmp_path), datasets=make_datasets(), device="cpu")
    trainer.fit()
    steps = trainer.global_step

    config = make_config(tmp_path, max_epochs=2, resume_from=str(tmp_path / "run" / "last.ckpt"))
    resumed = Trainer(config, datasets=make_datasets(), device="cpu")
    assert resumed.epoch == 1
    assert resumed.global_step == steps
    resumed.fit()
    assert resumed.global_step > steps


def test_deep_ensemble_combines_checkpoints(tmp_path):
    paths = []
    for seed in (1, 2):
        config = make_config(tmp_path, seed=seed, output_dir=str(tmp_path / f"run{seed}"))
        trainer = Trainer(config, datasets=make_datasets(), device="cpu")
        trainer.fit()
        paths.append(tmp_path / f"run{seed}" / "last.ckpt")

    ensemble, _ = DeepEnsemble.from_checkpoints(paths, device="cpu", members_per_model=3)
    assert ensemble.ensemble_size == 6
    dataset = SyntheticForecastDataset(num_lat=16, num_lon=32, num_times=24, rollout_steps=2, seed=1)
    scores = evaluate(ensemble, dataset, steps=2, batch_size=2, max_batches=2, progress=False)
    assert scores["ensemble_size"] == 6
    assert scores["avg/crps/avg"] > 0
    assert scores["t12/ssr/2m_temperature"] > 0


def test_evaluate_rejects_impossible_lead_times(tmp_path):
    config = make_config(tmp_path)
    dataset = SyntheticForecastDataset(num_lat=8, num_lon=16, num_times=16, rollout_steps=1)
    trainer = Trainer(config, datasets={"train": dataset}, device="cpu")
    ensemble = DeepEnsemble([trainer.model])
    with pytest.raises(ValueError, match="no lead times"):
        evaluate(ensemble, dataset, steps=1, lead_times=[5], progress=False)


def test_checkpoint_is_self_describing(tmp_path):
    config = make_config(tmp_path)
    trainer = Trainer(config, datasets=make_datasets(), device="cpu")
    path = save_checkpoint(
        tmp_path / "bare.ckpt",
        model=trainer.model,
        config=config,
        spec=trainer.spec,
        normalizer=trainer.normalizer,
    )
    # No EMA, no optimizer: still enough to rebuild a working forecaster.
    model, restored_config, spec = load_forecaster(path, use_ema=False)
    assert spec.num_input_channels == trainer.spec.num_input_channels
    assert restored_config.to_dict() == config.to_dict()
    assert torch.allclose(model.normalizer.std, trainer.normalizer.std)


def test_cli_summary_and_train_on_the_smoke_config(tmp_path, capsys, monkeypatch):
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    config = Path(__file__).parent.parent / "configs" / "synthetic_smoke.yaml"
    assert main(["--device", "cpu", "summary", "--config", str(config)]) == 0
    printed = capsys.readouterr().out
    assert "parameters" in printed and "rollout shape" in printed

    assert (
        main(
            [
                "--device",
                "cpu",
                "train",
                "--config",
                str(config),
                "train.max_epochs=1",
                "train.max_steps=2",
                "data.common.num_times=32",
                "data.val.num_times=24",
            ]
        )
        == 0
    )
    assert (tmp_path / "runs" / "smoke" / "last.ckpt").exists()


def test_cli_score_reports_metrics(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = make_config(tmp_path, output_dir=str(tmp_path / "run"))
    # The synthetic builder is the default, so the checkpoint's own config can rebuild the val split.
    config.data.builder = "synthetic"
    config.data.common = {"num_lat": 16, "num_lon": 32, "num_times": 24}
    config.data.val = {"seed": 1}
    Trainer(config, datasets=make_datasets(), device="cpu").fit()

    assert (
        main(
            [
                "--device",
                "cpu",
                "score",
                "--checkpoint",
                str(tmp_path / "run" / "last.ckpt"),
                "--ensemble-size",
                "2",
                "--steps",
                "2",
                "--max-batches",
                "2",
                "--output",
                str(tmp_path / "scores.json"),
            ]
        )
        == 0
    )
    assert "avg/crps/avg" in capsys.readouterr().out
    scores = json.loads((tmp_path / "scores.json").read_text())
    assert scores["ensemble_size"] == 2
