import pytest
import torch
import yaml

from ucast.config import ExperimentConfig, apply_overrides, deep_merge, load_config
from ucast.nn import UCastUNet
from ucast.optim import HybridOptimizer, Muon, build_optimizer, split_parameters
from ucast.optim.muon import muon_momentum_schedule, orthogonalize_via_newton_schulz
from ucast.optim.schedules import WarmupCosineSchedule


def test_defaults_reproduce_the_paper_settings():
    config = ExperimentConfig()
    assert config.model.model_channels == 320
    assert config.window == 2
    assert config.optimizer.muon.lr == pytest.approx(3.0e-3)
    assert config.train.resolved_loss() == "wmae"
    assert config.train.resolved_ensemble_members() == 1


def test_probabilistic_stage_switches_loss_and_members():
    config = ExperimentConfig.from_dict({"train": {"stage": "probabilistic"}})
    assert config.train.resolved_loss() == "wcrps"
    assert config.train.resolved_ensemble_members() == 2


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="stage"):
        ExperimentConfig.from_dict({"train": {"stage": "magic"}}).train.resolved_loss()


def test_unknown_config_keys_are_rejected_with_a_helpful_message():
    with pytest.raises(ValueError, match="unknown config keys"):
        ExperimentConfig.from_dict({"model": {"model_chanels": 8}})


def test_json_roundtrip_preserves_everything():
    config = ExperimentConfig.from_dict(
        {"model": {"channel_mult": [1, 2, 2]}, "input_variables": ["2m_temperature", {"name": "t", "levels": [1, 2]}]}
    )
    restored = ExperimentConfig.from_json(config.to_json())
    assert restored.to_dict() == config.to_dict()
    assert restored.model.channel_mult == (1, 2, 2)


def test_yaml_inheritance_merges_nested_keys(tmp_path):
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump({"model": {"model_channels": 64, "num_blocks": 2}, "train": {"max_epochs": 100}})
    )
    (tmp_path / "child.yaml").write_text(
        yaml.safe_dump({"base": "base.yaml", "model": {"num_blocks": 4}, "train": {"stage": "probabilistic"}})
    )
    config = load_config(tmp_path / "child.yaml")
    assert config.model.model_channels == 64  # inherited
    assert config.model.num_blocks == 4  # overridden
    assert config.train.max_epochs == 100
    assert config.train.stage == "probabilistic"


def test_circular_inheritance_is_detected(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"base": "b.yaml"}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"base": "a.yaml"}))
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_cli_overrides_are_parsed_as_yaml():
    data = apply_overrides({}, ["model.model_channels=64", "model.channel_mult=[1,2]", "train.compile=true"])
    config = ExperimentConfig.from_dict(data)
    assert config.model.model_channels == 64
    assert config.model.channel_mult == (1, 2)
    assert config.train.compile is True


def test_deep_merge_does_not_mutate_its_inputs():
    base = {"a": {"b": 1}}
    merged = deep_merge(base, {"a": {"c": 2}})
    assert merged == {"a": {"b": 1, "c": 2}}
    assert base == {"a": {"b": 1}}


def test_shipped_configs_load():
    from pathlib import Path

    configs = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))
    assert configs, "no example configs found"
    for path in configs:
        config = load_config(path)
        assert config.name


# ------------------------------------------------------------------ optimizer
def _small_net() -> UCastUNet:
    return UCastUNet(in_channels=4, out_channels=2, spatial_shape=(8, 16), model_channels=8, channel_mult=(1, 2),
                     num_blocks=1, attn_levels=())


def test_parameters_are_split_between_muon_and_adamw():
    net = _small_net()
    muon, decay, no_decay = split_parameters(net, muon_enabled=True, exclude_patterns=("out_conv", "stem"))
    assert all(p.ndim >= 2 for p in muon)
    assert all(p.ndim < 2 for p in no_decay)  # biases and norm gains
    # The stem and the output projection are excluded from Muon by default.
    assert any(p.ndim >= 2 for p in decay)
    assert sum(p.numel() for p in (*muon, *decay, *no_decay)) == sum(p.numel() for p in net.parameters())


def test_muon_disabled_puts_everything_on_adamw():
    net = _small_net()
    muon, decay, no_decay = split_parameters(net, muon_enabled=False)
    assert muon == []
    assert decay and no_decay


def test_build_optimizer_produces_a_hybrid_with_both_learning_rates():
    from ucast.config import OptimizerConfig

    net = _small_net()
    optimizer, schedule = build_optimizer(net, OptimizerConfig(), total_steps=100)
    assert isinstance(optimizer, HybridOptimizer)
    assert any(isinstance(child, Muon) for child in optimizer.optimizers)
    assert sorted(set(schedule.base_lrs)) == pytest.approx([3.0e-4, 3.0e-3])


def test_hybrid_optimizer_steps_all_children_and_roundtrips_state():
    from ucast.config import OptimizerConfig

    torch.manual_seed(0)
    net = _small_net()
    with torch.no_grad():  # the zero-initialised branches would give exactly zero gradients
        for name, parameter in net.named_parameters():
            if name.endswith("conv1.weight") or name.endswith("out_conv.weight"):
                parameter.normal_(0, 0.1)
    optimizer, schedule = build_optimizer(net, OptimizerConfig(), total_steps=10)
    assert schedule.get_last_lr() == [0.0] * len(schedule.base_lrs)  # warmup starts from zero
    schedule.step(10)

    before = {name: p.detach().clone() for name, p in net.named_parameters()}
    net(torch.randn(2, 4, 8, 16)).pow(2).mean().backward()
    optimizer.step()
    changed = {name for name, p in net.named_parameters() if not torch.equal(before[name], p.detach())}
    assert "enc_blocks.0.0.conv0.weight" in changed  # a hidden matrix -> Muon
    assert "out_conv.weight" in changed  # excluded from Muon -> AdamW, decay group
    assert "out_conv.bias" in changed  # 1-D -> AdamW, no-decay group
    assert len(changed) > 0.9 * len(before)
    optimizer.load_state_dict(optimizer.state_dict())


def test_newton_schulz_flattens_the_singular_value_spectrum():
    """Muon replaces the update with its (approximate) orthogonal factor: all singular values ~1."""
    torch.manual_seed(0)
    matrix = torch.randn(32, 16) @ torch.diag(torch.logspace(0, -2, 16))  # deliberately ill-conditioned
    result = orthogonalize_via_newton_schulz(matrix, steps=5).float()
    assert result.shape == matrix.shape
    before = torch.linalg.svdvals(matrix)
    after = torch.linalg.svdvals(result)
    assert (before.max() / before.min()) > 50
    assert (after.max() / after.min()) < 3
    # The iteration deliberately overshoots rather than converging exactly to 1.
    assert float(after.min()) > 0.4 and float(after.max()) < 1.6


def test_muon_rejects_one_dimensional_parameters():
    with pytest.raises(ValueError, match="ndim >= 2"):
        Muon([torch.nn.Parameter(torch.zeros(4))])


def test_muon_momentum_schedule_ramps_and_cools():
    assert muon_momentum_schedule(0, 1000, warmup_steps=100) == pytest.approx(0.85)
    assert muon_momentum_schedule(100, 1000, warmup_steps=100) == pytest.approx(0.95)
    assert muon_momentum_schedule(500, 1000, warmup_steps=100) == pytest.approx(0.95)
    assert muon_momentum_schedule(1000, 1000, warmup_steps=100, cooldown_steps=50) == pytest.approx(0.85)


def test_warmup_cosine_schedule_shape():
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    schedule = WarmupCosineSchedule(optimizer, total_steps=100, warmup_steps=10, min_lr_ratio=0.0)
    assert schedule.multiplier(0) == pytest.approx(0.0)
    assert schedule.multiplier(5) == pytest.approx(0.5)
    assert schedule.multiplier(10) == pytest.approx(1.0)
    assert schedule.multiplier(55) == pytest.approx(0.5, abs=0.02)
    assert schedule.multiplier(100) == pytest.approx(0.0)
    schedule.step(10)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)


def test_schedule_state_roundtrip():
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = torch.optim.SGD([parameter], lr=0.5)
    schedule = WarmupCosineSchedule(optimizer, total_steps=50, warmup_steps=5)
    schedule.step(20)
    state = schedule.state_dict()
    fresh = WarmupCosineSchedule(optimizer, total_steps=50, warmup_steps=5)
    fresh.load_state_dict(state)
    assert fresh.last_step == 20
    assert fresh.get_last_lr() == schedule.get_last_lr()
