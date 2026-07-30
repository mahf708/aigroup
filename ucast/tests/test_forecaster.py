import pytest
import torch

from ucast.config import ExperimentConfig
from ucast.data import SyntheticForecastDataset
from ucast.forecaster import UCast, build_forecaster, mc_dropout
from ucast.losses import build_loss
from ucast.normalization import compute_statistics


def build(rollout_steps: int = 3, dropout: float = 0.2, **dataset_kwargs) -> tuple[UCast, SyntheticForecastDataset]:
    dataset = SyntheticForecastDataset(
        num_lat=16, num_lon=32, num_times=32, rollout_steps=rollout_steps, **dataset_kwargs
    )
    config = ExperimentConfig.from_dict(
        {
            "model": {
                "model_channels": 16,
                "channel_mult": [1, 2],
                "num_blocks": 1,
                "attn_levels": [-1],
                "channels_per_head": 8,
                "dropout": dropout,
            }
        }
    )
    normalizer = compute_statistics(dataset, dataset.spec.input_variables, max_samples=16)
    return build_forecaster(config, dataset.spec, normalizer), dataset


def batch_of(dataset, size: int = 2) -> dict:
    items = [dataset[i] for i in range(size)]
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def perturb(model: UCast, scale: float = 0.05) -> UCast:
    """Break the zero-initialised residual/output branches so the model is not the identity.

    A freshly built U-Cast is deliberately a pure skip path (its increment is exactly zero), which is
    useful at the start of training but means the inputs cannot influence the output at all.
    """
    with torch.no_grad():
        for name, parameter in model.net.named_parameters():
            if name.endswith("conv1.weight") or name.endswith("out_conv.weight"):
                parameter.normal_(0, scale)
    return model


def test_input_channel_accounting():
    model, dataset = build()
    spec = dataset.spec
    assert spec.num_input_channels == 4 * 2 + 4 + 2  # state window + clock forcings + statics
    assert model.net.in_channels == spec.num_input_channels
    batch = batch_of(dataset)
    composed = model.compose_input(
        model.normalize_states(batch["dynamics"][:, :2]), batch["forcings"][:, 2], batch["statics"]
    )
    assert tuple(composed.shape) == (2, spec.num_input_channels, 16, 32)


def test_an_untrained_model_forecasts_persistence():
    """The zero-initialised output layer means step one is exactly the last input frame."""
    model, dataset = build(dropout=0.0)
    model.eval()
    batch = batch_of(dataset)
    forecast = model.rollout(batch, steps=1, ensemble_size=1, use_mc_dropout=False)
    last_input = batch["dynamics"][:, dataset.spec.window - 1]
    assert torch.allclose(forecast[0, :, 0], last_input, rtol=1e-3, atol=1e-2)


def test_rollout_shapes_and_ensemble_chunking():
    model, dataset = build(rollout_steps=3)
    batch = batch_of(dataset, size=2)
    forecast = model.rollout(batch, steps=3, ensemble_size=4)
    assert tuple(forecast.shape) == (4, 2, 3, 4, 16, 32)
    chunked = model.rollout(batch, steps=3, ensemble_size=4, members_per_forward=2)
    assert chunked.shape == forecast.shape


def test_mc_dropout_is_the_only_source_of_spread():
    model, dataset = build(dropout=0.3)
    perturb(model)
    batch = batch_of(dataset)
    stochastic = model.rollout(batch, steps=2, ensemble_size=4, use_mc_dropout=True)
    assert float(stochastic.std(dim=0).mean()) > 0
    deterministic = model.rollout(batch, steps=2, ensemble_size=4, use_mc_dropout=False)
    assert float(deterministic.std(dim=0).mean()) == pytest.approx(0.0, abs=1e-8)


def test_mc_dropout_context_leaves_norm_layers_in_eval_mode():
    model, _ = build(dropout=0.1)
    model.eval()
    with mc_dropout(model):
        modes = {type(m).__name__: m.training for m in model.modules()}
        assert modes["Dropout"] is True
        assert modes["GroupNorm"] is False
    assert not any(m.training for m in model.modules())


def test_residual_target_is_the_scaled_increment():
    model, dataset = build()
    batch = batch_of(dataset)
    states = model.normalize_states(batch["dynamics"])
    target = model.target_from_states(states[:, :2], states[:, 2])
    expected = model.normalizer.to_residual_scale(states[:, 2] - states[:, 1])
    assert torch.allclose(target, expected, atol=1e-6)
    # And the inverse mapping reconstructs the next state from the network's output space.
    reconstructed = model.prediction_from_output(target, states[:, :2])
    assert torch.allclose(reconstructed, states[:, 2], atol=1e-4)


def test_absolute_prediction_mode():
    model, dataset = build()
    model.predict_residual = False
    batch = batch_of(dataset)
    states = model.normalize_states(batch["dynamics"])
    target = model.target_from_states(states[:, :2], states[:, 2])
    assert torch.allclose(target, states[:, 2])


def test_deterministic_and_probabilistic_losses_run_and_backpropagate():
    model, dataset = build()
    batch = batch_of(dataset)
    weights = dataset.spec.grid.channel_area_weights(dataset.spec.output_variables.loss_weights())

    mae = build_loss("wmae", weights=weights)
    loss, _ = model.compute_loss(batch, mae, members=1)
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in model.parameters())

    model.zero_grad()
    crps = build_loss("wcrps", weights=weights)
    loss, info = model.compute_loss(batch, crps, members=2)
    loss.backward()
    assert torch.isfinite(loss) and info["members"] == 2


def test_crps_loss_rejects_a_single_member():
    model, dataset = build()
    with pytest.raises(ValueError, match="at least 2"):
        model.compute_loss(batch_of(dataset), build_loss("wcrps"), members=1)


def test_ensemble_forward_reshapes_members_correctly():
    model, dataset = build(dropout=0.0)
    model.eval()
    batch = batch_of(dataset, size=3)
    states = model.normalize_states(batch["dynamics"][:, :2])
    out = model.ensemble_forward(states, batch["forcings"][:, 2], batch["statics"], members=5)
    assert tuple(out.shape) == (5, 3, 4, 16, 32)
    # With dropout off every member must be identical.
    assert torch.allclose(out[0], out[4])


def test_prescribed_channels_are_not_forecast_but_are_fed_back():
    """A variable can be an input without being predicted -- prescribed SST, for instance."""
    dataset = SyntheticForecastDataset(
        num_lat=16,
        num_lon=32,
        num_times=32,
        rollout_steps=3,
        variables=["2m_temperature", "sea_surface_temperature", "temperature_850"],
        output_variables=["2m_temperature", "temperature_850"],
    )
    config = ExperimentConfig.from_dict(
        {"model": {"model_channels": 8, "channel_mult": [1, 2], "num_blocks": 1, "attn_levels": [], "dropout": 0.0}}
    )
    normalizer = compute_statistics(dataset, dataset.spec.input_variables, max_samples=8)
    model = perturb(build_forecaster(config, dataset.spec, normalizer))
    assert model.output_indices.tolist() == [0, 2]
    assert model.prescribed_indices.tolist() == [1]

    batch = batch_of(dataset)
    forecast = model.rollout(batch, steps=3, ensemble_size=1, use_mc_dropout=False)
    assert tuple(forecast.shape) == (1, 2, 3, 2, 16, 32)

    # Under "prescribed" the non-forecast channel is refreshed from truth at every step; under
    # "persist" it is frozen at its last known value. The two must give different forecasts.
    model.prescribed_policy = "persist"
    persisted = model.rollout(batch, steps=3, ensemble_size=1, use_mc_dropout=False)
    assert not torch.allclose(forecast, persisted, atol=1e-3)


def test_output_variables_must_be_a_subset_of_the_inputs():
    dataset = SyntheticForecastDataset(num_lat=8, num_lon=16, num_times=16, variables=["2m_temperature"])
    spec = dataset.spec.replace(output_variables=type(dataset.spec.output_variables)(["mean_sea_level_pressure"]))
    normalizer = compute_statistics(dataset, dataset.spec.input_variables, max_samples=4)
    with pytest.raises(ValueError, match="subset"):
        UCast(net=torch.nn.Identity(), normalizer=normalizer, spec=spec)
