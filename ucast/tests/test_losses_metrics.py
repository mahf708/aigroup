import numpy as np
import pytest
import torch

from ucast.grid import Grid
from ucast.losses import WeightedCRPS, WeightedMAE, build_loss, crps_ensemble
from ucast.metrics import ForecastMetrics, MetricCollection
from ucast.variables import VariableSet


def brute_force_crps(predictions: np.ndarray, target: float, fair: bool) -> float:
    """Reference implementation of E|X - y| - 1/2 E|X - X'| with explicit loops."""
    members = len(predictions)
    skill = np.mean(np.abs(predictions - target))
    pair_sum = sum(abs(a - b) for a in predictions for b in predictions)
    denominator = members * (members - 1) if fair else members**2
    return skill - 0.5 * pair_sum / denominator


@pytest.mark.parametrize("fair", [True, False])
@pytest.mark.parametrize("members", [2, 3, 5])
def test_crps_matches_a_brute_force_reference(fair, members):
    generator = torch.Generator().manual_seed(members)
    predictions = torch.randn(members, 4, 3, generator=generator, dtype=torch.float64)
    target = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    got = crps_ensemble(predictions, target, fair=fair)
    for i in range(4):
        for j in range(3):
            expected = brute_force_crps(predictions[:, i, j].numpy(), float(target[i, j]), fair=fair)
            assert float(got[i, j]) == pytest.approx(expected)


def test_single_member_crps_is_the_absolute_error():
    predictions = torch.randn(1, 5, 5)
    target = torch.randn(5, 5)
    assert torch.allclose(crps_ensemble(predictions, target), (predictions[0] - target).abs())


def test_fair_crps_is_unbiased_in_the_ensemble_size():
    """The fair estimator's expectation must not depend on M; the biased one drifts with it."""
    generator = torch.Generator().manual_seed(0)
    cases = 20_000
    truth = torch.zeros(cases)
    fair_means, biased_means = [], []
    for members in (2, 8, 32):
        samples = torch.randn(members, cases, generator=generator, dtype=torch.float64)
        fair_means.append(float(crps_ensemble(samples, truth, fair=True).mean()))
        biased_means.append(float(crps_ensemble(samples, truth, fair=False).mean()))
    # Analytic CRPS of a standard normal forecast against a truth of 0:
    # 2 * phi(0) - 1/sqrt(pi) = 2/sqrt(2 pi) - 1/sqrt(pi) ~ 0.2338.
    analytic = 2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi)
    assert fair_means == pytest.approx([analytic] * 3, abs=0.01)
    assert biased_means[0] > fair_means[0] + 0.1  # M=2 biased estimate is clearly too large
    assert biased_means[2] == pytest.approx(analytic, abs=0.03)  # bias vanishes as M grows


def test_crps_ensemble_dim_argument():
    predictions = torch.randn(4, 3, 2)  # ensemble on the last axis
    target = torch.randn(4, 3)
    assert torch.allclose(
        crps_ensemble(predictions, target, ensemble_dim=-1),
        crps_ensemble(predictions.movedim(-1, 0), target),
    )


def test_weighted_losses_apply_channel_weights():
    weights = torch.tensor([1.0, 0.0]).reshape(2, 1, 1)
    loss = WeightedMAE(weights=weights)
    predictions = torch.ones(2, 2, 3, 4)
    targets = torch.zeros(2, 2, 3, 4)
    # Only the first channel is weighted, and the mean is over all elements.
    assert float(loss(predictions, targets)) == pytest.approx(0.5)


def test_crps_loss_requires_an_ensemble_axis():
    loss = build_loss("wcrps")
    assert isinstance(loss, WeightedCRPS)
    assert loss.needs_ensemble
    value = loss(torch.randn(2, 1, 3, 4, 5), torch.randn(1, 3, 4, 5))
    assert value.ndim == 0


def test_unknown_loss_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss("nope")


def test_metrics_reproduce_a_hand_computed_rmse():
    variables = VariableSet(["2m_temperature", "temperature_850"])
    grid = Grid.equiangular(8, 16)
    metrics = ForecastMetrics(variables, grid)
    truth = torch.zeros(3, 2, 8, 16)
    prediction = torch.ones(3, 2, 8, 16)
    prediction[:, 1] = 2.0
    metrics.update(prediction, truth)
    scores = metrics.compute()
    # Area weights have mean 1, so a constant error of e gives rmse == e exactly.
    assert scores["rmse/2m_temperature"] == pytest.approx(1.0)
    assert scores["rmse/temperature_850"] == pytest.approx(2.0)
    assert scores["mae/temperature_850"] == pytest.approx(2.0)
    assert scores["bias/2m_temperature"] == pytest.approx(1.0)
    assert scores["crps/2m_temperature"] == pytest.approx(1.0)  # single member -> absolute error
    assert "ssr/2m_temperature" not in scores  # undefined for one member


def test_spread_skill_ratio_of_a_calibrated_ensemble_is_near_one():
    variables = VariableSet(["2m_temperature"])
    grid = Grid.equiangular(16, 32)
    metrics = ForecastMetrics(variables, grid)
    generator = torch.Generator().manual_seed(0)
    for _ in range(40):
        # Calibrated means the truth is drawn from the same distribution as the members: a shared
        # latent state plus independent unit noise for the observation and for every member.
        latent = torch.randn(2, 1, 16, 32, generator=generator)
        truth = latent + torch.randn(2, 1, 16, 32, generator=generator)
        predictions = latent.unsqueeze(0) + torch.randn(16, 2, 1, 16, 32, generator=generator)
        metrics.update(predictions, truth)
    scores = metrics.compute()
    assert scores["ssr/2m_temperature"] == pytest.approx(1.0, abs=0.05)
    assert scores["num_members"] == 16


def test_metric_collection_averages_over_lead_times():
    variables = VariableSet(["2m_temperature"])
    grid = Grid.equiangular(4, 8)
    collection = MetricCollection(variables, grid, lead_times=[12, 24])
    collection.update(12, torch.ones(1, 1, 4, 8), torch.zeros(1, 1, 4, 8))
    collection.update(24, torch.full((1, 1, 4, 8), 3.0), torch.zeros(1, 1, 4, 8))
    scores = collection.compute(prefix="val/")
    assert scores["val/t12/rmse/2m_temperature"] == pytest.approx(1.0)
    assert scores["val/t24/rmse/2m_temperature"] == pytest.approx(3.0)
    assert scores["val/avg/rmse/2m_temperature"] == pytest.approx(2.0)
    assert scores["val/avg/rmse/avg"] == pytest.approx(2.0)


def test_metrics_reject_mismatched_shapes():
    metrics = ForecastMetrics(VariableSet(["2m_temperature"]), Grid.equiangular(4, 8))
    with pytest.raises(ValueError):
        metrics.update(torch.zeros(2, 1, 2, 4, 8), torch.zeros(1, 2, 4, 8))
