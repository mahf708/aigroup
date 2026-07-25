import numpy as np
import pytest
import torch

from atlas.data import SyntheticStore, WindowDataset, collate, resolve_channel
from atlas.forcings import cos_zenith_angle
from atlas.losses import (
    Weights,
    acc,
    ensemble_rmse,
    ensemble_spread,
    fair_crps,
    spatial_rms,
    spread_skill_ratio,
    weighted_mean,
)
from atlas.normalize import NormalizerPair, compute_statistics
from atlas.spec import Channel, GridSpec, VariableSet
from atlas.train import TrainConfig, cosine_cycle_lr

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def test_synthetic_store_is_deterministic_and_smooth():
    vs = VariableSet.era5_atlas().subset(["z500", "t850"])
    grid = GridSpec.equiangular(49, 96)
    a = SyntheticStore(vs, grid, n_times=10, seed=3)
    b = SyntheticStore(vs, grid, n_times=10, seed=3)
    x = a.read(4)
    assert x.shape == (2, 49, 96)
    assert np.allclose(x, b.read(4))
    # neighbouring cells are correlated: the field is not white noise
    assert np.abs(np.diff(x, axis=-1)).mean() < 0.5 * np.abs(x - x.mean()).mean()


def test_window_dataset_shapes_and_times():
    vs = VariableSet.era5_atlas().subset(["z500"])
    store = SyntheticStore(vs, GridSpec.equiangular(25, 48), n_times=20)
    ds = WindowDataset(store, history=1)
    assert len(ds) == 18
    s = ds[3]
    assert s["window"].shape == (2, 1, 25, 48)
    assert s["next"].shape == (1, 25, 48)
    assert s["time"] == store.times[4]

    b = collate([ds[0], ds[1]])
    assert b["window"].shape == (2, 2, 1, 25, 48)
    assert b["time"].dtype == np.dtype("datetime64[s]")


def test_window_dataset_stride_skips_states():
    vs = VariableSet.era5_atlas().subset(["z500"])
    store = SyntheticStore(vs, GridSpec.equiangular(25, 48), n_times=40)
    ds = WindowDataset(store, history=1, stride=2)
    s = ds[0]
    assert np.allclose(s["window"][0].numpy(), store.read(0))
    assert np.allclose(s["window"][1].numpy(), store.read(2))
    assert np.allclose(s["next"].numpy(), store.read(4))


def test_window_dataset_rejects_out_of_range_indices():
    vs = VariableSet.era5_atlas().subset(["z500"])
    store = SyntheticStore(vs, GridSpec.equiangular(25, 48), n_times=10)
    with pytest.raises(ValueError, match="out of range"):
        WindowDataset(store, history=1, indices=[9])


def test_channel_resolution_for_era5_and_e3sm_conventions():
    era5 = resolve_channel(Channel("t850", "t", level=850.0, level_kind="pressure"))
    assert (era5.var, era5.dim, era5.value, era5.by_index) == ("t", "level", 850.0, False)

    e3sm = resolve_channel(Channel("T_lev07", "T", level=7.0, level_kind="hybrid"))
    assert (e3sm.var, e3sm.dim, e3sm.value, e3sm.by_index) == ("T", "lev", 7, True)

    surface = resolve_channel(Channel("PRECT", "surface"))
    assert surface.var == "PRECT" and surface.dim is None


# ---------------------------------------------------------------------------
# forcings
# ---------------------------------------------------------------------------


def test_cos_zenith_angle_tracks_the_seasons():
    lat = np.array([80.0, 0.0, -80.0])
    lon = np.array([0.0])
    june = cos_zenith_angle(np.datetime64("2020-06-21T12:00:00"), lat, lon)[0, :, 0]
    december = cos_zenith_angle(np.datetime64("2020-12-21T12:00:00"), lat, lon)[0, :, 0]
    assert june[0] > 0 > june[2]  # polar day in the north at the June solstice
    assert december[2] > 0 > december[0]
    assert np.all(np.abs(june) <= 1.0)


def test_cos_zenith_angle_has_a_diurnal_cycle():
    lat = np.array([0.0])
    lon = np.array([0.0])
    noon = cos_zenith_angle(np.datetime64("2020-03-20T12:00:00"), lat, lon)[0, 0, 0]
    midnight = cos_zenith_angle(np.datetime64("2020-03-20T00:00:00"), lat, lon)[0, 0, 0]
    assert noon > 0.9
    assert midnight < -0.9


# ---------------------------------------------------------------------------
# losses and metrics
# ---------------------------------------------------------------------------


def test_weighted_mean_of_a_constant_field_is_that_constant():
    grid = GridSpec.equiangular(49, 96)
    w = Weights(torch.from_numpy(grid.area_weights()).squeeze(-1), None)
    x = torch.full((2, 3, 49, 96), 7.0)
    assert torch.allclose(weighted_mean(x, w), torch.full((2,), 7.0), atol=1e-5)
    assert torch.allclose(spatial_rms(x, w.area), torch.full((2, 3), 7.0), atol=1e-5)


def test_area_weighting_downweights_the_poles():
    grid = GridSpec.equiangular(49, 96)
    w = Weights(torch.from_numpy(grid.area_weights()).squeeze(-1), None)
    x = torch.zeros(1, 1, 49, 96)
    x[..., 0, :] = 1.0  # a whole polar row
    unweighted = float(x.mean())
    assert float(weighted_mean(x, w)) < 0.2 * unweighted


def test_fair_crps_is_ensemble_size_independent_in_expectation():
    torch.manual_seed(0)
    truth = torch.zeros(1, 1, 8, 8)
    scores = []
    for m in (4, 32, 128):
        ens = torch.randn(m, 1, 1, 8, 8)
        scores.append(float(fair_crps(ens, truth).mean()))
    assert abs(scores[0] - scores[2]) < 0.1 * scores[2]


def test_crps_rewards_a_sharp_correct_ensemble():
    truth = torch.zeros(1, 1, 4, 4)
    sharp = 0.01 * torch.randn(8, 1, 1, 4, 4)
    broad = 2.0 * torch.randn(8, 1, 1, 4, 4)
    assert float(fair_crps(sharp, truth).mean()) < float(fair_crps(broad, truth).mean())


def test_spread_skill_ratio_is_one_for_a_calibrated_ensemble():
    torch.manual_seed(0)
    # truth and members drawn from the same distribution => exchangeable
    m, n = 200, 4096
    truth = torch.randn(1, 1, 1, n)
    ens = torch.randn(m, 1, 1, 1, n)
    ssr = float(spread_skill_ratio(ens, truth).mean())
    assert 0.9 < ssr < 1.1


def test_ensemble_rmse_and_spread_are_zero_for_a_perfect_deterministic_ensemble():
    truth = torch.randn(1, 2, 4, 4)
    ens = truth.unsqueeze(0).repeat(5, 1, 1, 1, 1)
    assert float(ensemble_rmse(ens, truth)) == pytest.approx(0.0, abs=1e-6)
    assert float(ensemble_spread(ens)) == pytest.approx(0.0, abs=1e-6)


def test_acc_is_one_for_a_perfect_forecast():
    torch.manual_seed(0)
    clim = torch.randn(1, 2, 8, 8)
    truth = clim + torch.randn_like(clim)
    assert float(acc(truth, truth, clim).mean()) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


def test_compute_statistics_matches_numpy():
    rng = np.random.default_rng(0)
    windows = [rng.normal(3.0, 2.0, size=(2, 2, 8, 8)) for _ in range(20)]
    stats = compute_statistics(iter(windows))
    states = np.stack([w[1] for w in windows]).transpose(1, 0, 2, 3).reshape(2, -1)
    assert np.allclose(stats["state_mean"], states.mean(1), atol=1e-8)
    assert np.allclose(stats["state_std"], states.std(1, ddof=1), rtol=1e-6)
    resid = np.stack([w[1] - w[0] for w in windows]).transpose(1, 0, 2, 3).reshape(2, -1)
    assert np.allclose(stats["residual_std"], resid.std(1, ddof=1), rtol=1e-6)


def test_normalizer_roundtrip_and_reordering(tmp_path):
    vs = VariableSet.era5_atlas().subset(["z500", "t850", "q850"])
    n = NormalizerPair(3)
    n.state.set_stats([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    n.residual.set_stats([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    x = torch.randn(2, 3, 4, 4)
    assert torch.allclose(n.state.denormalize(n.state.normalize(x)), x, atol=1e-4)

    p = tmp_path / "s.npz"
    n.save(p, vs)
    reordered = vs.subset(["t850", "z500", "q850"])
    n2 = NormalizerPair(3).load(p, reordered)
    assert float(n2.state.mean.flatten()[0]) == pytest.approx(2.0)
    assert float(n2.state.mean.flatten()[1]) == pytest.approx(1.0)


def test_normalizer_scale_only_ignores_the_mean():
    n = NormalizerPair(2)
    n.residual.set_stats([5.0, 5.0], [2.0, 4.0])
    d = torch.ones(1, 2, 3, 3)
    out = n.residual.scale_only(d)
    assert float(out[0, 0, 0, 0]) == pytest.approx(0.5, abs=1e-4)
    assert float(out[0, 1, 0, 0]) == pytest.approx(0.25, abs=1e-4)


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------


def test_cosine_cycle_schedule_warms_up_and_restarts_lower():
    cfg = TrainConfig(lr=1e-3, warmup_steps=100, cycle_steps=1000, cycles=3, cycle_decay=0.8)
    assert cosine_cycle_lr(0, cfg) < cosine_cycle_lr(50, cfg) < cosine_cycle_lr(99, cfg)
    assert cosine_cycle_lr(100, cfg) == pytest.approx(1e-3, rel=1e-6)
    assert cosine_cycle_lr(999, cfg) < 1e-5
    # each restart peaks at 0.8x the previous peak
    assert cosine_cycle_lr(1000, cfg) == pytest.approx(8e-4, rel=1e-6)
    assert cosine_cycle_lr(2000, cfg) == pytest.approx(6.4e-4, rel=1e-6)


def test_trainer_reduces_the_loss():
    from atlas.train import Trainer, fit_normalizers
    from tests.conftest import make_model

    m = make_model("si", excite=False, steps=2)
    store = SyntheticStore(m.cfg.variables, m.cfg.grid, n_times=48, seed=5)
    ds = WindowDataset(store, history=1)
    fit_normalizers(m, ds, max_samples=16)

    cfg = TrainConfig(
        lr=1e-3, warmup_steps=2, cycle_steps=40, cycles=1, batch_size=4,
        log_every=1000, checkpoint_every=0, out_dir="/tmp/atlas-test-run",
        device="cpu", amp_dtype=None, ema_decay=None,
    )
    tr = Trainer(m, cfg)
    b = collate([ds[i] for i in range(4)])
    first = tr.train_step(b)["loss"]
    for _ in range(30):
        tr.train_step(b)
    last = tr.train_step(b)["loss"]
    assert last < first
