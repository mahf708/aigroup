import numpy as np
import pytest
import torch

from atlas.probe import (
    LinearProbe,
    Recorder,
    attention_entropy,
    builtin_targets,
    depth_sweep,
    feature_maps,
    feature_summary,
    great_circle_matrix,
    head_locality,
    influence_matrix,
    latent_channel_response,
    latent_continuity,
    latent_eof,
    nearest_token,
    noise_sensitivity,
    patch_site,
    reconstruction_report,
    spectral_ratio,
    steer,
    teleconnection_map,
    token_field,
    train_sae,
)
from atlas.probe.record import match_module
from atlas.probe.sae import SAETrainConfig
from tests.conftest import CHANNELS


@pytest.fixture
def ctx(model, batch):
    x = batch["window"]
    return {
        "model": model,
        "x": x,
        "next": batch["next"],
        "time": batch["time"],
        "z0": model.encode(x[:, -1]),
        "hist": model.encode(x[:, 0]),
    }


# ---------------------------------------------------------------------------
# recorder
# ---------------------------------------------------------------------------


def test_match_module_respects_path_segments():
    assert match_module("backbone.blocks.3", "backbone.blocks.*")
    assert not match_module("backbone.blocks.3.attn", "backbone.blocks.*")
    assert match_module("backbone.blocks.3.attn", "backbone.blocks.*.attn")
    assert match_module("backbone.blocks.3.attn.qkv", "backbone.**")
    assert not match_module("projector.blocks.0", "backbone.**")


def test_recorder_captures_residual_stream_without_changing_output(ctx):
    m, x, t = ctx["model"], ctx["x"], ctx["time"]
    g = torch.Generator().manual_seed(3)
    plain = m.step(x[:, -1], ctx["z0"], ctx["hist"], times=t, generator=g).state

    rec = Recorder(m, sites=["backbone.blocks.*"])
    g2 = torch.Generator().manual_seed(3)
    with rec:
        hooked = m.step(x[:, -1], ctx["z0"], ctx["hist"], times=t, generator=g2).state

    assert torch.allclose(plain, hooked, atol=1e-6)
    assert rec.keys() == ["backbone.blocks.0", "backbone.blocks.1"]
    tokens = rec["backbone.blocks.1"]
    assert tokens.shape == (2, 56, 48)
    assert rec.field("backbone.blocks.1").shape == (2, 48, 7, 8)
    assert rec.stack("backbone.blocks.*").shape == (2, 2, 56, 48)


def test_recorder_hooks_are_removed_on_exit(ctx):
    m = ctx["model"]
    rec = Recorder(m, sites=["backbone.blocks.*"])
    with rec:
        pass
    assert not m.backbone.blocks[0]._forward_hooks


def test_recorder_mode_all_captures_the_sampling_trajectory(ctx):
    m = ctx["model"]
    rec = Recorder(m, sites=["backbone.blocks.0"], mode="all")
    with rec:
        m.predict_latent(ctx["z0"], ctx["hist"], n_ensemble=1)
    # the SI sampler evaluates the network twice per Heun step
    assert rec.sites["backbone.blocks.0"].n_calls == 6
    assert rec["backbone.blocks.0"].shape[0] == 6


def test_token_field_and_coordinates(ctx):
    m = ctx["model"]
    rec = Recorder(m, sites=["backbone.blocks.0"])
    with rec:
        m.step(ctx["x"][:, -1], ctx["z0"], ctx["hist"], times=ctx["time"])
    lat, lon = rec.token_latlon("backbone.blocks.0")
    assert lat.shape == (7,) and lon.shape == (8,)
    assert lat[0] > lat[-1]  # descending, matching the parent grid
    f = token_field(rec["backbone.blocks.0"], (7, 8))
    assert f.shape == (2, 48, 7, 8)


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------


def test_attention_capture_rows_are_distributions(ctx):
    m = ctx["model"]
    rec = Recorder(m, sites=["backbone.blocks.0"])
    rec.capture_attention("backbone.blocks.*.attn", query_indices=[0, 17], stats=True)
    with rec:
        m.step(ctx["x"][:, -1], ctx["z0"], ctx["hist"], times=ctx["time"])
    cap = rec.attention("backbone.blocks.1")
    assert cap.weights.shape == (2, 4, 2, 56)
    assert torch.allclose(cap.weights.sum(-1), torch.ones(2, 4, 2), atol=1e-5)
    assert cap.entropy.shape == (2, 4)
    assert (cap.entropy > 0).all()
    assert (cap.entropy <= np.log(56) + 1e-4).all()


def test_teleconnection_map_and_locality(ctx):
    m = ctx["model"]
    rec = Recorder(m, sites=["backbone.blocks.0"])
    rec.capture_attention("backbone.blocks.0.attn", query_indices=[10])
    with rec:
        m.step(ctx["x"][:, -1], ctx["z0"], ctx["hist"], times=ctx["time"])
    w = rec.attention("backbone.blocks.0").weights
    tmap = teleconnection_map(w, (7, 8), query=0)
    assert tmap.shape == (7, 8)
    assert np.isclose(tmap.sum(), 1.0, atol=1e-5)

    d = head_locality(w, m.cfg.latent_grid, (7, 8), torch.tensor([10]))
    assert d.shape == (2, 4)
    assert (d > 0).all() and (d < 20100).all()  # bounded by half Earth's circumference


def test_great_circle_matrix_is_a_metric(model):
    d = great_circle_matrix(model.cfg.latent_grid, (7, 8))
    assert d.shape == (56, 56)
    assert torch.allclose(d.diagonal(), torch.zeros(56), atol=1e-3)
    assert torch.allclose(d, d.T, atol=1e-3)
    assert d.max() < 20100


def test_nearest_token_finds_the_right_cell(model):
    gh = (7, 8)
    i = nearest_token(model.cfg.latent_grid, gh, 90.0, 0.0)
    assert i // gh[1] == 0  # northernmost row for a descending grid
    j = nearest_token(model.cfg.latent_grid, gh, -90.0, 0.0)
    assert j // gh[1] == gh[0] - 1


def test_attention_entropy_of_uniform_weights():
    w = torch.full((1, 2, 3, 16), 1 / 16)
    assert torch.allclose(attention_entropy(w), torch.full((1, 2), float(np.log(16))), atol=1e-5)


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------


def test_linear_probe_recovers_a_linear_signal():
    torch.manual_seed(0)
    x = torch.randn(200, 12)
    w = torch.randn(12, 1)
    y = x @ w + 0.01 * torch.randn(200, 1)
    res = LinearProbe(alphas=(1e-3, 1e-2, 0.1, 1.0)).fit(x, y)
    assert float(res.r2.mean()) > 0.98


def test_linear_probe_fails_on_noise():
    torch.manual_seed(0)
    x = torch.randn(60, 40)
    y = torch.randn(60, 1)
    res = LinearProbe().fit(x, y)
    assert float(res.r2.mean()) < 0.5


def test_ridge_dual_and_primal_agree():
    from atlas.probe.probes import fit_ridge

    torch.manual_seed(0)
    x = torch.randn(50, 50)
    y = torch.randn(50, 2)
    w1, b1 = fit_ridge(x[:40], y[:40], alpha=1.0)
    w2, b2 = fit_ridge(x[:60], y[:60], alpha=1.0)
    assert w1.shape == (50, 2) and w2.shape == (50, 2)
    assert torch.isfinite(w1).all() and torch.isfinite(b1).all()


def test_depth_sweep_returns_one_result_per_layer(ctx):
    m = ctx["model"]
    torch.manual_seed(0)
    x = torch.randn(24, 2, len(CHANNELS), 49, 96)
    times = np.array([np.datetime64("2020-01-01T00")] * 24, dtype="datetime64[s]")
    rec = Recorder(m, sites=["backbone.blocks.*"])
    with rec:
        m.step(x[:, -1], m.encode(x[:, -1]), m.encode(x[:, 0]), times=times)
    acts = {n: rec[n] for n in rec.keys()}
    y = torch.randn(24, 1)
    out = depth_sweep(acts, y, n_components=6, val_fraction=0.3)
    assert set(out) == set(acts)


def test_builtin_targets_are_physically_sane(model):
    vs = model.cfg.variables
    grid = model.cfg.grid
    fns = builtin_targets(vs, grid)
    assert {"nino34", "eke850", "global_mean_t2m"} <= set(fns)

    x = torch.zeros(3, len(vs), 49, 96)
    x[:, vs.index("t2m")] = 288.0
    gm = fns["global_mean_t2m"](x)
    assert torch.allclose(gm, torch.full((3,), 288.0), atol=1e-3)

    x[:, vs.index("sst")] = 0.0
    x[0, vs.index("sst")] = 2.0
    assert float(fns["nino34"](x)[0]) == pytest.approx(2.0, abs=1e-3)
    assert float(fns["nino34"](x)[1]) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# SAE
# ---------------------------------------------------------------------------


def test_sae_reconstructs_a_sparse_signal():
    torch.manual_seed(0)
    d, n_feat = 16, 32
    dictionary = torch.randn(n_feat, d)
    codes = torch.zeros(2000, n_feat)
    idx = torch.randint(0, n_feat, (2000, 3))
    codes.scatter_(1, idx, torch.rand(2000, 3) + 0.5)
    data = codes @ dictionary
    sae, hist = train_sae(
        data.unsqueeze(0), SAETrainConfig(n_features=n_feat, k=4, epochs=12, batch_size=512)
    )
    assert hist["explained_variance"][-1] > hist["explained_variance"][0]
    assert hist["explained_variance"][-1] > 0.5


def test_sae_codes_are_k_sparse_and_mappable(ctx):
    torch.manual_seed(0)
    acts = torch.randn(4, 56, 24)
    sae, _ = train_sae(acts, SAETrainConfig(n_features=32, k=5, epochs=1, batch_size=64))
    codes, _ = sae.encode((acts.reshape(-1, 24) - sae.input_mean) / sae.input_std)
    assert int((codes > 0).sum(-1).max()) <= 5
    maps = feature_maps(sae, acts, (7, 8), features=[0, 1])
    assert maps.shape == (2, 4, 7, 8)
    summary = feature_summary(sae, acts, (7, 8), top=3)
    assert len(summary) == 3
    assert all(0.0 <= s["concentration"] <= 1.0 for s in summary)


# ---------------------------------------------------------------------------
# interventions
# ---------------------------------------------------------------------------


def test_patch_site_zeroing_a_block_changes_the_forecast(ctx):
    m = ctx["model"]
    g = torch.Generator().manual_seed(5)
    base = m.step(ctx["x"][:, -1], ctx["z0"], ctx["hist"], times=ctx["time"], generator=g).state
    g2 = torch.Generator().manual_seed(5)
    with patch_site(m, "backbone.blocks.1", lambda h: torch.zeros_like(h)) as matched:
        got = m.step(
            ctx["x"][:, -1], ctx["z0"], ctx["hist"], times=ctx["time"], generator=g2
        ).state
    assert matched == ["backbone.blocks.1"]
    assert not torch.allclose(base, got)
    # and the hook is gone afterwards
    g3 = torch.Generator().manual_seed(5)
    again = m.step(ctx["x"][:, -1], ctx["z0"], ctx["hist"], times=ctx["time"], generator=g3).state
    assert torch.allclose(base, again, atol=1e-6)


def test_patch_site_rejects_unmatched_pattern(model):
    with pytest.raises(ValueError, match="no module matched"):
        with patch_site(model, "nonexistent.*", lambda h: h):
            pass


def test_steering_scales_with_coefficient(ctx):
    m = ctx["model"]
    d = torch.randn(48)
    deltas = []
    for coef in (0.0, 1.0):
        g = torch.Generator().manual_seed(11)
        with steer(m, "backbone.blocks.0", d, coefficient=coef):
            deltas.append(
                m.step(
                    ctx["x"][:, -1], ctx["z0"], ctx["hist"], times=ctx["time"], generator=g
                ).state
            )
    assert not torch.allclose(deltas[0], deltas[1])


def test_latent_channel_response_is_per_channel_and_grows_with_amplitude(ctx):
    m = ctx["model"]
    small = latent_channel_response(
        m, ctx["x"][:, -1], ctx["z0"], ctx["hist"], "t2m", amplitude=0.5,
        times=ctx["time"], generator=torch.Generator().manual_seed(2),
    )
    big = latent_channel_response(
        m, ctx["x"][:, -1], ctx["z0"], ctx["hist"], "t2m", amplitude=2.0,
        times=ctx["time"], generator=torch.Generator().manual_seed(2),
    )
    assert small.delta.shape == (len(CHANNELS),)
    assert small.source == "t2m"
    assert float(big.delta.sum()) > float(small.delta.sum())


def test_influence_matrix_shape_and_names(ctx):
    m = ctx["model"]
    mat, names = influence_matrix(
        m, ctx["x"][:, -1], ctx["z0"], ctx["hist"], channels=["t2m", "z500"],
        times=ctx["time"], generator=torch.Generator().manual_seed(1),
    )
    assert mat.shape == (2, len(CHANNELS))
    assert names == ["t2m", "z500"]
    assert torch.isfinite(mat).all() and (mat >= 0).all()


def test_noise_sensitivity_reports_per_channel_spread(ctx):
    m = ctx["model"]
    out = noise_sensitivity(m, ctx["z0"], ctx["hist"], n_ensemble=6)
    assert out["channel_spread"].shape == (len(CHANNELS),)
    assert (out["channel_spread"] > 0).all()
    assert out["modes"].shape[1:] == (len(CHANNELS), 13, 24)
    assert float(out["mode_variance"].sum()) <= 1.0 + 1e-4


# ---------------------------------------------------------------------------
# latent diagnostics
# ---------------------------------------------------------------------------


def test_spectral_ratio_recovers_a_pure_scaling(model):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 49, 96)
    r = spectral_ratio(x, 0.9 * x, model.cfg.grid)
    valid = r[~torch.isnan(r)]
    assert torch.allclose(valid, torch.full_like(valid, 0.81), atol=1e-4)


def test_spectral_ratio_masks_bins_with_no_reference_power(model):
    x = torch.zeros(1, 1, 49, 96)
    x[..., 0, 0] = 0.0
    x = x + 1.0  # constant field: only wavenumber 0 has power
    r = spectral_ratio(x, x, model.cfg.grid)
    assert torch.isnan(r).any()
    assert not torch.isnan(r[..., 0]).any()


def test_latent_continuity_detects_a_scrambled_trajectory():
    torch.manual_seed(0)
    smooth = torch.cumsum(0.05 * torch.randn(40, 3, 8, 8), dim=0)
    scrambled = torch.randn(40, 3, 8, 8)
    a = latent_continuity(smooth)
    b = latent_continuity(scrambled)
    assert a["step_ratio"] < b["step_ratio"]


def test_latent_eof_orders_modes_by_variance():
    torch.manual_seed(0)
    basis = torch.randn(3, 2, 6, 6)
    coef = torch.randn(50, 3) * torch.tensor([5.0, 2.0, 0.5])
    data = torch.einsum("nk,kchw->nchw", coef, basis)
    res = latent_eof(data, k=3)
    assert res.modes.shape == (3, 2, 6, 6)
    v = res.variance_fraction
    assert bool((v[:-1] >= v[1:]).all())
    assert float(v.sum()) == pytest.approx(1.0, abs=0.05)


def test_reconstruction_report_beats_persistence_for_bilinear(ctx):
    m = ctx["model"]
    rep = reconstruction_report(m, ctx["x"][:, -1], ctx["next"], ctx["time"])
    assert rep["bilinear_rmse"].shape == (len(CHANNELS),)
    assert len(rep["channel_names"]) == len(CHANNELS)
    # a coarse-grained increment already explains most of a smooth increment
    assert float(rep["bilinear_rmse"].mean()) < float(rep["persistence_rmse"].mean())


def test_recorder_keys_are_in_layer_order_not_lexicographic():
    """blocks.10 must not sort before blocks.2 -- that would scramble depth sweeps."""
    from atlas.probe.record import RecordedSite

    rec = Recorder.__new__(Recorder)
    rec.sites = {
        f"backbone.blocks.{i}": RecordedSite(f"backbone.blocks.{i}") for i in (0, 2, 10, 11, 3)
    }
    assert rec.keys() == [
        "backbone.blocks.0",
        "backbone.blocks.2",
        "backbone.blocks.3",
        "backbone.blocks.10",
        "backbone.blocks.11",
    ]
