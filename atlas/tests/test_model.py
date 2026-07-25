import numpy as np
import pytest
import torch

from atlas.grids import SphericalPad, bilinear_resample
from atlas.layers import MultiheadAttention, PatchEmbed, unpatchify
from atlas.model import Atlas
from atlas.spec import AtlasConfig
from tests.conftest import CHANNELS, make_model

# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


def test_patch_embed_and_unpatchify_roundtrip_shapes():
    emb = PatchEmbed(5, 16, (2, 3), (13, 24))
    assert emb.grid_hw == (7, 8)
    tokens = emb(torch.randn(2, 5, 13, 24))
    assert tokens.shape == (2, 56, 16)
    field = unpatchify(torch.randn(2, 56, 2 * 3 * 5), (7, 8), (2, 3), 5, (13, 24))
    assert field.shape == (2, 5, 13, 24)


def test_spherical_pad_wraps_longitude_and_crosses_poles():
    x = torch.arange(2 * 1 * 4 * 6, dtype=torch.float32).reshape(2, 1, 4, 6)
    pad = SphericalPad(pad_lat=1, pad_lon=1, mode="pole")
    y = pad(x)
    assert y.shape == (2, 1, 6, 8)
    # longitude wraps
    assert torch.allclose(y[..., 1:-1, 0], x[..., -1])
    assert torch.allclose(y[..., 1:-1, -1], x[..., 0])
    # across the pole: mirrored row, shifted half a revolution
    padded_lon = torch.nn.functional.pad(x, (1, 1, 0, 0), mode="circular")
    expect = torch.roll(padded_lon[..., :1, :].flip(-2), shifts=8 // 2, dims=-1)
    assert torch.allclose(y[..., :1, :], expect)


def test_regional_pad_does_not_wrap():
    x = torch.randn(1, 1, 4, 6)
    y = SphericalPad(pad_lat=0, pad_lon=1, periodic_lon=False)(x)
    assert torch.allclose(y[..., 0], x[..., 0])


@pytest.mark.parametrize("mode", ["global", "neighborhood", "window"])
def test_attention_modes_preserve_shape(mode):
    attn = MultiheadAttention(16, 4, mode=mode, grid_hw=(6, 8), kernel=(3, 3))
    x = torch.randn(2, 48, 16)
    assert attn(x).shape == x.shape


def test_neighborhood_attention_is_local():
    """A token's output must not depend on far-away tokens."""
    torch.manual_seed(0)
    attn = MultiheadAttention(8, 2, mode="neighborhood", grid_hw=(6, 8), kernel=(3, 3)).eval()
    x = torch.randn(1, 48, 8)
    base = attn(x)
    x2 = x.clone()
    x2[0, 0] += 100.0  # token (0, 0)
    got = attn(x2)
    changed = (got - base).abs().amax(-1)[0] > 1e-4
    # token (3, 4) = index 28 is far from (0, 0) in every direction
    assert not bool(changed[28])
    assert bool(changed[0])


def test_neighborhood_chunking_is_exact():
    torch.manual_seed(0)
    x = torch.randn(2, 48, 8)
    outs = []
    for chunk in (48, 7, 1):
        a = MultiheadAttention(8, 2, mode="neighborhood", grid_hw=(6, 8), chunk=chunk)
        torch.manual_seed(0)
        for p in a.parameters():
            torch.nn.init.normal_(p, std=0.05)
        outs.append(a(x))
    assert torch.allclose(outs[0], outs[1], atol=1e-5)
    assert torch.allclose(outs[0], outs[2], atol=1e-5)


def test_bilinear_resample_preserves_poles():
    x = torch.randn(1, 2, 721, 1440)
    y = bilinear_resample(x, (181, 360))
    assert y.shape == (1, 2, 181, 360)
    assert torch.allclose(y[..., 0, 0], x[..., 0, 0])
    assert torch.allclose(y[..., -1, 0], x[..., -1, 0])


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def test_encode_and_latent_advance_are_consistent(model, batch):
    x0 = batch["window"][:, -1]
    x1 = batch["next"]
    z0 = model.encode(x0)
    r = model.encode_residual(x1, x0)
    z1 = model.latent_advance(z0, r)
    assert torch.allclose(z1, model.encode(x1), atol=1e-4)


@pytest.mark.parametrize("kind", ["si", "edm", "crps"])
def test_all_estimators_train_and_sample(kind, batch):
    m = make_model(kind, steps=3) if kind != "crps" else make_model(kind)
    loss, logs = m.training_losses(batch["window"], batch["next"], batch["time"])
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in m.backbone.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)

    z0 = m.encode(batch["window"][:, -1])
    h = m.encode(batch["window"][:, 0])
    out = m.step(batch["window"][:, -1], z0, h, times=batch["time"], n_ensemble=3)
    assert out.state.shape == (6, len(CHANNELS), 49, 96)
    assert out.latent_residual.shape == (6, len(CHANNELS), 13, 24)
    assert torch.isfinite(out.state).all()


def test_ensemble_members_differ(model, batch):
    z0 = model.encode(batch["window"][:, -1])
    h = model.encode(batch["window"][:, 0])
    out = model.step(batch["window"][:, -1], z0, h, times=batch["time"], n_ensemble=4)
    members = out.latent_residual.reshape(2, 4, -1)
    spread = members.std(dim=1).mean()
    assert spread > 0


def test_rollout_advances_time_and_keeps_shapes(model, batch):
    outs = list(
        model.rollout(batch["window"], batch["time"], steps=3, n_ensemble=2)
    )
    assert len(outs) == 4  # initial condition + 3 steps
    for o in outs:
        assert o.state.shape == (4, len(CHANNELS), 49, 96)
        assert torch.isfinite(o.state).all()
    dt = outs[2].valid_time[0] - outs[1].valid_time[0]
    assert dt == np.timedelta64(6, "h")


def test_rollout_latent_is_not_re_encoded_from_the_decoded_state(model, batch):
    """The latent memory must be advanced in latent space, not re-encoded."""
    outs = list(model.rollout(batch["window"], batch["time"], steps=2, yield_initial=False))
    last = outs[-1]
    re_encoded = model.encode(last.state)
    assert not torch.allclose(last.latent_state, re_encoded, atol=1e-6)


def test_projector_sees_forcings(model, batch):
    n_forcing = model.forcings.n_channels
    assert n_forcing == 3  # cos zenith + sin/cos day of year
    f = model.forcing_channels(batch["time"], 2, "cpu")
    assert f.shape == (2, 3, 49, 96)
    assert f.abs().max() <= 1.0 + 1e-6


def test_save_and_load_roundtrip(model, batch, tmp_path):
    z0 = model.encode(batch["window"][:, -1])
    h = model.encode(batch["window"][:, 0])
    g = torch.Generator().manual_seed(7)
    before = model.step(batch["window"][:, -1], z0, h, times=batch["time"], generator=g).state

    p = tmp_path / "m.pt"
    model.save(p)
    loaded = Atlas.load(p).eval()
    g2 = torch.Generator().manual_seed(7)
    after = loaded.step(batch["window"][:, -1], z0, h, times=batch["time"], generator=g2).state
    assert torch.allclose(before, after, atol=1e-6)


def test_projector_grid_mismatch_is_rejected():
    from atlas.spec import LatentConfig

    cfg = AtlasConfig.from_dict(make_model("si", steps=2).cfg.to_dict())
    cfg.latent = LatentConfig(shape=(11, 24), history=1)
    with pytest.raises(ValueError, match="token grid"):
        Atlas(cfg)


def test_shipped_configs_build():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "configs"
    cfg = AtlasConfig.load(root / "dev_small.yaml")
    m = Atlas(cfg)
    assert m.parameter_counts()["total"] > 0
    # the large configs are only checked for consistency, not instantiated
    for name in ("era5_atlas_si.yaml", "e3sm_eam_1deg.yaml"):
        c = AtlasConfig.load(root / name)
        assert c.latent_grid.shape == c.latent.shape
        assert c.n_channels == len(c.variables)


# ---------------------------------------------------------------------------
# encoder ablation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["bilinear", "area", "learned"])
def test_encoder_modes_all_produce_a_usable_latent(mode, batch):
    from atlas.spec import LatentConfig
    from tests.conftest import make_config

    cfg = make_config("si", steps=2)
    cfg.latent = LatentConfig(
        shape=(13, 24), mode=mode, history=1, learned_encoder_dim=16, learned_encoder_depth=2
    )
    m = Atlas(cfg).eval()
    n = len(cfg.variables)
    m.normalizers.state.set_stats(np.zeros(n), np.ones(n))
    m.normalizers.residual.set_stats(np.zeros(n), np.ones(n))

    z = m.encode(batch["window"][:, -1])
    assert z.shape == (2, n, 13, 24)
    assert torch.isfinite(z).all()
    loss, _ = m.training_losses(batch["window"], batch["next"], batch["time"])
    assert torch.isfinite(loss)


def test_learned_encoder_starts_at_the_bilinear_baseline(batch):
    """The correction is zero-initialised, so any departure is measurable."""
    from atlas.grids import bilinear_resample
    from atlas.spec import LatentConfig
    from tests.conftest import make_config

    cfg = make_config("si", steps=2)
    cfg.latent = LatentConfig(shape=(13, 24), mode="learned", history=1, learned_encoder_dim=16)
    m = Atlas(cfg).eval()
    n = len(cfg.variables)
    m.normalizers.state.set_stats(np.zeros(n), np.ones(n))
    m.normalizers.residual.set_stats(np.zeros(n), np.ones(n))

    x = batch["window"][:, -1]
    assert torch.allclose(m.encode(x), bilinear_resample(x, (13, 24)), atol=1e-6)


def test_learned_encoder_target_is_detached_from_the_estimator(batch):
    """Encoder gradients must come only from the reconstruction path."""
    from atlas.spec import LatentConfig
    from tests.conftest import make_config

    cfg = make_config("si", steps=2)
    cfg.latent = LatentConfig(shape=(13, 24), mode="learned", history=1, learned_encoder_dim=16)
    m = Atlas(cfg)
    n = len(cfg.variables)
    m.normalizers.state.set_stats(np.zeros(n), np.ones(n))
    m.normalizers.residual.set_stats(np.zeros(n), np.ones(n))
    # the projector's final layer is zero-initialised, which would block every
    # upstream gradient and make this test vacuous
    torch.nn.init.normal_(m.projector.final.linear.weight, std=0.02)

    loss, _ = m.training_losses(
        batch["window"], batch["next"], batch["time"], parts=("latent",)
    )
    loss.backward()
    enc_grads = [p.grad for p in m.resampler.parameters() if p.grad is not None]
    assert all(g.abs().sum() == 0 for g in enc_grads)

    m.zero_grad(set_to_none=True)
    loss2, _ = m.training_losses(
        batch["window"], batch["next"], batch["time"], parts=("projector", "encoder")
    )
    loss2.backward()
    touched = [p.grad for p in m.resampler.parameters() if p.grad is not None]
    assert touched and any(g.abs().sum() > 0 for g in touched)
