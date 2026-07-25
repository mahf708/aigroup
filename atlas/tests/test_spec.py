import numpy as np
import pytest

from atlas.spec import AtlasConfig, Channel, GridSpec, VariableSet


def test_era5_preset_has_75_channels():
    vs = VariableSet.era5_atlas()
    assert len(vs) == 75
    assert vs.names[0] == "u10m"
    assert vs.names[-1] == "tp"
    assert len(vs.group_indices("z")) == 13


def test_duplicate_channels_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        VariableSet([Channel("a"), Channel("a")])


def test_area_weights_sum_to_one_and_are_symmetric():
    for nlat, poles in [(721, True), (180, False), (49, True)]:
        g = GridSpec.equiangular(nlat, 96, include_poles=poles)
        w = g.area_weights()
        assert w.shape == (nlat, 1)
        assert np.isclose(w.sum(), 1.0)
        assert np.allclose(w[:, 0], w[::-1, 0], atol=1e-12)


def test_area_weights_track_cos_latitude():
    g = GridSpec.equiangular(181, 360, include_poles=True)
    w = g.area_weights()[:, 0]
    # equatorial cells carry far more area than polar ones
    eq = int(np.argmin(np.abs(g.lat)))
    assert w[eq] > 50 * w[0]


def test_coarsen_keeps_endpoints():
    g = GridSpec.equiangular(721, 1440)
    c = g.coarsen(4)
    assert c.shape == (181, 360)
    assert np.isclose(c.lat[0], g.lat[0]) and np.isclose(c.lat[-1], g.lat[-1])


def test_variable_set_roundtrips_through_preset():
    vs = VariableSet.era5_atlas()
    assert vs.to_dict() == {"preset": "era5_atlas"}
    assert VariableSet.from_dict(vs.to_dict()).names == vs.names

    sub = vs.subset(["z500", "t850"])
    d = sub.to_dict()
    assert d["preset"] == "era5_atlas" and d["subset"] == ["z500", "t850"]
    assert VariableSet.from_dict(d).names == ["z500", "t850"]


def test_custom_variable_set_roundtrips_in_full():
    vs = VariableSet([Channel("foo", "bar", level=3.0, level_kind="hybrid")])
    d = vs.to_dict()
    assert "channels" in d
    assert VariableSet.from_dict(d).channels == vs.channels


def test_grid_roundtrip_compact_and_explicit():
    g = GridSpec.equiangular(49, 96, descending=False, include_poles=False)
    d = g.to_dict()
    assert d["nlat"] == 49 and d["descending"] is False and d["poles"] is False
    back = GridSpec.from_dict(d)
    assert np.allclose(back.lat, g.lat) and np.allclose(back.lon, g.lon)

    irregular = GridSpec(lat=np.array([0.0, 10.0, 45.0]), lon=np.array([0.0, 1.0, 5.0]))
    d2 = irregular.to_dict()
    assert "lat" in d2
    assert np.allclose(GridSpec.from_dict(d2).lat, irregular.lat)


def test_e3sm_variable_set_shape():
    vs = VariableSet.e3sm_eam(n_levels=8, prognostic=("T", "Q"), surface=("PS",))
    assert len(vs) == 1 + 2 * 8
    assert vs["T_lev03"].level_kind == "hybrid"
    assert vs["Q_lev00"].positive


def test_config_yaml_roundtrip(tmp_path):
    cfg = AtlasConfig(
        variables=VariableSet.era5_atlas().subset(["z500", "t850"]),
        grid=GridSpec.equiangular(49, 96),
    )
    p = tmp_path / "c.yaml"
    cfg.save(p)
    assert AtlasConfig.load(p).to_dict() == cfg.to_dict()

    p2 = tmp_path / "c.json"
    cfg.save(p2)
    assert AtlasConfig.load(p2).to_dict() == cfg.to_dict()


def test_latent_grid_inherits_topology():
    cfg = AtlasConfig(
        variables=VariableSet.era5_atlas().subset(["z500"]),
        grid=GridSpec.equiangular(721, 1440),
    )
    lg = cfg.latent_grid
    assert lg.shape == (181, 360)
    assert lg.descending_lat == cfg.grid.descending_lat
    assert np.isclose(cfg.compression, 16.0, atol=0.2)
