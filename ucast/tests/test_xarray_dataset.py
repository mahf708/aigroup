"""Tests for the xarray/zarr adapter, exercised against a miniature WeatherBench-2 style store."""

import numpy as np
import pytest
import torch

xr = pytest.importorskip("xarray")
pytest.importorskip("zarr")

from ucast.data.stats import build_normalizer, normalizer_from_directory  # noqa: E402
from ucast.data.xarray_dataset import XarrayForecastDataset  # noqa: E402
from ucast.variables import VariableSet  # noqa: E402

LEVELS = [500, 850]


def make_store(path, num_times: int = 24, hours: int = 6) -> str:
    """A tiny archive with WeatherBench-2 quirks: (time, level, longitude, latitude) dims,
    descending latitude, and SST that is NaN over land."""
    times = np.datetime64("2020-01-01T00:00:00") + np.arange(num_times) * np.timedelta64(hours, "h")
    latitudes = np.linspace(90, -90, 9)  # descending, as in the real store
    longitudes = np.linspace(0, 360, 12, endpoint=False)
    rng = np.random.default_rng(0)

    def surface(scale=1.0, offset=0.0):
        return offset + scale * rng.standard_normal((num_times, len(longitudes), len(latitudes)))

    sst = surface(2.0, 290.0)
    sst[:, :, :3] = np.nan  # "land" near one pole

    dataset = xr.Dataset(
        {
            "2m_temperature": (("time", "longitude", "latitude"), surface(5.0, 288.0)),
            "sea_surface_temperature": (("time", "longitude", "latitude"), sst),
            "temperature": (
                ("time", "level", "longitude", "latitude"),
                250 + rng.standard_normal((num_times, len(LEVELS), len(longitudes), len(latitudes))),
            ),
            "land_sea_mask": (("longitude", "latitude"), rng.random((len(longitudes), len(latitudes)))),
            "geopotential_at_surface": (
                ("longitude", "latitude"),
                rng.standard_normal((len(longitudes), len(latitudes))),
            ),
        },
        coords={"time": times, "level": LEVELS, "latitude": latitudes, "longitude": longitudes},
    )
    store = str(path / "mini.zarr")
    dataset.to_zarr(store, mode="w")
    return store


def build_dataset(path, **kwargs):
    store = make_store(path)
    defaults = dict(
        path=store,
        variables=["2m_temperature", "sea_surface_temperature", "temperature_850", "temperature_500"],
        window=2,
        rollout_steps=2,
        step_hours=12,
        fill_values={"sea_surface_temperature": "min"},
    )
    defaults.update(kwargs)
    return XarrayForecastDataset(**defaults)


def test_orientation_shapes_and_channel_order(tmp_path):
    dataset = build_dataset(tmp_path)
    spec = dataset.spec
    assert spec.grid.latitudes[0] == -90.0  # reoriented south to north
    assert spec.grid.shape == (9, 12)
    assert spec.input_variables.keys == [
        "2m_temperature",
        "sea_surface_temperature",
        "temperature_850",
        "temperature_500",
    ]
    item = dataset[0]
    assert tuple(item["dynamics"].shape) == (4, 4, 9, 12)  # window 2 + rollout 2 frames
    assert tuple(item["forcings"].shape) == (4, 4, 9, 12)
    assert tuple(item["statics"].shape) == (2, 9, 12)


def test_temporal_subsampling_to_the_requested_step(tmp_path):
    dataset = build_dataset(tmp_path)  # 6-hourly archive, 12-hourly frames
    assert dataset.stride_hours == 12
    times = dataset.initial_times()
    assert (np.diff(times).astype("timedelta64[h]").astype(int) == 12).all()


def test_step_hours_must_divide_the_archive_spacing(tmp_path):
    with pytest.raises(ValueError, match="multiple"):
        build_dataset(tmp_path, step_hours=9)


def test_nan_filling_and_the_error_when_it_is_missing(tmp_path):
    dataset = build_dataset(tmp_path)
    sst = dataset[0]["dynamics"][:, 1]
    assert not torch.isnan(sst).any()
    # 'min' fills with each frame's minimum over valid points, so the 3 x 12 "land" block collapses
    # onto that single coldest value.
    for frame in sst:
        assert int((frame == frame.min()).sum()) >= 3 * 12

    unfilled = build_dataset(tmp_path, fill_values={})
    with pytest.raises(ValueError, match="sea_surface_temperature"):
        unfilled[0]


def test_initial_hours_filter_matches_the_wb2_protocol(tmp_path):
    dataset = build_dataset(tmp_path, initial_hours=[0])
    hours = dataset.initial_times().astype("datetime64[h]").astype(int) % 24
    assert set(hours.tolist()) == {0}
    assert len(dataset) < len(build_dataset(tmp_path))


def test_max_items_subsamples_evenly(tmp_path):
    full = build_dataset(tmp_path)
    limited = build_dataset(tmp_path, max_items=3)
    assert len(limited) == 3
    assert len(full) > 3


def test_time_range_selection(tmp_path):
    dataset = build_dataset(tmp_path, time_range=["2020-01-01", "2020-01-02T23:00"])
    assert dataset.initial_times()[-1] < np.datetime64("2020-01-03")


def test_missing_variable_names_are_reported(tmp_path):
    with pytest.raises(KeyError, match="does_not_exist"):
        build_dataset(tmp_path, variables=["does_not_exist"])


def test_statics_are_standardised(tmp_path):
    dataset = build_dataset(tmp_path)
    statics = dataset[0]["statics"]
    assert float(statics.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(statics[0].std(correction=0)) == pytest.approx(1.0, abs=1e-3)


def test_extra_forcing_fields_are_normalised(tmp_path):
    dataset = build_dataset(
        tmp_path, extra_forcing_fields=["2m_temperature"], variables=["temperature_850"]
    )
    assert dataset.spec.num_forcing_channels == 5  # 4 clock channels + 1 extra
    forcings = dataset[0]["forcings"]
    assert abs(float(forcings[:, 4].mean())) < 3.0  # standardised, not raw kelvin


def test_normalizer_from_statistics_store(tmp_path):
    variables = VariableSet(["2m_temperature", "temperature_850"])
    stats = xr.Dataset(
        {
            "2m_temperature": ((), 288.0),
            "temperature": (("level",), [250.0, 260.0]),
        },
        coords={"level": LEVELS},
    )
    for kind, scale in (("mean", 1.0), ("std", 0.1), ("residual_std", 0.01)):
        (stats * scale).to_zarr(tmp_path / f"era5_{kind}.zarr", mode="w")

    normalizer = normalizer_from_directory(variables, tmp_path)
    assert float(normalizer.mean.flatten()[0]) == pytest.approx(288.0)
    assert float(normalizer.std.flatten()[1]) == pytest.approx(26.0)
    assert normalizer.has_residual_std

    # ... and the same thing through the config-level resolver.
    resolved = build_normalizer(variables, str(tmp_path))
    assert torch.allclose(resolved.mean, normalizer.mean)


def test_identity_statistics_shortcut():
    variables = VariableSet(["a_1"])
    normalizer = build_normalizer(variables, "identity")
    assert float(normalizer.mean) == 0.0 and float(normalizer.std) == 1.0


def test_training_end_to_end_on_the_xarray_dataset(tmp_path):
    """The real data path, wired through the Trainer exactly as a production run would be."""
    from ucast.config import ExperimentConfig
    from ucast.train import Trainer

    store = make_store(tmp_path, num_times=48)
    config = ExperimentConfig.from_dict(
        {
            "model": {"model_channels": 8, "channel_mult": [1, 2], "num_blocks": 1, "attn_levels": []},
            "input_variables": ["2m_temperature", "temperature_850"],
            "data": {
                "builder": "era5",
                "common": {
                    "path": store,
                    "step_hours": 12,
                    "fill_values": {"sea_surface_temperature": "min"},
                },
                "val": {"initial_hours": [0], "max_items": 2},
                "statistics": "compute",
                "compute_statistics_samples": 4,
                "batch_size": 2,
                "batch_size_per_device": 2,
                "eval_batch_size": 1,
                "num_workers": 0,
            },
            "train": {
                "max_epochs": 1,
                "precision": "fp32",
                "output_dir": str(tmp_path / "run"),
                "monitor": "val/avg/rmse/avg",
            },
            "eval": {"rollout_steps": 2, "ensemble_size": 2},
        }
    )
    scores = Trainer(config, device="cpu").fit()
    assert scores["val/avg/rmse/avg"] > 0
    assert (tmp_path / "run" / "last.ckpt").exists()
