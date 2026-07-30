import numpy as np
import pytest
import torch

from ucast.data import SyntheticForecastDataset
from ucast.data.base import DatasetSpec, ForecastDataset
from ucast.data.forcings import clock_forcings
from ucast.grid import Grid
from ucast.normalization import Normalizer, compute_statistics
from ucast.variables import VariableSet


def test_synthetic_dataset_shapes_and_determinism():
    dataset = SyntheticForecastDataset(num_lat=16, num_lon=32, window=2, rollout_steps=3, num_times=40)
    assert len(dataset) == 40 - 5 + 1
    item = dataset[0]
    assert tuple(item["dynamics"].shape) == (5, 4, 16, 32)
    assert tuple(item["forcings"].shape) == (5, 4, 16, 32)
    assert tuple(item["statics"].shape) == (2, 16, 32)
    twin = SyntheticForecastDataset(num_lat=16, num_lon=32, window=2, rollout_steps=3, num_times=40)
    assert torch.equal(item["dynamics"], twin[0]["dynamics"])
    assert not torch.equal(item["dynamics"], SyntheticForecastDataset(num_lat=16, num_lon=32, seed=7)[0]["dynamics"])


def test_synthetic_dataset_is_physically_scaled():
    dataset = SyntheticForecastDataset(num_lat=16, num_lon=32, num_times=32)
    states = dataset[0]["dynamics"]
    pressure = states[:, dataset.spec.input_variables.index("mean_sea_level_pressure")]
    assert 90_000 < float(pressure.mean()) < 110_000
    temperature = states[:, dataset.spec.input_variables.index("2m_temperature")]
    assert 200 < float(temperature.mean()) < 350


def test_dataset_item_validation_catches_a_bad_shape():
    class Broken(ForecastDataset):
        def __init__(self):
            self.spec = DatasetSpec(
                input_variables=VariableSet(["2m_temperature"]),
                output_variables=VariableSet(["2m_temperature"]),
                grid=Grid.equiangular(4, 8),
                window=2,
            )
            self.rollout_steps = 1

        def __len__(self):
            return 1

        def get_item(self, index):
            return {"dynamics": torch.zeros(2, 1, 4, 8)}  # one frame short

    with pytest.raises(ValueError, match="dynamics"):
        Broken()[0]


def test_clock_forcings_are_periodic_and_longitude_dependent():
    longitudes = np.linspace(0, 360, 8, endpoint=False)
    seconds = np.array([0, 86_400, 86_400 * 366], dtype=np.int64)
    forcings = clock_forcings(seconds, num_lat=4, longitudes=longitudes)
    assert forcings.shape == (3, 4, 4, 8)
    # Day progress repeats exactly after 24h ...
    assert forcings[0, 2:] == pytest.approx(forcings[1, 2:], abs=1e-6)
    # ... and varies with longitude, while year progress does not.
    assert forcings[0, 2].std(axis=-1).max() > 0.1
    assert forcings[0, 0].std() == pytest.approx(0.0, abs=1e-7)
    # Year progress differs a year later (366 days versus a 365.24-day year).
    assert abs(float(forcings[0, 0, 0, 0] - forcings[2, 0, 0, 0])) > 1e-3


def test_normalizer_roundtrip_and_residual_scaling():
    variables = VariableSet(["2m_temperature", "mean_sea_level_pressure"])
    normalizer = Normalizer(variables, mean=[288.0, 101_325.0], std=[15.0, 1_000.0], residual_std=[1.5, 200.0])
    states = torch.tensor([288.0, 101_325.0]).reshape(1, 2, 1, 1) + torch.randn(4, 2, 8, 8)
    normalized = normalizer.normalize(states)
    assert torch.allclose(normalizer.denormalize(normalized), states, atol=1e-3)
    increment = torch.randn(4, 2, 8, 8)
    scaled = normalizer.to_residual_scale(increment)
    assert torch.allclose(normalizer.from_residual_scale(scaled), increment, atol=1e-5)
    # Channel 0: std/residual_std = 10, so increments are inflated tenfold.
    assert torch.allclose(scaled[:, 0], increment[:, 0] * 10.0, atol=1e-4)


def test_normalizer_subset_reorders_channels():
    variables = VariableSet(["a_100", "b_100", "c_100"])
    normalizer = Normalizer(variables, mean=[1.0, 2.0, 3.0], std=[1.0, 2.0, 3.0])
    subset = normalizer.subset(VariableSet(["c_100", "a_100"]))
    assert subset.mean.flatten().tolist() == [3.0, 1.0]


def test_normalizer_save_and_load(tmp_path):
    variables = VariableSet(["2m_temperature", "temperature_850"])
    normalizer = Normalizer(variables, mean=[1.0, 2.0], std=[3.0, 4.0], residual_std=[0.5, 0.25])
    path = tmp_path / "stats.npz"
    normalizer.save(path)
    loaded = Normalizer.load(path)
    assert loaded.variables == variables
    assert torch.allclose(loaded.std, normalizer.std)
    assert torch.allclose(loaded.residual_std, normalizer.residual_std)
    assert loaded.has_residual_std


def test_computed_statistics_recover_the_synthetic_scaling():
    dataset = SyntheticForecastDataset(num_lat=16, num_lon=32, num_times=64, noise=0.0)
    normalizer = compute_statistics(dataset, dataset.spec.input_variables, max_samples=32)
    index = dataset.spec.input_variables.index("2m_temperature")
    assert float(normalizer.mean[index]) == pytest.approx(288.0, abs=2.0)
    assert float(normalizer.std[index]) == pytest.approx(15.0, rel=0.3)
    # One-step increments are much smaller than the state's own spread; that gap is exactly what the
    # residual scaling exists to remove.
    assert float(normalizer.residual_std[index]) < float(normalizer.std[index])


def test_normalizer_rejects_wrong_length_statistics():
    with pytest.raises(ValueError, match="channels"):
        Normalizer(VariableSet(["a_1", "b_1"]), mean=[0.0, 1.0, 2.0], std=[1.0, 1.0, 1.0])
