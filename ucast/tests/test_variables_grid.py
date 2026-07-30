import numpy as np
import pytest

from ucast.grid import Grid, equiangular_with_poles, latitude_area_weights
from ucast.variables import LEVEL_WEIGHT_DIVISOR, Variable, VariableSet, era5_variable_set


def test_variable_parsing_distinguishes_levels_from_names():
    assert Variable.parse("temperature_850") == Variable("temperature", 850)
    assert Variable.parse("2m_temperature") == Variable("2m_temperature", None)
    assert Variable.parse("10m_u_component_of_wind").is_surface
    assert Variable.parse({"name": "temperature", "level": 500}).key == "temperature_500"


def test_level_expansion_in_variable_set():
    variables = VariableSet(["2m_temperature", {"name": "temperature", "levels": [850, 500]}])
    assert variables.keys == ["2m_temperature", "temperature_850", "temperature_500"]


def test_era5_variable_set_matches_the_paper():
    variables = era5_variable_set()
    assert len(variables) == 83
    assert len(variables.levels()) == 13
    assert variables.index("geopotential_500") == 5 + 7


def test_duplicate_variables_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        VariableSet(["2m_temperature", "2m_temperature"])


def test_loss_weights_follow_graphcast_conventions():
    variables = VariableSet(["2m_temperature", "10m_u_component_of_wind", "temperature_850", "geopotential_500"])
    weights = variables.loss_weights()
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.1)
    assert weights[2] == pytest.approx(850 / LEVEL_WEIGHT_DIVISOR)
    assert weights[3] == pytest.approx(500 / LEVEL_WEIGHT_DIVISOR)


def test_explicit_weight_overrides_default():
    variables = VariableSet(["2m_temperature"]).with_weights({"2m_temperature": 3.0})
    assert variables.loss_weights()[0] == pytest.approx(3.0)


def test_indices_in_and_subset_roundtrip():
    inputs = era5_variable_set()
    outputs = inputs.difference(VariableSet(["sea_surface_temperature"]))
    assert len(outputs) == 82
    indices = outputs.indices_in(inputs)
    assert [inputs[i].key for i in indices] == outputs.keys


def test_area_weights_sum_to_the_sphere():
    latitudes = equiangular_with_poles(121)
    weights = latitude_area_weights(latitudes)
    assert weights.mean() == pytest.approx(1.0)
    # The unnormalised weights are cell areas: normalised, they must be largest at the equator.
    assert weights.argmax() == 60
    assert weights[0] < weights[60]


def test_area_weights_match_analytic_cell_areas():
    latitudes = equiangular_with_poles(5)  # -90, -45, 0, 45, 90
    weights = latitude_area_weights(latitudes)
    bounds = np.deg2rad([-90.0, -67.5, -22.5, 22.5, 67.5, 90.0])
    expected = np.sin(bounds[1:]) - np.sin(bounds[:-1])
    assert weights == pytest.approx(expected / expected.mean())


def test_grid_orients_latitude_south_to_north():
    grid = Grid.from_arrays(np.linspace(90, -90, 37), np.arange(0, 360, 10))
    assert grid.latitudes[0] == -90.0
    assert grid.shape == (37, 36)


def test_channel_area_weights_broadcast_shape():
    grid = Grid.equiangular(16, 32)
    weights = grid.channel_area_weights(np.array([1.0, 0.1, 2.0]))
    assert tuple(weights.shape) == (3, 16, 1)
    assert float(weights[1].mean()) == pytest.approx(0.1, rel=1e-5)
