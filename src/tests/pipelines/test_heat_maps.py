"""
Tests for the `heat_maps` module.
"""


import numpy as np
import pytest

from src.cotton_flower_mot.pipelines import heat_maps


@pytest.mark.parametrize(
    ("points", "expected_map"),
    [
        (np.array([[0.5, 0.8]]), np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])),
        (
            np.array([[0.0, 0.5], [0.8, 0.2]]),
            np.array([[0, 0, 1], [1, 0, 0], [0, 0, 0]]),
        ),
        (np.reshape(np.array([]), (0, 2)), np.zeros((3, 3))),
    ],
    ids=["single_point", "two_points", "no_points"],
)
def test_make_point_annotation_map(
    points: np.ndarray, expected_map: np.ndarray
) -> None:
    """
    Tests that `make_point_annotation_map` works.

    Args:
        points: The sparse points to add to the map.
        expected_map: The expected map.

    """
    # Act.
    got_heat_map = heat_maps.make_point_annotation_map(
        points, map_size=(3, 3)
    ).numpy()

    # Assert.
    # An extra channel dimension will be added.
    got_heat_map = got_heat_map[:, :, 0]
    np.testing.assert_array_equal(expected_map, got_heat_map)


@pytest.mark.parametrize("sigma", [3, 5], ids=["small_sigma", "big_sigma"])
def test_make_heat_map(sigma: int) -> None:
    """
    Tests that `make_heat_map` works.

    Args:
        sigma: The sigma value to use.

    """
    # Arrange.
    # Create some test points.
    points = np.array([[0.3, 0.4], [0.2, 0.1], [0.7, 0.3]])

    # Act.
    got_heat_map = heat_maps.make_heat_map(
        points, map_size=(50, 50), sigma=sigma
    ).numpy()

    # Assert.
    # It should have three points overall.
    assert np.sum(got_heat_map) == pytest.approx(3.0, abs=0.1)
