"""
Tests for the `sinkhorn` module.
"""


import json

import numpy as np
import pytest
import tensorflow as tf
import yaml
from pytest_snapshot.plugin import Snapshot

from src.cotton_flower_mot.pipelines.model_training import sinkhorn


def test_solve_optimal_transport_obvious() -> None:
    """
    Tests that `solve_optimal_transport` works when there is an obvious
    solution.

    """
    # Arrange.
    # Set up the problem.
    cost = 1.0 - tf.eye(3)
    cost = tf.expand_dims(cost, axis=0)

    # Act.
    transport, dist = sinkhorn.solve_optimal_transport(cost, lamb=100)
    transport = transport.numpy()
    dist = dist.numpy()

    # Assert.
    # It should have solved the problem in the obvious way.
    np.testing.assert_array_almost_equal_nulp(np.eye(3), transport)
    # The sinkhorn distance should be very small.
    assert dist == pytest.approx(0.0)


def test_solve_optimal_transport_entropy() -> None:
    """
    Tests that using different entropy values for
    `solve_optimal_transport` works as we would expect.

    """
    # Arrange.
    # Set up the problem.
    cost = 1.0 - tf.eye(3)
    cost = tf.expand_dims(cost, axis=0)

    # Act.
    transport_good, dist_good = sinkhorn.solve_optimal_transport(
        cost, lamb=100
    )
    transport_good = transport_good.numpy()
    dist_good = dist_good.numpy()

    transport_homogeneous, dist_homogeneous = sinkhorn.solve_optimal_transport(
        cost, lamb=0.1
    )
    transport_homogeneous = transport_homogeneous.numpy()
    dist_homogeneous = dist_homogeneous.numpy()

    # Assert.
    # Lower lambda should lead to a more homogenous solution.
    assert np.std(transport_good) > np.std(transport_homogeneous)
    # It should also have led to a worse solution.
    assert dist_good < dist_homogeneous


def test_solve_optimal_transport_deserts(snapshot: Snapshot) -> None:
    """
    Tests that `solve_optimal_transport` can solve the example "desert problem"
    from here:
    https://michielstock.github.io/posts/2017/2017-11-5-OptimalTransport/

    Args:
        snapshot: The fixture to use for snapshot testing.

    """
    # Arrange.
    # Desert preferences.
    preferences = tf.constant(
        [
            [
                [2.0, 2.0, 1.0, 0.0, 0.0],
                [0.0, -2.0, -2.0, -2.0, 2.0],
                [1.0, 2.0, 2.0, 2.0, -1.0],
                [2.0, 1.0, 0.0, 1.0, -1.0],
                [0.5, 2.0, 2.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, -1.0],
                [-2.0, 2.0, 2.0, 1.0, 1.0],
                [2.0, 1.0, 2.0, 1.0, -1.0],
            ]
        ]
    )
    # Allowed portions. (Row sums)
    portions = tf.constant([[3.0, 3.0, 3.0, 4.0, 2.0, 2.0, 2.0, 1.0]])
    # Amount of each desert. (Column sums)
    desert_amounts = tf.constant([[4.0, 2.0, 6.0, 4.0, 4.0]])

    # Act.
    transport, dist = sinkhorn.solve_optimal_transport(
        -preferences,
        lamb=10,
        row_sums=portions,
        column_sums=desert_amounts,
    )

    # Assert.
    # Convert to a human-readable form for snapshotting.
    results = {
        "transport": transport.numpy().tolist(),
        "dist": dist.numpy().tolist(),
    }
    snapshot.assert_match(yaml.dump(results), "desert_dist.yml")
