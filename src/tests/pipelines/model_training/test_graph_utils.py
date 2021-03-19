"""
Tests for the `graph_utils` module.
"""


import itertools
from typing import Iterable, Tuple

import numpy as np
import pytest
from faker import Faker

from src.cotton_flower_mot.pipelines.model_training import graph_utils


def _iter_feature_pairs(
    edge_features: np.ndarray,
) -> Iterable[Tuple[np.ndarray]]:
    """
    Iterates through pairs of node features in the edge feature tensor.

    Args:
        edge_features: The edge features.

    Yields:
        Each pair of node features.

    """
    _batch_size, num_left, num_right, _, _ = edge_features.shape

    for b, l, r in itertools.product(
        *map(range, [_batch_size, num_left, num_right])
    ):
        feature_pair = edge_features[b][l][r]
        yield feature_pair[0], feature_pair[1]


@pytest.mark.parametrize(
    ("batch_size", "num_left_nodes", "num_right_nodes", "num_features"),
    [(1, 2, 3, 4), (3, 2, 3, 4), (3, 5, 5, 5), (3, 4, 4, 1)],
    ids=(
        "batch_size_1",
        "different_num_nodes",
        "same_num_nodes",
        "feature_size_1",
    ),
)
def test_compute_bipartite_edge_features(
    faker: Faker,
    batch_size: int,
    num_left_nodes: int,
    num_right_nodes: int,
    num_features: int,
) -> None:
    """
    Tests that `compute_bipartite_edge_features` works.

    Args:
        faker: The fixture to use for generating fake data.
        batch_size: The batch size to use for testing.
        num_left_nodes: The number of nodes on the left side of the graph.
        num_right_nodes: The number of nodes on the right side of the graph.
        num_features: The length of the node feature vectors.

    """
    # Arrange.
    # Create two sets of features.
    left_nodes = faker.tensor((batch_size, num_left_nodes, num_features))
    right_nodes = faker.tensor((batch_size, num_right_nodes, num_features))

    # Act.
    edge_features = graph_utils.compute_bipartite_edge_features(
        left_nodes, right_nodes
    ).numpy()

    # Assert.
    # Make sure the shape is correct.
    assert edge_features.shape == (
        batch_size,
        num_left_nodes,
        num_right_nodes,
        2,
        num_features,
    )

    def _is_feature_pair_present(
        _left_feature: np.ndarray, _right_feature: np.ndarray
    ) -> bool:
        """
        Checks that a particular pair of left and right features are present
        in the edge features.

        Args:
            _left_feature: The left feature to check.
            _right_feature: The right feature to check.

        Returns:
            True if the pair is present, false otherwise.

        """
        for candidate_left, candidate_right in _iter_feature_pairs(
            edge_features
        ):
            if np.allclose(_left_feature, candidate_left) and np.allclose(
                _right_feature, candidate_right
            ):
                return True

        return False

    # Make sure we have all possible pairs of features.
    left_nodes = left_nodes.numpy()
    right_nodes = right_nodes.numpy()

    for b, l, r in itertools.product(
        *map(range, [batch_size, num_left_nodes, num_right_nodes])
    ):
        left_feature = left_nodes[b][l]
        right_feature = right_nodes[b][r]
        assert _is_feature_pair_present(left_feature, right_feature)


@pytest.mark.parametrize(
    ("batch_size", "num_left", "num_right", "num_features"),
    [(3, 4, 5, 3), (3, 4, 5, 1), (1, 4, 5, 3), (3, 4, 4, 3)],
    ids=(
        "different_num_nodes",
        "feature_size_1",
        "batch_size_1",
        "same_num_nodes",
    ),
)
def test_make_affinity_matrix(
    faker: Faker,
    batch_size: int,
    num_left: int,
    num_right: int,
    num_features: int,
) -> None:
    """
    Tests that `make_affinity_matrix` works.

    Args:
        faker: The fixture to use for generating fake data.
        batch_size: The batch size to use for testing.
        num_left: The number of left nodes to use for testing.
        num_right: The number of right nodes to use for testing.
        num_features: The feature length to use for testing.

    """
    # Arrange.
    # Create features to test with.
    edge_features = faker.tensor(
        (batch_size, num_left, num_right, num_features)
    )

    # Act.
    affinity_matrix = graph_utils.make_affinity_matrix(edge_features).numpy()

    # Assert.
    # It should have the correct shape.
    num_nodes = num_left + num_right
    assert affinity_matrix.shape == (
        batch_size,
        num_nodes,
        num_nodes,
        num_features,
    )

    # It should match the input features for the bipartite edges.
    for b, l, r in itertools.product(
        *map(range, [batch_size, num_left, num_right])
    ):
        np.testing.assert_array_almost_equal(
            edge_features[b][l][r], affinity_matrix[b][l][num_left + r]
        )

    # It should have zeros for non bipartite edges.
    zero_feature = np.zeros((num_features,))
    for b, l, r in itertools.product(
        *map(range, [batch_size, num_left, num_right])
    ):
        np.testing.assert_array_almost_equal(
            zero_feature, affinity_matrix[b][num_left + l][r]
        )
