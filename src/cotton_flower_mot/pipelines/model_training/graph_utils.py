"""
Selection of utilities for dealing with graphs.
"""


import tensorflow as tf


def compute_bipartite_edge_features(
    left_nodes: tf.Tensor, right_nodes: tf.Tensor
) -> tf.Tensor:
    """
    Computes edge features for a bipartite graph connecting nodes in
    `left_nodes` to nodes in `right_nodes`. Each node in `left_node` will
    be assumed to have an edge to every node in `right_nodes`.

    Args:
        left_nodes: Node features for the left nodes. This should be a 3D tensor
            with shape [batch, n_left_nodes, n_features].
        right_nodes: Node features for the right nodes. This should be a 3D
            tensor with shape [batch, n_right_nodes, n_features].

    Returns:
        A 5D tensor that enumerates every possible combination of a left node
        with a right node. The shape would end up being
        [batch, n_left_nodes, n_right_nodes, 2, n_features].

        The last two dimensions contain the original features from the left node
        and right node.

    """
    left_rank_3 = tf.assert_rank(left_nodes, 3)
    right_rank_3 = tf.assert_rank(right_nodes, 3)

    with tf.control_dependencies([left_rank_3, right_rank_3]):
        # Compute the node dimension index pairs.
        num_left_nodes = tf.shape(left_nodes)[1]
        num_right_nodes = tf.shape(right_nodes)[1]
        left_node_indices = tf.range(0, limit=num_left_nodes)
        right_node_indices = tf.range(0, limit=num_right_nodes)

        index_pair_left, index_pair_right = tf.meshgrid(
            left_node_indices, right_node_indices
        )
        index_pair_left = tf.reshape(index_pair_left, (-1,))
        index_pair_right = tf.reshape(index_pair_right, (-1,))

        # Shapes to use for broadcasting the indices.
        batch_size = tf.shape(left_nodes)[0]
        left_indices_shape = tf.concat(
            [batch_size, tf.shape(index_pair_left)[0]], axis=0
        )
        right_indices_shape = tf.concat(
            [batch_size, tf.shape(index_pair_right)[0]], axis=0
        )

        # Collect the features.
        index_pair_left = tf.broadcast_to(index_pair_left, left_indices_shape)
        index_pair_right = tf.broadcast_to(
            index_pair_right, right_indices_shape
        )
        left_features = tf.gather(
            left_nodes, index_pair_left, axis=1, batch_dims=1
        )
        right_features = tf.gather(
            right_nodes, index_pair_right, axis=1, batch_dims=1
        )

        # Combine into a single tensor.
        split_feature_shape = tf.concat(
            (batch_size, num_left_nodes, num_right_nodes, -1), axis=0
        )
        left_features = tf.reshape(left_features, split_feature_shape)
        right_features = tf.reshape(right_features, split_feature_shape)
        return tf.stack([left_features, right_features], axis=3)


def make_affinity_matrix(edge_features: tf.Tensor) -> tf.Tensor:
    """
    Computes a dense affinity matrix based on provided bipartite edge features.
    Zeros will be used for any edge that does not exist.

    Args:
        edge_features: The edge features. Should have the shape
            [batch_size, n_left_nodes, n_right_nodes, n_features].

    Returns:
        The affinity matrix that it created, which will have the shape
        [batch_size, n_left_nodes + n_right_nodes, n_left_nodes + n_right_nodes,
         n_features].

    """
    features_rank_4 = tf.assert_rank(edge_features, 4)

    with tf.control_dependencies([features_rank_4]):
        edge_feature_shape = tf.shape(edge_features)
        num_left_nodes = edge_feature_shape[1]
        num_right_nodes = edge_feature_shape[2]

        # Pad the unconnected edges with zeros.
        paddings = tf.concat(
            [0, 0, 0, num_right_nodes, num_left_nodes, 0, 0, 0], axis=0
        )
        paddings = tf.reshape(paddings, (4, 2))
        return tf.pad(edge_features, paddings)


def augment_adjacency_matrix(
    *, adjacency_matrix: tf.Tensor, node_features: tf.Tensor
) -> tf.Tensor:
    """
    Creates an "augmented" adjacency matrix where the corresponding features
    for both nodes in each edge are concatenated to that edge feature. For
    instance, the feature at `(n, m)` is going to be

    `concat(affinity_matrix[n, m], node_features[n], node_features[m])`.

    Args:
        adjacency_matrix: The input affinity matrix. Should have the shape
            `[batch_size, n_nodes, n_nodes, n_edge_features]`.
        node_features: The complete node features. Should have the shape
            `[batch_size, n_nodes, n_node_features]`.

    Returns:
        The affinity matrix concatenated with corresponding node features.

    """
    affinity_rank_4 = tf.assert_rank(adjacency_matrix, 4)
    nodes_rank_3 = tf.assert_rank(node_features, 3)

    with tf.control_dependencies([affinity_rank_4, nodes_rank_3]):
        node_features_shape = tf.shape(node_features)
        batch_size = node_features_shape[0]
        num_nodes = node_features_shape[1]
        num_features = node_features_shape[2]

        expanded_feature_shape = tf.concat(
            (batch_size, num_nodes, num_nodes, num_features), axis=0
        )
        node_features_left = tf.broadcast_to(
            node_features, expanded_feature_shape
        )
        node_features_right = tf.transpose(node_features_left, (0, 2, 1, 3))

        # Concatenate on the feature dimension.
        return tf.concat(
            (adjacency_matrix, node_features_left, node_features_right), axis=3
        )
