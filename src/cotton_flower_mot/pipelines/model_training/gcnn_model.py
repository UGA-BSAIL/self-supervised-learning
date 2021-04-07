"""
Implements a model inspired by GCNNTrack.
https://arxiv.org/pdf/2010.00067.pdf
"""

from typing import Any, Callable, Tuple

import spektral
import tensorflow as tf
from loguru import logger
from tensorflow.keras import layers

from ..config import ModelConfig
from ..schemas import ModelInputs, ModelTargets
from ..sinkhorn import solve_optimal_transport
from .graph_utils import (
    augment_adjacency_matrix,
    compute_bipartite_edge_features,
    gcn_filter,
    make_adjacency_matrix,
    normalize_adjacency,
)
from .layers.dense import DenseBlock, TransitionLayer
from .similarity_utils import compute_ious, cosine_similarity


def _bn_relu_conv(
    *args: Any, **kwargs: Any
) -> Callable[[tf.Tensor], tf.Tensor]:
    """
    Small helper function that builds a bn-relu-conv block.

    Args:
        *args: Forwarded to `Conv2D()`.
        **kwargs: Forwarded to `Conv2D()`.

    Returns:
        The block, which can be called to apply it to some input, similar to
        a Keras layer.

    """
    conv = layers.Conv2D(*args, **kwargs)
    norm = layers.BatchNormalization()
    relu = layers.Activation("relu")

    def _apply_block(block_input: tf.Tensor) -> tf.Tensor:
        return conv(relu(norm(block_input)))

    return _apply_block


def _bn_relu_dense(
    *args: Any, **kwargs: Any
) -> Callable[[tf.Tensor], tf.Tensor]:
    """
    Small helper function that builds a bn-relu-dense block.

    Args:
        *args: Forwarded to `Dense()`.
        **kwargs: Forwarded to `Dense()`.

    Returns:
        The block, which can be called to apply it to some input, similar to
        a Keras layer.

    """
    dense = layers.Dense(*args, **kwargs)
    norm = layers.BatchNormalization()
    relu = layers.Activation("relu")

    def _apply_block(block_input: tf.Tensor) -> tf.Tensor:
        return dense(relu(norm(block_input)))

    return _apply_block


def _build_appearance_feature_extractor(
    normalized_input: tf.Tensor, *, config: ModelConfig
) -> tf.Tensor:
    """
    Builds a CNN for extracting appearance features from detection images.

    Args:
        normalized_input: The normalized input detections.
        config: Model configuration.

    Returns:
        A batch of corresponding appearance features.

    """
    logger.debug(
        "Appearance features will have length {}.",
        config.num_appearance_features,
    )

    # Input convolution layers.
    conv1_1 = _bn_relu_conv(48, 3, padding="same")(normalized_input)
    conv1_2 = _bn_relu_conv(48, 3, padding="same")(conv1_1)
    pool1 = layers.MaxPool2D()(conv1_2)

    # Dense blocks.
    dense1 = DenseBlock(3, growth_rate=4)(pool1)
    transition1 = TransitionLayer()(dense1)

    dense2 = DenseBlock(6, growth_rate=4)(transition1)
    transition2 = TransitionLayer()(dense2)

    dense3 = DenseBlock(12, growth_rate=4)(transition2)
    transition3 = TransitionLayer()(dense3)

    dense4 = DenseBlock(8, growth_rate=4)(transition3)

    # Fully-connected layer to generate feature vector.
    flat5 = layers.Flatten()(dense4)
    return _bn_relu_dense(config.num_appearance_features)(flat5)


def _build_appearance_model(*, config: ModelConfig) -> tf.keras.Model:
    """
    Creates a sub-model that extracts appearance features.

    Args:
        config: Model configuration.

    Returns:
        The model that it created.

    """
    logger.debug(
        "Using input shape {} for appearance feature extractor.",
        config.image_input_shape,
    )
    images = layers.Input(shape=config.image_input_shape)

    def _normalize(_images: tf.Tensor) -> tf.Tensor:
        # Normalize the images before putting them through the model.
        float_images = tf.cast(_images, tf.keras.backend.floatx())
        return tf.image.per_image_standardization(float_images)

    normalized = layers.Lambda(_normalize)(images)

    # Apply the model layers.
    features = _build_appearance_feature_extractor(normalized, config=config)

    # Create the model.
    return tf.keras.Model(inputs=images, outputs=features)


def _build_edge_mlp(
    *,
    geometric_features: Tuple[tf.Tensor, tf.Tensor],
    appearance_features: Tuple[tf.Tensor, tf.Tensor],
) -> tf.Tensor:
    """
    Builds the MLP that computes edge features.

    Args:
        geometric_features: Batch of geometric features for both the
            detections and tracklets, in that order. Should have the
            shape `[batch_size, n_nodes, n_features]`.
        appearance_features: Batch of appearance features for both the
            detections and tracklets, in that order. Should have the shape
            `[batch_size, n_nodes, n_features]`.

    Returns:
        The computed edge features. It will be a tensor of shape
        `[batch_size, n_left_nodes, n_right_nodes, 1]`.

    """
    right_features, left_features = geometric_features
    num_right_nodes = tf.shape(right_features)[1]
    num_left_nodes = tf.shape(left_features)[1]

    def _combine_input_impl(
        features: Tuple[tf.Tensor, tf.Tensor]
    ) -> tf.Tensor:
        detections, tracklets = features
        input_edge_features = compute_bipartite_edge_features(
            left_nodes=tracklets, right_nodes=detections
        )
        # We should know this statically.
        num_features = detections.shape[-1]
        # We need to get rid of the node dimension so we can
        # feed it into an MLP. We also concatenate the detection and tracklet
        # features at the same time.
        return tf.ensure_shape(
            tf.reshape(input_edge_features, (-1, num_features * 2)),
            (None, num_features * 2),
        )

    geometric_combined = layers.Lambda(
        _combine_input_impl,
    )(geometric_features)
    appearance_combined = layers.Lambda(
        _combine_input_impl,
    )(appearance_features)
    all_features = layers.Concatenate()(
        (geometric_combined, appearance_combined)
    )

    # Apply the MLP. We need to use a feature size of one for the output,
    # since these values are going directly in the affinity matrix.
    edge_features = _bn_relu_dense(1)(all_features)

    # Transform back to the expanded shape.
    return tf.reshape(edge_features, (-1, num_left_nodes, num_right_nodes, 1))


def _compute_pairwise_similarities(
    similarity_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    *,
    detection_features: tf.Tensor,
    tracklet_features: tf.Tensor,
) -> tf.Tensor:
    """
    Computes some similarity metric between every possible combination of
    detection and tracklet features.

    Args:
        similarity_function: A function that computes a similarity metric
            between two batches of feature vectors. Should return a batch
            of scalar similarity values.
        detection_features: The padded detection features,
            with shape `[batch_size, max_n_detections, n_features]`.
        tracklet_features: The padded tracklet features,
            with shape `[batch_size, max_n_tracklets, n_features]`.

    Returns:
        The similarities computed between every combination of detections and
        tracklets. Will have shape `[batch_size, max_n_tracklets,
        max_n_detections]`.

    """
    # Compute all possible combinations.
    combinations = compute_bipartite_edge_features(
        tracklet_features, detection_features
    )

    combinations_shape = tf.shape(combinations)
    combinations_outer_shape = combinations_shape[:3]
    num_features = combinations_shape[-1]

    # Reshape so we can compute IOUs in one pass.
    flat_shape = tf.stack((-1, 2, num_features), axis=0)
    combinations = tf.reshape(combinations, flat_shape)
    tracklet_boxes = combinations[:, 0, :]
    detection_boxes = combinations[:, 1, :]

    # Compute similarities.
    similarities = similarity_function(tracklet_boxes, detection_boxes)

    # Add the other dimensions back.
    return tf.reshape(similarities, combinations_outer_shape)


def _build_affinity_mlp(
    *,
    detection_geom_features: tf.Tensor,
    tracklet_geom_features: tf.Tensor,
    detection_inter_features: tf.Tensor,
    tracklet_inter_features: tf.Tensor,
) -> tf.Tensor:
    """
    Builds the MLP that computes the affinity score between two nodes.

    Args:
        detection_geom_features: The padded detection geometry features,
            with shape `[batch_size, max_n_detections, 4]`.
        tracklet_geom_features: The padded tracklet geometry features,
            with shape `[batch_size, max_n_tracklets, 4]`.
        detection_inter_features: The padded detection interaction features,
            with shape `[batch_size, max_n_detections, 4]`.
        tracklet_inter_features: The padded tracklet interaction features,
            with shape `[batch_size, max_n_tracklets, 4]`.

    Returns:
        The final affinity scores between each pair of tracklet and detections.
        Will have a shape of `[batch_size, max_n_tracklets, max_n_detections]`.

    """
    # Compute IOUs and cosine similarity.
    iou = layers.Lambda(
        lambda f: _compute_pairwise_similarities(
            compute_ious, tracklet_features=f[0], detection_features=f[1]
        ),
        name="iou",
    )((tracklet_geom_features, detection_geom_features))
    cosine = layers.Lambda(
        lambda f: _compute_pairwise_similarities(
            cosine_similarity, tracklet_features=f[0], detection_features=f[1]
        ),
        name="cosine_similarity",
    )((tracklet_inter_features, detection_inter_features))

    # Concatenate into our input.
    similarity_input = tf.stack((iou, cosine), axis=-1)
    # Make sure the channels dimension is defined statically so Keras layers
    # work.
    similarity_input = tf.ensure_shape(similarity_input, (None, None, None, 2))

    # Apply the MLP. 1x1 convolution is an efficient way to apply the same MLP
    # to every detection/tracklet pair.
    conv1_1 = layers.Conv2D(1, 1, name="affinity_conv")(similarity_input)

    # Remove the extraneous 1 dimension.
    return conv1_1[:, :, :, 0]


def _update_adjacency_matrix() -> Callable[
    [Tuple[tf.Tensor, tf.Tensor]], tf.Tensor
]:
    """
    Updates the adjacency matrix for the next layer according to the method
    specified in https://arxiv.org/pdf/2010.00067.pdf.

    Returns:
        A function returning the adjacency matrix for the next layer.

    """
    # Combine the affinity matrix with node features.
    aug1_1 = layers.Lambda(
        lambda x: augment_adjacency_matrix(
            adjacency_matrix=tf.expand_dims(x[0], axis=-1), node_features=x[1]
        )
    )
    # The MLP operation over all edges is actually implemented as a 1x1
    # convolution for convenience.
    conv1_2 = _bn_relu_conv(1, 1, padding="same")
    # Normalization operation for the matrix.
    norm1_3 = layers.Lambda(
        lambda x: normalize_adjacency(x), name="normalize_adjacency"
    )

    def _apply_block(inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Args:
            inputs: Inputs for this operation, including:
            - node_features: The node features from the previous layer. Should
                have the shape `[batch_size, n_nodes, n_features]`.
            - adjacency_matrix: The adjacency matrix from the previous layer.
                Should have the shape `[batch_size, n_nodes, n_nodes]`.

        Returns:
            The adjacency matrix for the next layer.

        """
        node_features, adjacency_matrix = inputs
        adjacency = norm1_3(conv1_2(aug1_1((adjacency_matrix, node_features))))
        # Remove the final dimension, which should be one.
        return adjacency[:, :, :, 0]

    return _apply_block


def _gcn_block(
    *args: Any, **kwargs: Any
) -> Callable[[Tuple[tf.Tensor, tf.Tensor]], Tuple[tf.Tensor, tf.Tensor]]:
    """
    Creates a new GCN layer with a corresponding affinity matrix update,
    in accordance with the method described in
    https://arxiv.org/pdf/2010.00067.pdf

    Args:
        *args: Will be forwarded to the `GCNConv` layer.
        **kwargs: Will be forwarded to the `GCNConv` layer.

    Returns:
        A function taking the same inputs as `GCNConv`. As output, it returns
        both the new node features and the new adjacency matrix.

    """
    # Compute the next node features.
    laplacian1_1 = layers.Lambda(gcn_filter)
    gcn1_1 = spektral.layers.GCNConv(*args, **kwargs)

    # Compute the next edge features.
    edges1_1 = _update_adjacency_matrix()

    def _apply_block(
        inputs: Tuple[tf.Tensor, tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            inputs: Inputs for this operation, including:
            - node_features: The node features from the previous layer. Should
                have the shape `[batch_size, n_nodes, n_features]`.
            - adjacency_matrix: The adjacency matrix from the previous layer.
                Should have the shape `[batch_size, n_nodes, n_nodes, 1]`.

        Returns:
            The new node features and the new adjacency matrix.

        """
        node_features, adjacency_matrix = inputs
        new_nodes = gcn1_1((node_features, laplacian1_1(adjacency_matrix)))
        return new_nodes, edges1_1((new_nodes, adjacency_matrix))

    return _apply_block


def _build_gnn(
    *,
    adjacency_matrix: tf.Tensor,
    node_features: tf.Tensor,
    config: ModelConfig,
) -> tf.Tensor:
    """
    Builds the GNN for performing feature association.

    Args:
        adjacency_matrix: The initial affinity matrix. Should have the shape
            `[batch_size, n_nodes, n_nodes, 1]`.
        node_features: The input node features. Should have the shape
            `[batch_size, n_nodes, n_features]`.
        config: The model configuration.

    Returns:
        The output node features from the GNN, which will have the shape
        `[batch_size, n_nodes, n_gcn_channels]`.

    """
    # Remove the final dimension from the adjacency matrix, since it's just 1.
    adjacency_matrix = adjacency_matrix[:, :, :, 0]

    gcn1_1 = _gcn_block(config.num_gcn_channels)(
        (node_features, adjacency_matrix)
    )
    gcn1_2 = _gcn_block(config.num_gcn_channels)(gcn1_1)

    nodes, _ = gcn1_2
    return nodes


def _compute_row_or_column_sum(length: tf.Tensor) -> tf.Tensor:
    """
    For Sinkhorn normalization, we expect rows and columns to sum to one,
    except for the births/deaths row/column, which we expect to sum to their
    own length.

    Args:
        length: The length of the row or column.

    Returns:
        The expected sums. This will have the form `[1, 1, ..., length]`.

    """
    row_shape = tf.expand_dims(length, axis=0)
    sums = tf.ones(row_shape, dtype=tf.float32)

    # Add the last element.
    return tf.concat((sums, tf.cast(row_shape, tf.float32)), axis=0)


def _solve_association(
    *,
    affinity_scores: tf.Tensor,
    num_detections: tf.Tensor,
    num_tracklets: tf.Tensor,
    config: ModelConfig,
) -> tf.RaggedTensor:
    """
    Solves the association problem using Sinkhorn normalization.

    Args:
        affinity_scores: The affinity scores computed for each possible
            tracklet/detection pair. Should have a shape of
            `[batch_size, max_n_tracklets, max_n_detections]`.
        num_detections: The number of detections in each example. Should be
            a vector of shape `[batch_size]`.
        num_tracklets: The number of tracklets in each example. Should be
            a vector of shape `[batch_size]`.
        config: The model configuration to use.

    Returns:
        The normalized optimal transport matrix. This will be a `RaggedTensor`
        where the second dimension is ragged, so it will have the shape
        `[batch_size, n_tracklets * n_detections]`.

    """

    def _normalize(
        element: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    ) -> tf.Tensor:
        affinity_matrix, _num_detections, _num_tracklets = element

        # Remove the padding.
        affinity_un_padded = affinity_matrix[:_num_tracklets, :_num_detections]
        # Add additional row and column for track births/deaths.
        affinity_expanded = tf.pad(affinity_un_padded, [[0, 1], [0, 1]])

        row_sums = _compute_row_or_column_sum(_num_tracklets)
        column_sums = _compute_row_or_column_sum(_num_detections)

        # Add fake batch dimension.
        affinity_expanded = tf.expand_dims(affinity_expanded, axis=0)
        row_sums = tf.expand_dims(row_sums, axis=0)
        column_sums = tf.expand_dims(column_sums, axis=0)

        # Normalize it.
        transport, _ = solve_optimal_transport(
            # Cost matrix is -affinity.
            -affinity_expanded,
            lamb=config.sinkhorn_lambda,
            row_sums=row_sums,
            column_sums=column_sums,
        )

        # Remove births/deaths row/column.
        transport = transport[:, :-1, :-1]
        # Flatten and remove fake batch dimension.
        return tf.reshape(transport, (-1,))

    # Unfortunately, we can't have padding for the affinity scores, because
    # it affects the optimization. Therefore, this process has to be done
    # with map_fn instead of vectorized.
    return layers.Lambda(
        lambda f: tf.map_fn(
            _normalize,
            f,
            fn_output_signature=tf.RaggedTensorSpec(
                shape=[None], dtype=tf.float32
            ),
        ),
        name="sinkhorn",
    )((affinity_scores, num_detections, num_tracklets))


def extract_appearance_features(
    *,
    detections: tf.RaggedTensor,
    tracklets: tf.RaggedTensor,
    config: ModelConfig,
) -> Tuple[tf.RaggedTensor, tf.RaggedTensor]:
    """
    Builds the portion of the system that extracts appearance features.

    Args:
        detections: Extracted detection images. Should have the shape
            `[batch_size, n_detections, height, width, channels]`, where the
            second dimension is ragged.
        tracklets: Extracted final images from each tracklet. Should have the
            shape `[batch_size, n_tracklets, height, width, channels]`, where
            the second dimension is ragged.
        config: The model configuration.

    Returns:
        The extracted appearance features, for both the detections and
        tracklets. Each set will have the shape
        `[batch_size, n_nodes, n_features]`, where the second dimension
        is ragged.

    """
    # Convert detections and tracklets to a normal batch for appearance
    # feature extraction.
    merge_dims = layers.Lambda(lambda rt: rt.merge_dims(0, 1))
    detections_flat = merge_dims(detections)
    tracklets_flat = merge_dims(tracklets)

    # Extract appearance features.
    appearance_feature_extractor = _build_appearance_model(config=config)
    detections_features_flat = appearance_feature_extractor(detections_flat)
    tracklets_features_flat = appearance_feature_extractor(tracklets_flat)

    # Add the flattened dimensions back.
    to_ragged = layers.Lambda(
        lambda t: tf.RaggedTensor.from_row_lengths(t[0], t[1].row_lengths())
    )
    detections_features_ragged = to_ragged(
        (detections_features_flat, detections)
    )
    tracklets_features_ragged = to_ragged((tracklets_features_flat, tracklets))
    return detections_features_ragged, tracklets_features_ragged


def extract_interaction_features(
    *,
    detections: tf.RaggedTensor,
    tracklets: tf.RaggedTensor,
    detections_geometry: tf.RaggedTensor,
    tracklets_geometry: tf.RaggedTensor,
    config: ModelConfig,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Builds the portion of the system that extracts interaction features.

    Args:
        detections: Extracted detection images. Should have the shape
            `[batch_size, n_detections, height, width, channels]`, where the
            second dimension is ragged.
        tracklets: Extracted final images from each tracklet. Should have the
            shape `[batch_size, n_tracklets, height, width, channels]`, where
            the second dimension is ragged.
        detections_geometry: The geometric features associated with the
            detections. Should have the shape
            `[batch_size, n_detections, n_features]`, where the second dimension
            is ragged.
        tracklets_geometry: The geometric features associated with the
            tracklets. Should have the shape
            `[batch_size, n_tracklets, n_features]`, where the second dimension
            is ragged.
        config: The model configuration.

    Returns:
        The extracted interaction features, for both the left and right nodes
        in the graph, in that order. Each set will have the shape
        `[batch_size, max_n_nodes, n_inter_features]`, where the second
        dimension will be padded. Left nodes correspond to tracklets, and
        right nodes to detections.

    """
    # Extract the appearance features.
    (
        detection_app_features,
        tracklets_app_features,
    ) = extract_appearance_features(
        detections=detections, tracklets=tracklets, config=config
    )

    # For the rest of the pipeline, we need a dense representation of the
    # features.
    to_tensor = layers.Lambda(lambda rt: rt.to_tensor())
    detection_app_features = to_tensor(detection_app_features)
    tracklets_app_features = to_tensor(tracklets_app_features)
    detections_geom_features = to_tensor(detections_geometry)
    tracklets_geom_features = to_tensor(tracklets_geometry)

    # Create the edge feature extractor.
    edge_features = _build_edge_mlp(
        geometric_features=(detections_geom_features, tracklets_geom_features),
        appearance_features=(detection_app_features, tracklets_app_features),
    )

    # Create the adjacency matrix and build the GCN.
    adjacency_matrix = layers.Lambda(
        lambda f: normalize_adjacency(make_adjacency_matrix(f)),
        name="adjacency_matrix",
    )(edge_features)
    # Note that the order of concatenation is important here.
    combined_app_features = layers.Concatenate(axis=-2)(
        (tracklets_app_features, detection_app_features)
    )
    final_node_features = _build_gnn(
        adjacency_matrix=adjacency_matrix,
        node_features=combined_app_features,
        config=config,
    )

    # Split back into separate tracklets and detections.
    max_num_tracklets = tf.shape(tracklets_app_features)[1]
    tracklets_inter_features = final_node_features[:, :max_num_tracklets, :]
    detections_inter_features = final_node_features[:, max_num_tracklets:, :]
    return tracklets_inter_features, detections_inter_features


def compute_association(
    *,
    detections: tf.RaggedTensor,
    tracklets: tf.RaggedTensor,
    detections_geometry: tf.RaggedTensor,
    tracklets_geometry: tf.RaggedTensor,
    config: ModelConfig,
) -> tf.RaggedTensor:
    """
    Builds a model that computes associations between tracklets and detections.

    Args:
        detections: Extracted detection images. Should have the shape
            `[batch_size, n_detections, height, width, channels]`, where the
            second dimension is ragged.
        tracklets: Extracted final images from each tracklet. Should have the
            shape `[batch_size, n_tracklets, height, width, channels]`, where
            the second dimension is ragged.
        detections_geometry: The geometric features associated with the
            detections. Should have the shape
            `[batch_size, n_detections, n_features]`, where the second dimension
            is ragged.
        tracklets_geometry: The geometric features associated with the
            tracklets. Should have the shape
            `[batch_size, n_tracklets, n_features]`, where the second dimension
            is ragged.
        config: The model configuration.

    Returns:
        The association Sinkhorn matrices. Will have shape
        `[batch_size, n_tracklets * n_detections]`, where the inner
        dimension is ragged and represents the flattened Sinkhorn matrix.

    """
    # Extract interaction features.
    (
        tracklets_inter_features,
        detections_inter_features,
    ) = extract_interaction_features(
        detections=detections,
        tracklets=tracklets,
        detections_geometry=detections_geometry,
        tracklets_geometry=tracklets_geometry,
        config=config,
    )

    # Compute affinity scores. For this, we need a dense representation of
    # the geometric features.
    to_tensor = layers.Lambda(lambda rt: rt.to_tensor())
    detections_geom_features = to_tensor(detections_geometry)
    tracklets_geom_features = to_tensor(tracklets_geometry)
    affinity_scores = _build_affinity_mlp(
        detection_geom_features=detections_geom_features,
        tracklet_geom_features=tracklets_geom_features,
        detection_inter_features=detections_inter_features,
        tracklet_inter_features=tracklets_inter_features,
    )

    # Compute the Sinkhorn matrices.
    return _solve_association(
        affinity_scores=affinity_scores,
        num_detections=detections.row_lengths(),
        num_tracklets=tracklets.row_lengths(),
        config=config,
    )


def _make_image_input(config: ModelConfig, *, name: str) -> layers.Input:
    """
    Creates an input for detection or tracklet images.

    Args:
        config: The model configuration to use.
        name: The name to use for the input.

    Returns:
        The input that it created.

    """
    input_shape = (None,) + config.image_input_shape
    return layers.Input(input_shape, ragged=True, name=name, dtype="uint8")


def build_model(config: ModelConfig) -> tf.keras.Model:
    """
    Builds the complete Keras model.

    Args:
        config: The model configuration.

    Returns:
        The model that it created.

    """
    # Create the inputs.
    tracklet_input = _make_image_input(
        config, name=ModelInputs.TRACKLETS.value
    )
    detection_input = _make_image_input(
        config, name=ModelInputs.DETECTIONS.value
    )

    geometry_input_shape = (None, 4)
    detection_geometry_input = layers.Input(
        geometry_input_shape,
        ragged=True,
        name=ModelInputs.DETECTION_GEOMETRY.value,
    )
    tracklet_geometry_input = layers.Input(
        geometry_input_shape,
        ragged=True,
        name=ModelInputs.TRACKLET_GEOMETRY.value,
    )

    # Build the actual model.
    sinkhorn_matrices = compute_association(
        detections=detection_input,
        tracklets=tracklet_input,
        detections_geometry=detection_geometry_input,
        tracklets_geometry=tracklet_geometry_input,
        config=config,
    )
    return tf.keras.Model(
        inputs=[
            detection_input,
            tracklet_input,
            detection_geometry_input,
            tracklet_geometry_input,
        ],
        outputs={ModelTargets.SINKHORN.value: sinkhorn_matrices},
    )
