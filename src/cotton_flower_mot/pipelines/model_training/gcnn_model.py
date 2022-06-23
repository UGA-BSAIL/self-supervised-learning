"""
Implements a model inspired by GCNNTrack.
https://arxiv.org/pdf/2010.00067.pdf
"""

from functools import partial
from typing import Tuple

import tensorflow as tf
from loguru import logger
from spektral.utils.convolution import line_graph
from tensorflow.keras import layers

from ..config import ModelConfig
from ..schemas import ModelInputs, ModelTargets
from .graph_utils import (
    compute_bipartite_edge_features,
    compute_pairwise_similarities,
    gcn_filter,
    make_complete_bipartite_adjacency_matrices,
)
from .layers import (
    AssociationLayer,
    BnActConv,
    HdaStage,
    ResidualCensNet,
    TransitionLayer,
)
from .similarity_utils import (
    aspect_ratio_penalty,
    compute_ious,
    cosine_similarity,
    distance_penalty,
)

# Use mixed precision to speed up training.
tf.keras.mixed_precision.set_global_policy("mixed_float16")


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

    stage1 = HdaStage(
        agg_depth=1, num_channels=64, activation="relu", name="hda_stage_1"
    )
    stage2 = HdaStage(
        agg_depth=2, num_channels=128, activation="relu", name="hda_stage_2"
    )
    stage3 = HdaStage(
        agg_depth=2, num_channels=256, activation="relu", name="hda_stage_3"
    )
    stage4 = HdaStage(
        agg_depth=1, num_channels=512, activation="relu", name="hda_stage_4"
    )

    hda1 = stage1(normalized_input)
    transition1 = TransitionLayer()(hda1)
    hda2 = stage2(transition1)
    transition2 = TransitionLayer()(hda2)
    hda3 = stage3(transition2)
    transition3 = TransitionLayer()(hda3)
    hda4 = stage4(transition3)

    # Generate feature vector.
    conv5_1 = BnActConv(config.num_appearance_features, 1, padding="same")(
        hda4
    )
    conv5_2 = BnActConv(config.num_appearance_features, 1, padding="same")(
        conv5_1
    )
    pool5_1 = layers.GlobalAvgPool2D()(conv5_2)

    return pool5_1


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

    normalized = layers.Lambda(_normalize, name="normalize")(images)

    # Apply the model layers.
    features = _build_appearance_feature_extractor(normalized, config=config)

    # Create the model.
    return tf.keras.Model(
        inputs=images, outputs=features, name="appearance_model"
    )


def _build_edge_mlp(
    *,
    geometric_features: Tuple[tf.Tensor, tf.Tensor],
    appearance_features: Tuple[tf.Tensor, tf.Tensor],
    config: ModelConfig,
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
        `[batch_size, n_edges, n_features]`.

    """

    def _combine_input_impl(
        _features: Tuple[tf.Tensor, tf.Tensor]
    ) -> tf.Tensor:
        detections, tracklets = _features
        input_edge_features = compute_bipartite_edge_features(
            left_nodes=tracklets, right_nodes=detections
        )

        # Concatenate the detection and tracklet features.
        tracklet_features = input_edge_features[:, :, :, 0, :]
        detection_features = input_edge_features[:, :, :, 1, :]
        fused_features = tf.concat(
            (tracklet_features, detection_features), axis=3
        )

        # We should know this statically.
        num_features = detections.shape[-1]
        return tf.ensure_shape(
            fused_features,
            (None, None, None, num_features * 2),
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

    # Apply the MLP.
    features = BnActConv(config.num_edge_features, 1)(all_features)

    def _reshape_outputs(_features: tf.Tensor) -> tf.Tensor:
        # Since our graph is complete and bipartite, to get the output shape we
        # want, we just have to fuse the inner two dimensions.
        feature_shape = tf.shape(_features)
        output_shape = tf.stack([feature_shape[0], -1, feature_shape[-1]])
        reshaped = tf.reshape(_features, output_shape)

        # We should know part of the shape statically.
        return tf.ensure_shape(reshaped, [None, None, _features.shape[3]])

    return layers.Lambda(_reshape_outputs, name="reshape_edge_features")(
        features
    )


def _build_affinity_mlp(
    *,
    detection_geom_features: tf.Tensor,
    tracklet_geom_features: tf.Tensor,
    detection_inter_features: tf.Tensor,
    tracklet_inter_features: tf.Tensor,
    detection_app_features: tf.Tensor,
    tracklet_app_features: tf.Tensor,
) -> tf.Tensor:
    """
    Builds the MLP that computes the affinity score between two nodes.

    Args:
        detection_geom_features: The padded detection geometry features,
            with shape `[batch_size, max_n_detections, 4]`.
        tracklet_geom_features: The padded tracklet geometry features,
            with shape `[batch_size, max_n_tracklets, 4]`.
        detection_inter_features: The padded detection interaction features,
            with shape `[batch_size, max_n_detections, num_features]`.
        tracklet_inter_features: The padded tracklet interaction features,
            with shape `[batch_size, max_n_tracklets, num_features]`.
        detection_app_features: The padded detection appearance features,
            with shape `[batch_size, max_n_detections, num_features]`.
        tracklet_app_features: The padded tracklet appearance features,
            with shape `[batch_size, max_n_tracklets, num_features]`.

    Returns:
        The final affinity scores between each pair of tracklet and detections.
        Will have a shape of `[batch_size, max_n_tracklets, max_n_detections]`.

    """
    # Compute similarity parameters.
    iou = layers.Lambda(
        lambda f: compute_pairwise_similarities(
            compute_ious, left_features=f[0], right_features=f[1]
        ),
        name="iou",
    )((tracklet_geom_features, detection_geom_features))
    distance = layers.Lambda(
        lambda f: compute_pairwise_similarities(
            distance_penalty, left_features=f[0], right_features=f[1]
        ),
        name="distance_penalty",
    )((tracklet_geom_features, detection_geom_features))
    aspect_ratio = layers.Lambda(
        lambda f: compute_pairwise_similarities(
            aspect_ratio_penalty,
            left_features=f[0],
            right_features=f[1],
        ),
        name="aspect_ratio_penalty",
    )((tracklet_geom_features, detection_geom_features))
    interaction_cosine = layers.Lambda(
        lambda f: compute_pairwise_similarities(
            cosine_similarity, left_features=f[0], right_features=f[1]
        ),
        name="interaction_cosine_similarity",
    )((tracklet_inter_features, detection_inter_features))
    appearance_cosine = layers.Lambda(
        lambda f: compute_pairwise_similarities(
            cosine_similarity, left_features=f[0], right_features=f[1]
        ),
        name="appearance_cosine_similarity",
    )((tracklet_app_features, detection_app_features))

    # Concatenate into our input.
    similarity_input = tf.stack(
        (iou, distance, aspect_ratio, interaction_cosine, appearance_cosine),
        axis=-1,
    )
    # Make sure the channels dimension is defined statically so Keras layers
    # work.
    similarity_input = tf.ensure_shape(similarity_input, (None, None, None, 5))

    # Apply the MLP. 1x1 convolution is an efficient way to apply the same MLP
    # to every detection/tracklet pair.
    conv1_1 = BnActConv(128, 1, name="affinity_conv_1")(similarity_input)
    conv1_2 = BnActConv(128, 1, name="affinity_conv_2")(conv1_1)
    conv1_3 = BnActConv(1, 1, name="affinity_conv_3")(conv1_2)

    # Remove the extraneous 1 dimension.
    return conv1_3[:, :, :, 0]


def _build_gnn(
    *,
    graph_structure: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    node_features: tf.Tensor,
    edge_features: tf.Tensor,
    config: ModelConfig,
) -> tf.Tensor:
    """
    Builds the GNN for performing feature association.

    Args:
        graph_structure: The pre-processed matrices describing the structure
            of the graph. These can be generated using
            `CensNetConv.preprocess()`.
        node_features: The input node features. Should have the shape
            `[batch_size, n_nodes, n_features]`.
        edge_features: The input edge features. Should have the shape
            `[batch_size, n_edges, n_features]`.
        config: The model configuration.

    Returns:
        The output node features from the GNN, which will have the shape
        `[batch_size, n_nodes, n_gcn_channels]`.

    """
    nodes1_1, edges1_1 = ResidualCensNet(
        config.num_node_features, config.num_edge_features
    )((node_features, graph_structure, edge_features))
    nodes1_2, _ = ResidualCensNet(
        config.num_node_features, config.num_edge_features
    )(nodes1_1, graph_structure, edges1_1)

    return nodes1_2


def _triangular_adjacency(adjacency):
    """
    Gets the triangular version of the adjacency matrix, removing redundant
    values.

    :param adjacency: The full adjacency matrix, with shape
        ([batch], n_nodes, n_nodes).
    :return: The upper triangle of the adjacency matrix, with the lower
        triangle set to zero.
    """
    return tf.linalg.band_part(adjacency, 0, -1)


def _incidence_matrix_single(triangular_adjacency, *, num_edges):
    """
    Creates the corresponding incidence matrix for a graph with a particular
    adjacency matrix.

    :param triangular_adjacency: The binary adjacency matrix. Should have shape
        (n_nodes, n_nodes), and be triangular.
    :param num_edges: The number of edges to use in the output. Should be large
        enough to accommodate all the edges in the adjacency matrix.

    :return: The computed incidence matrix. It will have a shape of
        (n_nodes, n_edges).
    """
    # The adjacency matrix should be sparse, so get the indices of the edges.
    connected_node_indices = tf.where(triangular_adjacency)

    # Match each edge with one of the nodes connected by that edge. We refer
    # to the two nodes connected by each edge as "right" and "left",
    # for convenience.
    edge_indices = tf.range(
        tf.shape(connected_node_indices)[0], dtype=tf.int64
    )
    edges_with_left_nodes = tf.stack(
        [connected_node_indices[:, 0], edge_indices], axis=1
    )
    edges_with_right_nodes = tf.stack(
        [connected_node_indices[:, 1], edge_indices], axis=1
    )

    # We now have all the points that should go in the sparse binary
    # transformation matrix.
    edge_indicators = tf.ones_like(edge_indices, dtype=tf.float32)
    num_nodes = tf.cast(tf.shape(triangular_adjacency)[0], tf.int64)
    output_shape = tf.stack([num_nodes, num_edges])
    left_sparse = tf.SparseTensor(
        indices=edges_with_left_nodes,
        values=edge_indicators,
        dense_shape=output_shape,
    )
    left_sparse = tf.sparse.reorder(left_sparse)
    right_sparse = tf.SparseTensor(
        indices=edges_with_right_nodes,
        values=edge_indicators,
        dense_shape=output_shape,
    )
    right_sparse = tf.sparse.reorder(right_sparse)
    # Combine the matrices for the left and right nodes.
    combined_sparse = tf.sparse.maximum(left_sparse, right_sparse)

    return tf.sparse.to_dense(combined_sparse)


def _incidence_matrix(adjacency):
    """
    Creates the corresponding incidence matrices for graphs with particular
    adjacency matrices.

    :param adjacency: The binary adjacency matrices. Should have shape
        ([batch], n_nodes, n_nodes).
    :return: The computed incidence matrices. It will have a shape of
        ([batch], n_nodes, n_edges).
    """
    adjacency = tf.convert_to_tensor(adjacency, dtype=tf.float32)
    added_batch = False
    if len(adjacency.shape) == 2:
        # Add the extra batch dimension if needed.
        adjacency = tf.expand_dims(adjacency, axis=0)
        added_batch = True

    # Compute the maximum number of edges. We will pad everything in the
    # batch to this dimension.
    adjacency_upper = _triangular_adjacency(adjacency)
    num_edges = tf.math.count_nonzero(adjacency_upper, axis=(1, 2))
    max_num_edges = tf.reduce_max(num_edges)

    # Compute all the transformation matrices.
    make_single_matrix = partial(
        _incidence_matrix_single, num_edges=max_num_edges
    )
    transformation_matrices = tf.map_fn(
        make_single_matrix,
        adjacency_upper,
        fn_output_signature=tf.TensorSpec(
            shape=[None, None], dtype=tf.float32
        ),
    )

    if added_batch:
        # Remove the extra batch dimension before returning.
        transformation_matrices = transformation_matrices[0]
    return transformation_matrices


def _preprocess_adjacency(
    adjacency_matrices: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Pre-processes that adjacency matrix for `CensNet`. Equivalent to
    `CensNetConv.preprocess()` except that it works on tensors.

    Args:
        adjacency_matrices: The binary adjacency matrix. Should have shape
            `[[batch_size], n_nodes, n_nodes]`.

    Returns:
        The node Laplacian, edge Laplacian, and incidence matrix.

    """
    # Force use of float32 here, but convert back once finished.
    input_dtype = adjacency_matrices.dtype
    adjacency_matrices = tf.cast(adjacency_matrices, tf.float32)

    node_laplacian = gcn_filter(adjacency_matrices)
    incidence = _incidence_matrix(adjacency_matrices)
    edge_laplacian = gcn_filter(line_graph(incidence))

    return (
        tf.cast(node_laplacian, input_dtype),
        tf.cast(edge_laplacian, input_dtype),
        tf.cast(incidence, input_dtype),
    )


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


def extract_dense_appearance_features(
    *,
    detections: tf.RaggedTensor,
    tracklets: tf.RaggedTensor,
    config: ModelConfig,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Builds the portion of the system that extracts appearance features, and
    produces features that are padded into a tensor.

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
        `[batch_size, max_num_nodes, n_features]`, where the second dimension is
        padded.

    """
    # Extract the appearance features.
    (
        detections_app_features,
        tracklets_app_features,
    ) = extract_appearance_features(
        detections=detections, tracklets=tracklets, config=config
    )

    # Pad them to dense tensors.
    to_tensor = layers.Lambda(lambda rt: rt.to_tensor())
    detections_app_features = to_tensor(detections_app_features)
    tracklets_app_features = to_tensor(tracklets_app_features)

    return detections_app_features, tracklets_app_features


def extract_interaction_features(
    *,
    detections_app_features: tf.Tensor,
    tracklets_app_features: tf.Tensor,
    detections_geometry: tf.RaggedTensor,
    tracklets_geometry: tf.RaggedTensor,
    config: ModelConfig,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Builds the portion of the system that extracts interaction features.

    Args:
        detections_app_features: Padded detection appearance features, with
            shape `[batch_size, max_num_detections, num_features]`.
        tracklets_app_features: Padded tracklet appearance features, with shape
            `[batch_size, max_num_tracklets, num_features]`.
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
    # For the rest of the pipeline, we need a dense representation of the
    # features.
    to_tensor = layers.Lambda(lambda rt: rt.to_tensor())
    detections_geom_features = to_tensor(detections_geometry)
    tracklets_geom_features = to_tensor(tracklets_geometry)

    # Create the edge feature extractor.
    edge_features = _build_edge_mlp(
        geometric_features=(detections_geom_features, tracklets_geom_features),
        appearance_features=(detections_app_features, tracklets_app_features),
        config=config,
    )

    # Create the adjacency matrix and build the GCN.
    num_detections = detections_geometry.row_lengths()
    num_tracklets = tracklets_geometry.row_lengths()
    adjacency_matrices = layers.Lambda(
        lambda n: make_complete_bipartite_adjacency_matrices(n[0], n[1]),
        name="adjacency_matrices",
    )((num_tracklets, num_detections))
    # Compute CensNet graph structure inputs.
    graph_structure = layers.Lambda(
        _preprocess_adjacency, name="preprocess_cens_net"
    )(adjacency_matrices)

    # Note that the order of concatenation is important here.
    combined_app_features = layers.Concatenate(axis=1)(
        (tracklets_app_features, detections_app_features)
    )
    final_node_features = _build_gnn(
        graph_structure=graph_structure,
        node_features=combined_app_features,
        edge_features=edge_features,
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
) -> Tuple[tf.RaggedTensor, tf.RaggedTensor]:
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
        The association and assignment matrices. Will have shape
        `[batch_size, n_tracklets * n_detections]`, where the inner
        dimension is ragged and represents the flattened matrix. The association
        matrix is simply the Sinkhorn-normalized associations, whereas the
        assignment matrix is the hard assignments calculated with the
        Hungarian algorithm.

    """
    # Extract appearance features.
    (
        detections_app_features,
        tracklets_app_features,
    ) = extract_dense_appearance_features(
        detections=detections, tracklets=tracklets, config=config
    )
    # Extract interaction features.
    (
        tracklets_inter_features,
        detections_inter_features,
    ) = extract_interaction_features(
        detections_app_features=detections_app_features,
        tracklets_app_features=tracklets_app_features,
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
        detection_app_features=detections_app_features,
        tracklet_app_features=tracklets_app_features,
    )

    # Compute the association matrices.
    return AssociationLayer(sinkhorn_lambda=config.sinkhorn_lambda)(
        (affinity_scores, detections.row_lengths(), tracklets.row_lengths())
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
    sinkhorn, assignment = compute_association(
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
        outputs={
            ModelTargets.SINKHORN.value: sinkhorn,
            ModelTargets.ASSIGNMENT.value: assignment,
        },
        name="gcnnmatch",
    )
