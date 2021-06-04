"""
Layers for graph neural networks.
"""


from typing import Any, Dict, Optional, Tuple

import spektral
import tensorflow as tf
from loguru import logger
from tensorflow.keras import layers

from ..graph_utils import augment_adjacency_matrix, bound_adjacency, gcn_filter
from .utility import BnActConv


class _AdjacencyMatrixUpdate(layers.Layer):
    """
    Updates the adjacency matrix for the next layer according to the method
    specified in https://arxiv.org/pdf/2010.00067.pdf.

    """

    def __init__(self, name: Optional[str] = None):
        """
        Args:
            name: The name of this layer.
        """
        super().__init__(name=name)

        # Pre-create the sub-layers.
        # Combine the affinity matrix with node features.
        self._aug1_1 = layers.Lambda(
            lambda x: augment_adjacency_matrix(
                adjacency_matrix=tf.expand_dims(x[0], axis=-1),
                node_features=x[1],
            ),
            name="augment_adjacency",
        )

        # The MLP operation over all edges is actually implemented as 1x1
        # convolution for convenience.
        self._conv1_2 = BnActConv(1, 1, name="edge_conv_1")

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        """

        Args:
            inputs: Inputs for this operation, including:
            - node_features: The node features from the previous layer. Should
                have the shape `[batch_size, n_nodes, n_features]`.
            - adjacency_matrix: The adjacency matrix from the previous layer.
                Should have the shape `[batch_size, n_nodes, n_nodes]`.
            training: Whether to enable training mode on the BN layers.

        Returns:
            The adjacency matrix for the next layer.

        """
        node_features, adjacency_matrix = inputs
        augmented_features = self._aug1_1((adjacency_matrix, node_features))
        adjacency = self._conv1_2(augmented_features, training=training)

        # Remove the final dimension, which should be one.
        return adjacency[:, :, :, 0]


class DynamicEdgeGcn(layers.Layer):
    """
    A new GCN layer with a corresponding affinity matrix update,
    in accordance with the method described in
    https://arxiv.org/pdf/2010.00067.pdf. Note that it performs the Laplacian
    calculation as well as batch normalization and activation internally. It
    will also take care of clipping the adjacency matrix to remove negative
    values.

    """

    def __init__(self, *args: Any, name: Optional[str] = None, **kwargs: Any):
        """
        Args:
            *args: Will be forwarded to the `GCNConv` layer.
            name: The name of this layer.
            **kwargs: Will be forwarded to the `GCNConv` layer.
        """
        super().__init__(name=name)

        # We deliberately don't use double-underscores here to avoid
        # Autograph issues with mangled names.
        self._gcn_args = args
        self._gcn_kwargs = kwargs

        # Pre-create the sub-layers.
        self._norm1_1 = layers.BatchNormalization()
        self._relu1_1 = layers.ReLU()

        self._laplacian1_1 = layers.Lambda(gcn_filter)
        self._gcn1_1 = spektral.layers.GCNConv(*args, **kwargs)

        self._edges1_1 = _AdjacencyMatrixUpdate()
        # Normalization operation for the adjacency matrix.
        self._edge_norm1_1 = layers.Lambda(
            lambda x: bound_adjacency(x), name="normalize_adjacency"
        )

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: Optional[bool] = None,
        skip_edge_update: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """

        Args:
            inputs: Inputs for this operation, including:
            - node_features: The node features from the previous layer. Should
                have the shape `[batch_size, n_nodes, n_features]`.
            - adjacency_matrix: The adjacency matrix from the previous layer.
                Should have the shape `[batch_size, n_nodes, n_nodes, 1]`.
            training: Whether to enable training mode on the BN layers.
            skip_edge_update: If true, it will skip updating the adjacency
                matrix and simply return an empty tensor in its place. This
                can be useful if this is the last layer in a stack and you
                don't need to use this output.

        Returns:
            The new node features and the new adjacency matrix.

        """
        node_features, adjacency_matrix = inputs

        nodes_normalized = self._norm1_1(node_features, training=training)
        nodes_pre_activated = self._relu1_1(nodes_normalized)

        normalized_laplacian = self._laplacian1_1(
            self._edge_norm1_1(adjacency_matrix)
        )
        new_nodes = self._gcn1_1((nodes_pre_activated, normalized_laplacian))

        new_edges = tf.constant([])
        if not skip_edge_update:
            new_edges = self._edges1_1((new_nodes, adjacency_matrix))
        return new_nodes, new_edges

    def get_config(self) -> Dict[str, Any]:
        return dict(
            gcn_args=self._gcn_args,
            gcn_kwargs=self._gcn_kwargs,
            name=self.name,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DynamicEdgeGcn":
        gcn_args = config.pop("gcn_args")
        gcn_kwargs = config.pop("gcn_kwargs")
        return cls(*gcn_args, **gcn_kwargs, **config)


class ResidualGcn(layers.Layer):
    """
    An extension to `DynamicEdgeGcn` that uses it as part of a residual block.
    """

    def __init__(
        self,
        channels: int,
        *args: Any,
        name: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Args:
            channels: Number of output channels for the GCN layer.
            *args: Will be forwarded to the `DynamicEdgeGcn` layer.
            name: The name of this layer.
            **kwargs: Will be forwarded to the `DynamicEdgeGcn` layer.
        """
        super().__init__(name=name)

        self._gcn_args = args
        self._gcn_kwargs = kwargs
        self._num_channels = channels

        # Pre-create the sub-layers.
        self._conv1_1 = None

        self._gcn1_1 = DynamicEdgeGcn(channels, *args, **kwargs)
        self._add_nodes = layers.Add(name="add_nodes")
        self._add_edges = layers.Add(name="add_edges")

    def build(
        self, input_shape: Tuple[tf.TensorShape, tf.TensorShape]
    ) -> None:
        # Add the additional convolution layer if the sizes don't match.
        node_shape, _ = input_shape
        num_input_channels = node_shape[-1]
        if num_input_channels != self._num_channels:
            logger.debug(
                "Adding extra convolution to {} to "
                "convert from {} channels to {}.",
                self.name,
                num_input_channels,
                self._num_channels,
            )

            # We bring the number of channels in-line with a 1D convolution,
            # since the input has 3 channels and the first 2 should always be
            # the same.
            self._conv1_1 = layers.Conv1D(
                self._num_channels, 1, padding="same", name="adapt_outputs"
            )

        super().build(input_shape)

    def call(
        self, inputs: Tuple[tf.Tensor, tf.Tensor], **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        nodes, edges = inputs
        nodes_res, edges_res = self._gcn1_1(inputs, **kwargs)

        if self._conv1_1 is not None:
            # Adapt the input size so it matches up.
            nodes = self._conv1_1(nodes)

        # Compute the residual.
        new_nodes = self._add_nodes([nodes, nodes_res])
        new_edges = self._add_edges([edges, edges_res])
        return new_nodes, new_edges

    def get_config(self) -> Dict[str, Any]:
        return dict(
            gcn_args=self._gcn_args,
            gcn_kwargs=self._gcn_kwargs,
            channels=self._num_channels,
            name=self.name,
        )

    @classmethod
    def from_config(cls, config) -> "ResidualGcn":
        gcn_args = config.pop("gcn_args")
        gcn_kwargs = config.pop("gcn_kwargs")
        channels = config.pop("channels")

        return cls(channels, *gcn_args, **gcn_kwargs, **config)
