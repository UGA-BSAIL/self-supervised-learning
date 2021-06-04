"""
Custom layers for building DLA networks.
"""

import enum
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import tensorflow as tf
from loguru import logger
from tensorflow.keras import layers

from .utility import BnActConv


class AggregationNode(layers.Layer):
    """
    Represents an aggregation node in the model.
    """

    def __init__(self, *args: Any, name: Optional[str] = None, **kwargs: Any):
        """
        Args:
            *args: Will be forwarded to the internal convolution layer.
            activation: The activation to use.
            name: The name of this layer.
            **kwargs: Will be forwarded to the internal convolution layer.
        """
        super().__init__(name=name)

        # We deliberately don't use double-underscores here to avoid
        # Autograph issues with mangled names.
        self._conv_args = args
        self._conv_kwargs = kwargs

        # Pre-create the sub-layers.
        self._add1_1 = layers.Add()
        # We will create the convolutions when we know how many inputs we have.
        self._conv_layers = []

    def build(self, input_shape: Tuple[tf.TensorShape, ...]) -> None:
        logger.debug(
            "Aggregation node {} will have {} inputs.",
            self.name,
            len(input_shape),
        )

        # We will need separate layers for each input.
        for _ in range(len(input_shape)):
            conv_layer = BnActConv(*self._conv_args, **self._conv_kwargs)
            self._conv_layers.append(conv_layer)

        super().build(input_shape)

    def call(
        self, inputs: Tuple[tf.Tensor, ...], training: Optional[bool] = None
    ) -> tf.Tensor:
        """

        Args:
            inputs: Inputs to be aggregated.
            training: Whether to enable training mode on the BN layers.

        Returns:
            The aggregated features.

        """
        # First, apply the transformation to each input.
        to_aggregate = []
        for node_input, conv_layer in zip(inputs, self._conv_layers):
            to_aggregate.append(conv_layer(node_input))

        # Aggregate the results.
        return self._add1_1(to_aggregate)

    def get_config(self) -> Dict[str, Any]:
        return dict(
            conv_args=self._conv_args,
            conv_kwargs=self._conv_kwargs,
            name=self.name,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AggregationNode":
        conv_args = config.pop("conv_args")
        conv_kwargs = config.pop("conv_kwargs")
        return cls(*conv_args, **conv_kwargs, **config)


class BasicBlock(layers.Layer):
    """
    A basic residual block for building the backbone of DLA networks.
    """

    def __init__(
        self,
        channels: int,
        *args: Any,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Args:
            channels: The number of output channels for the convolution.
            *args: Will be forwarded to the internal `BnActConv` layer.
            name: The name of the layer.
            **kwargs: Will be forwarded to the internal `BnActConv` layer.
        """
        super().__init__(name=name)

        self._conv_args = args
        self._conv_kwargs = kwargs
        self._num_channels = channels

        # Create the sub-layers.
        self._conv1_1 = BnActConv(channels, *args, **kwargs)
        self._conv1_2 = BnActConv(channels, *args, **kwargs)
        # Possible extra convolution that could be needed to make the output
        # channels match.
        self._channel_projection = None
        self._add1_1 = layers.Add()

    def build(self, input_shape: tf.TensorShape) -> None:
        # Add the additional convolution layer if the output channels don't
        # match.
        num_input_channels = input_shape[-1]
        if num_input_channels != self._num_channels:
            logger.debug(
                "Adding extra convolution to {} to "
                "convert from {} channels to {}.",
                self.name,
                num_input_channels,
                self._num_channels,
            )
            self._channel_projection = BnActConv(
                1, 1, padding="same", name="adapt_outputs", activation=None
            )

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        # Compute the residuals.
        residuals = self._conv1_1(inputs, **kwargs)
        residuals = self._conv1_2(residuals, **kwargs)

        if self._channel_projection is not None:
            # Adapt the size so it matches up.
            inputs = self._channel_projection(inputs)

        return self._add1_1((residuals, inputs))

    def get_config(self) -> Dict[str, Any]:
        return dict(
            conv_args=self._conv_args,
            conv_kwargs=self._conv_kwargs,
            channels=self._num_channels,
            name=self.name,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BasicBlock":
        conv_args = config.pop("conv_args")
        conv_kwargs = config.pop("conv_kwargs")
        channels = config.pop("channels")

        return cls(channels, *conv_args, **conv_kwargs, **config)


class HdaStage(layers.Layer):
    """
    A stage that performs Hierarchical Deep Aggregation.
    """

    @enum.unique
    class Block(enum.Enum):
        """
        Enumerates acceptable backbone blocks to use for HDA.
        """

        BASIC = BasicBlock

    def __init__(
        self,
        *,
        block_type: Block = Block.BASIC,
        agg_depth: int,
        num_channels: int,
        agg_filter_size: Union[Tuple[int, int], int] = 1,
        activation: Optional[str] = None,
        add_ida_skip: bool = True,
        name: Optional[str] = None,
    ):
        """
        Args:
            block_type: The layer type to use for the backbone blocks. It should
                accept the same input arguments as convolutional layers.
            agg_depth: The aggregation depth to use for this stage. This will
                determine the size of the backbone.
            num_channels: The number of output channels.
            agg_filter_size: The size of the filters to use in the
                aggregation nodes.
            activation: The activation function.
            add_ida_skip: If true, it will add a skip connection from the input
                to the top-level aggregation node, implementing IDA.
            name: The name of this stage.

        """
        super().__init__(name=name)

        self._block_type = block_type
        self._agg_depth = agg_depth
        self._num_channels = num_channels
        self._agg_filter_size = agg_filter_size
        self._activation = activation
        self._add_ida_skip = add_ida_skip

        # Create the backbone blocks. We use a grid system to index nodes,
        # by both level and horizontal location. The level index starts from
        # the bottom and goes up, while the horizontal index starts from the
        # left and goes right.
        logger.debug(
            "Stage {} has {} backbone blocks.",
            self.name,
            self.num_backbone_blocks,
        )
        self._backbone = {
            i: block_type.value(
                num_channels,
                3,
                activation=activation,
                padding="same",
                name=f"backbone_{i}",
            )
            for i in range(self.num_backbone_blocks)
        }
        # Create the aggregation nodes.
        self._agg_nodes = [
            self._create_agg_nodes(d) for d in range(1, agg_depth + 1)
        ]
        self._nodes = [self._backbone] + self._agg_nodes

    def _create_agg_nodes(self, depth: int) -> Dict[int, AggregationNode]:
        """
        Creates the set of aggregation nodes at a particular depth.

        Args:
            depth: The depth, where level 0 is the top-most level in the
                hierarchy, having only one aggregation node.

        Returns:
            The aggregation nodes at this depth.

        """
        # Calculate the nominal stride of the horizontal indices.
        stride = 2 ** depth
        # The stride is increased by a factor of two because vertical chains
        # of aggregation nodes will be fused into one. Therefore, every other
        # node is just going to get removed anyway.
        fused_stride = stride * 2
        start_index = stride - 1

        return {
            i: AggregationNode(
                self._num_channels,
                self._agg_filter_size,
                activation=self._activation,
                padding="same",
                name=f"level_{depth}_agg_node_{i}",
            )
            for i in range(start_index, self.num_backbone_blocks, fused_stride)
        }

    @cached_property
    def num_backbone_blocks(self) -> int:
        """
        Returns:
            The number of total blocks in the backbone.

        """
        # Each aggregation level reduces the number of nodes by half.
        return 2 ** self._agg_depth

    def _find_reentrant_input(self, index: int) -> layers.Layer:
        """
        Finds the reentrant input for a node in the backbone.

        Args:
            index: The index of the node in the backbone.

        Returns:
            The reentrant input for that node.

        """
        # The nearest aggregation node should be one behind the current
        # node, but we need to find the right level.
        agg_index = index - 1
        for level in range(1, self._agg_depth + 1):
            if agg_index in self._nodes[level]:
                return self._nodes[level][agg_index]
        # We shouldn't get here.
        raise AssertionError(f"Backbone node {index} has no reentrant parent?")

    def _get_backbone_inputs(
        self,
        index: int,
    ) -> Tuple[layers.Layer, layers.Layer]:
        """
        Gets the inputs from the backbone that feed into a particular
        aggregation node.

        Args:
            index: The horizontal index of the aggregation node.

        Returns:
            The two backbone inputs for that node.

        """
        assert index > 0, "There should be no aggregation node at index 0."
        return self._nodes[0][index], self._nodes[0][index - 1]

    def _find_child_agg_node(
        self, level: int, index: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Finds the downstream aggregation node that this one feeds into.

        Args:
            level: The level of this aggregation node.
            index: The index of this aggregation node.

        Returns:
            The level and index of the child node, or (None, None) if
            this node is the last one and has no child.

        """
        # The child node will be one level up, and the closest node to
        # the right.
        child_level = level + 1
        if child_level > self._agg_depth:
            # This must be the output node.
            assert (
                index == self.num_backbone_blocks - 1
            ), f"Output node has unexpected index {index}."
            return None, None

        for child_index in range(index, self.num_backbone_blocks):
            if child_index in self._nodes[child_level]:
                return child_level, child_index

        # We should never get here, because only the output has no child.
        raise AssertionError(f"Node at {level}, {index} has no children.")

    @cached_property
    def _node_pos_to_inputs(
        self,
    ) -> List[Dict[int, List[layers.Layer]]]:
        """
        Returns:
            A mapping of (level, horizontal index) to the corresponding inputs
            for that node.

        """
        node_pos_to_inputs = [
            {i: [] for i in self._nodes[level].keys()}
            for level in range(self._agg_depth + 1)
        ]

        # Do the backbone routing linear routing.
        for i in range(0, self.num_backbone_blocks, 2):
            node_pos_to_inputs[0][i + 1].append(self._backbone[i])
        # Do the backbone reentrant routing.
        for i in range(2, self.num_backbone_blocks, 2):
            node_pos_to_inputs[0][i].append(self._find_reentrant_input(i))

        for level_i_1, level in enumerate(self._agg_nodes):
            # Actual index is off by one since we're skipping the backbone.
            level_i = level_i_1 + 1

            for i, node in level.items():
                # Do the routing from the backbone to the aggregation nodes.
                backbone_inputs = self._get_backbone_inputs(i)
                node_pos_to_inputs[level_i][i].extend(backbone_inputs)

                # Do the routing between aggregation nodes.
                child_level_i, child_i = self._find_child_agg_node(level_i, i)
                if child_level_i is not None:
                    node_pos_to_inputs[child_level_i][child_i].append(node)

        return node_pos_to_inputs

    @cached_property
    def _node_to_inputs(self) -> Dict[layers.Layer, List[layers.Layer]]:
        """
        Returns:
            A flattened mapping of node layers to the particular inputs for that
            node layer.

        """
        node_to_inputs = {}
        for layer_i, layer in enumerate(self._nodes):
            for i, node in layer.items():
                node_to_inputs[node] = self._node_pos_to_inputs[layer_i][i]

        # Backbone nodes should never have more than one input.
        for node in self._backbone.values():
            assert (
                len(node_to_inputs[node]) <= 1
            ), f"Backbone node {node.name} has more than one input."

        return node_to_inputs

    def _get_output_tensor(
        self,
        layer: layers.Layer,
        *,
        layers_to_outputs: Dict[layers.Layer, tf.Tensor],
        extra_inputs: List[tf.Tensor] = [],
        layer_kwargs: Dict[str, Any] = {},
    ) -> tf.Tensor:
        """
        Gets the output tensor for a particular layer.

        Args:
            layer: The layer to get the output for.
            layers_to_outputs: Dictionary mapping layers to output tensors for
                that layer. It will essentially be used as a cache and may
                be modified by this method.
            extra_inputs: Auxiliary inputs that will be fed into this node,
                along with the ones we calculated.
            layer_kwargs: Will be passed as keyword arguments to any layer
                that we call.

        Returns:
            The output tensor for this layer.

        """
        cached_output = layers_to_outputs.get(layer)
        if cached_output is not None:
            # We have it cached already.
            return cached_output

        # Gather the inputs for this layer.
        layer_inputs = self._node_to_inputs[layer]
        input_tensors = extra_inputs[:]
        for input_layer in layer_inputs:
            # Compute to output tensor for each input.
            input_tensors.append(
                self._get_output_tensor(
                    input_layer,
                    layers_to_outputs=layers_to_outputs,
                    layer_kwargs=layer_kwargs,
                )
            )

        assert len(input_tensors) > 0, "Expected at least one input."
        if len(input_tensors) == 1:
            # Keras is weird about singleton inputs.
            input_tensors = input_tensors[0]
        # Apply the layer.
        output_tensor = layer(input_tensors, **layer_kwargs)
        # Update the cache.
        layers_to_outputs[layer] = output_tensor

        return output_tensor

    @property
    def _input_node(self) -> layers.Layer:
        """
        Returns:
            The first node in the backbone.

        """
        return self._nodes[0][0]

    @property
    def _output_node(self) -> layers.Layer:
        """
        Returns:
            The highest-level aggregation node.

        """
        return self._nodes[self._agg_depth][self.num_backbone_blocks - 1]

    def call(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        """
        Args:
            inputs: The input from the last stage.
            **kwargs: Will be forwarded to the internal layers.

        Returns:
            The result from this stage.

        """
        # Mapping from layers to their output tensors.
        # Inputs always get routed to the first backbone node.
        layers_to_outputs = {
            self._input_node: self._input_node(inputs, **kwargs)
        }

        extra_inputs = []
        if self._add_ida_skip:
            # Make sure the input also gets fed directly to the output node.
            extra_inputs.append(inputs)

        return self._get_output_tensor(
            self._output_node,
            layers_to_outputs=layers_to_outputs,
            extra_inputs=extra_inputs,
            layer_kwargs=kwargs,
        )

    def get_config(self) -> Dict[str, Any]:
        return dict(
            block_type=self._block_type.name,
            agg_depth=self._agg_depth,
            num_channels=self._num_channels,
            agg_filter_size=self._agg_filter_size,
            activation=self._activation,
            add_ida_skip=self._add_ida_skip,
            name=self.name,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HdaStage":
        block_type = cls.Block[config.pop("block_type")]
        return cls(block_type=block_type, **config)
