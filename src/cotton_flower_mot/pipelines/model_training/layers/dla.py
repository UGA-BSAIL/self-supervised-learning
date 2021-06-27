"""
Custom layers for building DLA networks.
"""

import abc
import enum
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow.keras import layers

from .utility import BnActConv


class _BilinearInitializer(tf.keras.initializers.Initializer):
    """
    A kernel initializer for deconvolution layers that initially sets them to
    bilinear up-sampling.

    """

    _BILINEAR_KERNEL = tf.constant(
        [[0.0, 0.25, 0.0], [0.25, 1.0, 0.25], [0.0, 0.25, 0.0]]
    )
    """
    The basic kernel to use for bilinear up-sampling.
    """

    def __call__(
        self, shape: tf.TensorShape, dtype: Optional[Type] = None, **_: Any
    ) -> tf.Tensor:
        """

        Args:
            shape: The shape of the kernel.
            dtype: The dtype to use.

        Returns:
            The initial tensor it created.

        """
        if shape[:2] != (3, 3):
            raise ValueError(
                f"Bilinear kernels must be 3x3, but this one is {shape[:2]}"
            )

        kernel = tf.cast(self._BILINEAR_KERNEL, dtype)

        # If we have additional dimensions, repeat the kernel.
        num_extra_dims = len(shape) - 2
        kernel = tf.reshape(kernel, (3, 3) + (1,) * num_extra_dims)
        tile_multiples = shape[2:]
        return tf.tile(kernel, (1, 1) + tile_multiples)


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
        self,
        inputs: Tuple[tf.Tensor, ...],
        training: Optional[bool] = None,
        **_,
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


class AggregationWithUpSample(layers.Layer):
    """
    Fused aggregation and up-sampling layer. One of the inputs will be
    up-sampled.
    """

    def __init__(
        self,
        channels: int,
        *args: Any,
        name: Optional[str] = None,
        activation: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Args:
            channels: The number of output channels.
            *args: Will be forwarded to the aggregation layer.
            name: The name of this layer.
            activation: The activation to use.
            **kwargs: Will be forwarded to the aggregation layer.

        """
        super().__init__(name=name)

        self._channels = channels
        self._activation = activation
        self._agg_args = args
        self._agg_kwargs = kwargs

        # Create the sub-layers.
        self._deconv1_1 = None
        self._pad1_1 = None
        self._aggregation1_1 = AggregationNode(
            channels, *args, activation=activation, **kwargs
        )

    def build(
        self, input_shape: Tuple[tf.TensorShape, tf.TensorShape]
    ) -> None:
        # Sanity-check the inputs.
        big_input, small_input = input_shape
        big_input_size = np.array(big_input[1:3])
        small_input_double_size = np.array(small_input[1:3]) * 2

        if not np.all(big_input_size - small_input_double_size <= 1):
            # Up-sampling exactly doubles the input size.
            raise ValueError(
                f"Up-sampled input must be half the size of the "
                f"other input, but sizes are {small_input}, "
                f"{big_input} respectively."
            )

        # We need both inputs to have the same number of channels.
        big_input_num_channels = big_input[3]
        self._deconv1_1 = layers.Conv2DTranspose(
            big_input_num_channels,
            3,
            strides=2,
            padding="same",
            activation=self._activation,
            kernel_initializer=_BilinearInitializer(),
            bias_initializer=tf.keras.initializers.constant(0.0),
        )

        # If our input size was not evenly divisible by two, we're going to
        # have to add some padding to make the sizes come out right.
        pad_top, pad_left = (
            np.not_equal(big_input_size, small_input_double_size)
            .astype(np.int)
            .tolist()
        )
        self._pad1_1 = layers.ZeroPadding2D(
            padding=((pad_top, 0), (pad_left, 0))
        )

        super().build(input_shape)

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: Optional[bool] = None,
        **_,
    ) -> tf.Tensor:
        """
        Args:
            inputs: Tensor inputs. The second should be the one that requires
                up-sampling.
            training: Whether this layer should be used in training mode.

        Returns:
            The aggregation output.

        """
        big_input, small_input = inputs

        up_sampled = self._pad1_1(self._deconv1_1(small_input))
        return self._aggregation1_1((big_input, up_sampled))

    def get_config(self) -> Dict[str, Any]:
        return dict(
            channels=self._channels,
            agg_args=self._agg_args,
            agg_kwargs=self._agg_kwargs,
            activation=self._activation,
            name=self.name,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AggregationWithUpSample":
        agg_args = config.pop("agg_args")
        agg_kwargs = config.pop("agg_kwargs")
        channels = config.pop("channels")

        return cls(channels, *agg_args, **agg_kwargs, **config)


class _ResidualBlock(layers.Layer):
    """
    Common superclass for all residual blocks.
    """

    def __init__(self, channels: int, *args: Any, **kwargs: Any):
        """
        Args:
            channels: The number of output channels.
            *args: Will be forwarded to the superclass.
            **kwargs: Will be forwarded to the superclass.

        """
        super().__init__(*args, **kwargs)

        self._num_channels = channels
        # Possible extra convolution that could be needed to make the output
        # channels match.
        self._projection_layer = None

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
            self._projection_layer = BnActConv(
                1, 1, padding="same", name="adapt_outputs", activation=None
            )

        super().build(input_shape)

    @property
    def _channel_projection(self) -> Optional[layers.Layer]:
        """
        Gets the extra 1x1 convolution used for making the number of channels
        compatible between the input and the output, if necessary.

        Returns:
            The 1x1 convolutional layer, if needed. Otherwise, just returns
            None.

        """
        return self._projection_layer

    def get_config(self) -> Dict[str, Any]:
        return dict(
            channels=self._num_channels,
            name=self.name,
        )


class BasicBlock(_ResidualBlock):
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
        super().__init__(channels, name=name)

        self._conv_args = args
        self._conv_kwargs = kwargs

        # Create the sub-layers.
        self._conv1_1 = BnActConv(channels, *args, **kwargs)
        self._conv1_2 = BnActConv(channels, *args, **kwargs)
        self._add1_1 = layers.Add()

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
            **super().get_config(),
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BasicBlock":
        conv_args = config.pop("conv_args")
        conv_kwargs = config.pop("conv_kwargs")
        channels = config.pop("channels")

        return cls(channels, *conv_args, **conv_kwargs, **config)


class BottleneckBlock(_ResidualBlock):
    """
    A residual block that uses a bottleneck architecture for reducing the
    number of parameters.
    """

    def __init__(
        self,
        channels: int,
        *args: Any,
        reduction_factor: float = 0.25,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Args:
            channels: The number of output channels for the convolution.
            *args: Will be forwarded to the internal `BnActConv` layer.
            reduction_factor: Factor by which to reduce the number of
                features in the bottleneck.
            name: The name of the layer.
            **kwargs: Will be forwarded to the internal `BnActConv` layer.

        """
        super().__init__(channels, name=name)

        self._conv_args = args
        self._conv_kwargs = kwargs
        self._reduction_factor = reduction_factor

        # Create the sub-layers.
        num_reduced_features = int(channels * reduction_factor)
        logger.debug("Bottleneck will have {} channels.", num_reduced_features)
        self._reduction_conv = BnActConv(num_reduced_features, *args, **kwargs)
        self._main_conv = BnActConv(num_reduced_features, *args, **kwargs)
        self._expansion_conv = BnActConv(channels, *args, **kwargs)
        self._add1_1 = layers.Add()

    def call(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        # Compute the residuals.
        residuals = self._reduction_conv(inputs)
        residuals = self._main_conv(residuals)
        residuals = self._expansion_conv(residuals)

        if self._channel_projection is not None:
            # Adapt the size so it matches up.
            inputs = self._channel_projection(inputs)

        return self._add1_1((residuals, inputs))

    def get_config(self) -> Dict[str, Any]:
        return dict(
            conv_args=self._conv_args,
            conv_kwargs=self._conv_kwargs,
            reduction_factor=self._reduction_factor,
            **super().get_config(),
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BottleneckBlock":
        conv_args = config.pop("conv_args")
        conv_kwargs = config.pop("conv_kwargs")
        channels = config.pop("channels")

        return cls(channels, *conv_args, **conv_kwargs, **config)


class GraphLayerMixin(abc.ABC):
    """
    Mixin for layers that have a lot of sub-layers connected in a complicated
    and dynamic way. The name derives from the fact that the layer
    connections are treated as a graph. It has nothing to do with GNNs.

    """

    @property
    @abc.abstractmethod
    def _node_to_inputs(self) -> Dict[layers.Layer, List[layers.Layer]]:
        """
        Returns:
            A flattened mapping of node layers to the particular inputs for that
            node layer.

        """

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


class HdaStage(layers.Layer, GraphLayerMixin):
    """
    A stage that performs Hierarchical Deep Aggregation.
    """

    @enum.unique
    class Block(enum.Enum):
        """
        Enumerates acceptable backbone blocks to use for HDA.
        """

        BASIC = BasicBlock
        BOTTLENECK = BottleneckBlock

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


class UpSamplingIda(layers.Layer, GraphLayerMixin):
    """
    Adds the special IDA nodes that are used for up-sampling the model output.
    """

    def __init__(
        self,
        agg_filter_size: Union[Tuple[int, int], int] = 1,
        activation: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Args:
            agg_filter_size: The size of the filters to use in the
                aggregation nodes.
            activation: The activation function.
            name: The name of this stage.

        """
        super().__init__(name=name)

        self._agg_filter_size = agg_filter_size
        self._activation = activation

        # List of all the aggregation nodes, organized by (level, horizontal
        # index). Levels start from 0 at the bottom and go up. Horizontal
        # indices start from 0 at the left and go right.
        self._agg_nodes: Optional[List[Dict[int, layers.Layer]]] = None
        # Number of output channels of the first stage. This will be the
        # number of channels in the overall output.
        self._num_first_stage_channels = None

    def _create_agg_nodes(
        self, num_stages: int
    ) -> List[Dict[int, layers.Layer]]:
        """
        Creates all the aggregation nodes.

        Args:
            num_stages: The total number of stages we are aggregating.

        Returns:
            The aggregation nodes that it created, indexed by level and then
            horizontal position.

        """
        # Overall structure will form an isosceles triangle pattern.
        num_levels = num_stages

        agg_nodes = []
        # Level 0 is the input stages, so we start at 1.
        for level in range(1, num_levels):
            num_nodes_in_level = num_levels - level
            start_index = level
            end_index = start_index + num_nodes_in_level

            agg_nodes.append(
                {
                    i: AggregationWithUpSample(
                        self._num_first_stage_channels * i,
                        self._agg_filter_size,
                        activation=self._activation,
                        padding="same",
                        name=f"level_{level}_up_sample_agg_{i}",
                    )
                    for i in range(start_index, end_index)
                }
            )

        return agg_nodes

    @cached_property
    def _node_to_inputs(self) -> Dict[layers.Layer, List[layers.Layer]]:
        nodes_to_inputs = {}

        # We don't add connections for the first level, because these come
        # directly from the input stages. Since these inputs are already
        # provided in tensor form, there's nothing we need to do about them.
        for level_i_1, level in enumerate(self._agg_nodes[1:]):
            # Actual level index will be one more since we exclude the input
            # stages.
            level_i = level_i_1 + 1

            for pos, node in level.items():
                # Get the inputs.
                agg_node_input = self._agg_nodes[level_i - 1][pos - 1]
                up_sampling_input = self._agg_nodes[level_i - 1][pos]

                # Note: These should be in the right order.
                nodes_to_inputs[node] = [agg_node_input, up_sampling_input]

        return nodes_to_inputs

    @property
    def _output_node(self) -> layers.Layer:
        """
        Returns:
            The highest-level aggregation node.

        """
        return self._agg_nodes[-1][len(self._agg_nodes)]

    def build(self, input_shape: Tuple[tf.Tensor, ...]) -> None:
        # Do some sanity checks on the input.
        current_stage_channels = input_shape[0][3]
        self._num_first_stage_channels = current_stage_channels
        for shape in input_shape[1:]:
            next_stage_channels = shape[3]
            if next_stage_channels != current_stage_channels * 2:
                raise ValueError(
                    f"Number of channels should double every "
                    f"stage, but it instead goes from "
                    f"{current_stage_channels} to "
                    f"{next_stage_channels}."
                )
            current_stage_channels = next_stage_channels

        # The number of input tensors is the number of stages that we need.
        num_stages = len(input_shape)
        # Build the aggregation nodes.
        self._agg_nodes = self._create_agg_nodes(num_stages)

        super().build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, ...], **kwargs: Any) -> tf.Tensor:
        """
        Args:
            inputs: The tensors from each input stage, in order.
            **kwargs: Will be forwarded to the sub-layers.

        Returns:
            The result from the up-sampling fusion operation.

        """
        # Mapping from layers to their output tensors.
        layers_to_outputs = {}
        # Compute the initial level of aggregation on the inputs.
        for pos, node in self._agg_nodes[0].items():
            big_input = inputs[pos - 1]
            small_input = inputs[pos]
            layers_to_outputs[node] = node((big_input, small_input), **kwargs)

        # Compute the final output tensor.
        return self._get_output_tensor(
            self._output_node,
            layers_to_outputs=layers_to_outputs,
            layer_kwargs=kwargs,
        )

    def get_config(self) -> Dict[str, Any]:
        return dict(
            agg_filter_size=self._agg_filter_size,
            activation=self._activation,
            name=self.name,
        )
