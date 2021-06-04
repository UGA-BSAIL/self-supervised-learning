"""
Custom Keras layers used by this pipeline.
"""


from kedro.extras.datasets.tensorflow import TensorFlowModelDataset

from .association import AssociationLayer
from .dense import DenseBlock, TransitionLayer
from .dla import AggregationNode, BasicBlock, HdaStage
from .gnn import DynamicEdgeGcn, ResidualGcn
from .mlp_conv import MlpConv
from .utility import BnActConv, BnActDense

# Make sure that Kedro is aware of custom layers.
if "custom_objects" not in TensorFlowModelDataset.DEFAULT_LOAD_ARGS:
    TensorFlowModelDataset.DEFAULT_LOAD_ARGS["custom_objects"] = {}
TensorFlowModelDataset.DEFAULT_LOAD_ARGS["custom_objects"].update(
    {
        "MlpConv": MlpConv,
        "DenseBlock": DenseBlock,
        "TransitionLayer": TransitionLayer,
        "AssociationLayer": AssociationLayer,
        "BnActConv": BnActConv,
        "BnActDense": BnActDense,
        "DynamicEdgeGcn": DynamicEdgeGcn,
        "ResidualGcn": ResidualGcn,
        "AggregationNode": AggregationNode,
        "BasicBlock": BasicBlock,
        "HdaStage": HdaStage,
    }
)
