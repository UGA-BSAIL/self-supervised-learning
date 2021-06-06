"""
Custom Keras layers used by this pipeline.
"""


from kedro.extras.datasets.tensorflow import TensorFlowModelDataset

from .association import AssociationLayer
from .dense import DenseBlock, TransitionLayer
from .dla import AggregationNode, BasicBlock, HdaStage, UpSamplingIda
from .gnn import DynamicEdgeGcn, ResidualGcn
from .mlp_conv import MlpConv
from .utility import BnActConv, BnActDense

# Make sure that Kedro is aware of custom layers.
CUSTOM_LAYERS = {
    layer_class.__name__: layer_class
    for layer_class in (
        MlpConv,
        DenseBlock,
        TransitionLayer,
        AssociationLayer,
        BnActConv,
        BnActDense,
        DynamicEdgeGcn,
        ResidualGcn,
        AggregationNode,
        BasicBlock,
        HdaStage,
        UpSamplingIda,
    )
}
if "custom_objects" not in TensorFlowModelDataset.DEFAULT_LOAD_ARGS:
    TensorFlowModelDataset.DEFAULT_LOAD_ARGS["custom_objects"] = {}
TensorFlowModelDataset.DEFAULT_LOAD_ARGS["custom_objects"].update(
    CUSTOM_LAYERS
)
