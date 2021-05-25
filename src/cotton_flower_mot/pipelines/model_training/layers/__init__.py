"""
Custom Keras layers used by this pipeline.
"""


from kedro.extras.datasets.tensorflow import TensorFlowModelDataset

from .association import AssociationLayer
from .dense import DenseBlock, TransitionLayer
from .gnn import DynamicEdgeGcn
from .mlp_conv import MlpConv
from .utility import BnReluConv, BnReluDense

# Make sure that Kedro is aware of custom layers.
if "custom_objects" not in TensorFlowModelDataset.DEFAULT_LOAD_ARGS:
    TensorFlowModelDataset.DEFAULT_LOAD_ARGS["custom_objects"] = {}
TensorFlowModelDataset.DEFAULT_LOAD_ARGS["custom_objects"].update(
    {
        "MlpConv": MlpConv,
        "DenseBlock": DenseBlock,
        "TransitionLayer": TransitionLayer,
        "AssociationLayer": AssociationLayer,
        "BnReluConv": BnReluConv,
        "BnReluDense": BnReluDense,
        "DynamicEdgeGcn": DynamicEdgeGcn,
    }
)
