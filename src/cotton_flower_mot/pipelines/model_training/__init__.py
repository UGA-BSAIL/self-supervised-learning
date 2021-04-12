from kedro.extras.datasets.tensorflow import TensorFlowModelDataset
from spektral.layers.convolutional import GCNConv

from .pipeline import create_pipeline  # NOQA

# Make sure that Kedro is aware of custom layers.
if "custom_objects" not in TensorFlowModelDataset.DEFAULT_LOAD_ARGS:
    TensorFlowModelDataset.DEFAULT_LOAD_ARGS["custom_objects"] = []
TensorFlowModelDataset.DEFAULT_LOAD_ARGS["custom_objects"].update(
    {"GCNConv": GCNConv}
)
