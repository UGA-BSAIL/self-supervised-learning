"""
Loads/stores Pytorch models.
"""


import torch
from kedro.io import AbstractVersionedDataSet
from kedro.io import Version
from pathlib import PurePosixPath
from typing import Optional, Dict, Any
from torch import nn


class PytorchModelDataSet(AbstractVersionedDataSet[nn.Module, nn.Module]):
    """
    Loads/stores Pytorch models.
    """

    def __init__(
        self, filepath: PurePosixPath, version: Optional[Version] = None
    ):
        """
        Args:
            filepath: The path to the Torch model (.pth) file.
            version: The version information for the `DataSet`.

        """
        super().__init__(PurePosixPath(filepath), version)

    def _load(self) -> nn.Module:
        """
        Returns:
            The loaded model.

        """
        return torch.load(self._get_load_path().as_posix())

    def _save(self, model: nn.Module) -> None:
        """
        Args:
            model: The model to save.

        """
        torch.save(model, self._get_save_path().as_posix())

    def _describe(self) -> Dict[str, Any]:
        return {}
