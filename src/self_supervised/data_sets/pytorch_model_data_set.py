"""
Loads/stores Pytorch models.
"""


from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

import torch
from kedro.io import AbstractVersionedDataSet, Version
from loguru import logger
from torch import nn


class PytorchModelDataSet(AbstractVersionedDataSet[nn.Module, nn.Module]):
    """
    Loads/stores Pytorch models.
    """

    def __init__(
        self,
        filepath: PurePosixPath,
        version: Optional[Version] = None,
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
        load_args = {}
        if not torch.cuda.is_available():
            # In this case, we probably need to map to the CPU.
            logger.warning(
                "CUDA not available, automatically mapping "
                "variables to the CPU."
            )
            load_args["map_location"] = torch.device("cpu")

        return torch.load(self._get_load_path().as_posix(), **load_args)

    def _save(self, model: nn.Module) -> None:
        """
        Args:
            model: The model to save.

        """
        parent_dir = Path(self._get_save_path()).parent
        # Create parent directories if needed.
        parent_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model, self._get_save_path().as_posix())

    def _describe(self) -> Dict[str, Any]:
        return {}
