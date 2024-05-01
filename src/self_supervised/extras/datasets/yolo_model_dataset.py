"""
Loads/stores YOLO models.
"""

from pathlib import Path, PurePosixPath

from kedro.io import AbstractDataSet, Version
from ultralytics import YOLO


class YoloModelDataSet(AbstractDataSet):
    """
    Loads/stores YOLO models.
    """

    def __init__(self, filepath: PurePosixPath):
        super().__init__()

        self.__filepath = Path(filepath)

    def _load(self):
        return YOLO(self.__filepath)

    def _save(self, model: YOLO):
        model.save(self.__filepath)

    def _describe(self):
        return dict(filepath=self.__filepath)
