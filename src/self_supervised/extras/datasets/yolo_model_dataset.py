"""
Loads/stores YOLO models.
"""

from pathlib import Path, PurePosixPath

from kedro.io import AbstractDataset
from ultralytics import YOLO


class YoloModelDataSet(AbstractDataset):
    """
    Loads/stores YOLO models.
    """

    def __init__(self, filepath: PurePosixPath):
        super().__init__()

        self.__filepath = Path(filepath)

    def load(self):
        return YOLO(self.__filepath)

    def save(self, model: YOLO):
        model.save(self.__filepath)

    def _describe(self):
        return dict(filepath=self.__filepath)
