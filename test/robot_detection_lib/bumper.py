from typing import Any
from numpy import ndarray, dtype
from ultralytics import YOLO
from ultralytics.engine.results import Results

from robot_detection_lib.model_base import BaseModel


class Bumper(BaseModel):
    def __init__(self, model_path: str, device: str):
        super().__init__(model_path, device)
        self.model: YOLO | None = None

        self._load()

    def _load(self):
        self.model = YOLO(self.model_path)

    def predict(self, frame: ndarray[tuple[Any, ...], dtype[Any]]) -> list[Results]:
        return self.model.predict(
                frame,
                verbose=False,
                device=self.device
            )