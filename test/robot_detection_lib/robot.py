from typing import Any
from numpy import ndarray, dtype
from ultralytics import YOLO
from ultralytics.engine.results import Results

from robot_detection_lib.model_base import BaseModel


class Robot(BaseModel):
    def __init__(self, model_path: str, device: str):
        super().__init__(model_path, device)
        self.model: YOLO | None = None

        self._load()

    def _load(self):
        self.model = YOLO(self.model_path)

    def track(self, frame: ndarray[tuple[Any, ...], dtype[Any]]) -> list[Results]:
        return self.model.track(
                frame,
                tracker="botsort.yaml",
                persist=True,
                verbose=False,
                device=self.device
            )