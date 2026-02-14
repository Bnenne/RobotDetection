from typing import Any
from ultralytics import YOLO
import torch

from cli.parser import add_defaults
from cli.types import BaseModelConfig, Action


class Detector(BaseModelConfig):
    def build(self, action: Action, options: dict[str, Any]):
        self.action = action
        self.options = add_defaults(options)

    def train(self):
        options = self.options

        device = torch.device(options["device"])

        model = YOLO(options["model"])

        model.train(
            data="dataset.yaml",
            epochs=50,
            imgsz=768,
            batch=8,
            device=device,
            project="runs/detect",
            name="robot_detector",
            workers=8,
            patience=50,
            pretrained=True,
            verbose=True
        )

        model.val()

        print("Training completed")