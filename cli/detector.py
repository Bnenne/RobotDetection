from typing import Any
from ultralytics import YOLO
import cv2
import torch
from termcolor import colored

from cli.parser import add_defaults
from cli.types import BaseModelConfig, Action


class Detector(BaseModelConfig):
    def build(self, action: Action, options: dict[str, Any]):
        self.action = action
        self.options = add_defaults(options)

    def train(self):
        options = self.options

        device = torch.device(options["device"])

        model = YOLO(options["model"] + ".pt")

        model.train(
            data=options["data"],
            epochs=options["epochs"],
            imgsz=options["images"],
            batch=options["batch"],
            device=device,
            project=options["destination"] + options["project"],
            name=["robot_detector"],
            workers=options["workers"],
            patience=options["patience"],
            pretrained=options["pretrained"],
            verbose=options["verbose"]
        )

        model.val()

        print(colored("Training completed", "green"))

    def validate(self):
        options = self.options

        model = YOLO(options["model"] + ",pt")

        video_path = options["data"]
        output_path = options["destination"] + options["project"]

        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                tracker=options["tracker"] + ".yaml",  # enables tracker
                persist=True,
                verbose=options["verbose"]
            )

            annotated_frame = results[0].plot()

            out.write(annotated_frame)

        cap.release()
        out.release()

        print(f"Saved to {output_path}")