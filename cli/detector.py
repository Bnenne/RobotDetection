from typing import Any
from ultralytics import YOLO, settings
import cv2, torch, os
from termcolor import colored

from cli.parser import add_defaults
from cli.types import BaseModelConfig, Action


class Detector(BaseModelConfig):
    def build(self, action: Action, options: dict[str, Any]) -> None:
        self.action = action
        self.options = add_defaults(options)

    def train(self) -> None:
        options = self.options

        device = options["device"]

        if device == "cuda" and not torch.cuda.is_available():
            print(colored("CUDA not available, switching to CPU", "yellow"))
            device = "cpu"

        print(colored("Loading model", "green"))
        model = YOLO(options["model"])

        print(colored("Training started", "green"))
        model.train(
            data=options["data"],
            epochs=options["epochs"],
            imgsz=options["images"],
            batch=options["batch"],
            device=device,
            project=os.path.join(options["destination"], options["project"]),
            name=options["project"],
            workers=options["workers"],
            patience=options["patience"],
            pretrained=options["pretrained"],
            verbose=options["verbose"],
            lr0=options["lr0"],
            weight_decay=options["weight_decay"],
            momentum=options["momentum"],
            cos_lr=options["cos_lr"]
        )

        print(colored("Saving model", "green"))
        model.val()

        print(colored("Training completed", "green"))

    def validate(self):
        options = self.options

        model = YOLO(options["model"])

        video_path = options["data"]

        if not os.path.isfile(video_path):
            raise ValueError(colored("Video file does not exist", "red"))

        if not video_path.endswith(".mp4"):
            raise ValueError(colored("Video path must end with .mp4", "red"))

        destination_dir = options["destination"]

        os.makedirs(destination_dir, exist_ok=True)

        output_path = os.path.join(
            destination_dir,
            f"{options['project']}.mp4"
        )

        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(colored("Validating started", "green"))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_count = colored(f"Frame: {frame_number}/{max_frames}", "blue")
            print(f"\r{frame_count}", end="", flush=True)

            results = model.track(
                frame,
                tracker=options["tracker"] + ".yaml",  # enables tracker
                persist=options["persist"],
                verbose=options["verbose"]
            )

            annotated_frame = results[0].plot()

            out.write(annotated_frame)

        cap.release()
        out.release()

        print(colored(f"Saved to {output_path}", "green"))