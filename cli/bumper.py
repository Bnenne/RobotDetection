from typing import Any
from ultralytics import YOLO, settings
import cv2, torch, os
from termcolor import colored

from cli.parser import add_defaults
from cli.types import BaseModelConfig, Action


class BumperReader(BaseModelConfig):
    def build(self, action: Action, options: dict[str, Any]) -> BaseModelConfig:
        self.action = action
        self.options = add_defaults(options)

        settings.update({"wandb": False})

        return self

    def _load_models(self) -> tuple[YOLO, YOLO]:
        options = self.options

        device = options["device"]
        if device == "cuda" and not torch.cuda.is_available():
            print(colored("CUDA not available, switching to CPU", "yellow"))
            device = "cpu"

        self._device = device

        print(colored("Loading robot detection model", "green"))
        robot_model = YOLO(options["model"])

        print(colored("Loading bumper model", "green"))
        bumper_model = YOLO(options["bumper_model"])

        return robot_model, bumper_model

    def validate(self) -> None:
        """
        Runs robot detection on each frame of a video, crops each detected
        robot, then runs the bumper model on the crop to extract the number.
        Saves annotated frames with both robot boxes and bumper number boxes
        into an output video.
        """
        options = self.options

        robot_model, bumper_model = self._load_models()

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

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(colored("Reading started", "green"))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            max_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if frame_number < 1000:
                continue

            if frame_number > 2000:
                break

            frame_count = colored(f"Frame: {frame_number}/{max_frames}", "blue")
            print(f"\r{frame_count}", end="", flush=True)

            annotated = frame.copy()

            # ── Step 1: detect robots ──────────────────────────────────────
            robot_results = robot_model.track(
                frame,
                tracker=options["tracker"] + ".yaml",
                persist=options["persist"],
                verbose=False,
                device=self._device,
            )

            boxes = robot_results[0].boxes
            if boxes is None or len(boxes) == 0:
                out.write(annotated)
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # clamp to frame bounds
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(width, x2); y2 = min(height, y2)

                robot_crop = frame[y1:y2, x1:x2]

                if robot_crop.size == 0:
                    continue

                # draw robot box on annotated frame
                track_id = int(box.id[0]) if box.id is not None else -1
                label    = f"Robot {track_id}" if track_id != -1 else "Robot"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated, label,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

                # ── Step 2: run bumper model on the crop ───────────────────
                bumper_results = bumper_model.predict(
                    robot_crop,
                    verbose=False,
                    device=self._device,
                )

                bumper_boxes = bumper_results[0].boxes
                if bumper_boxes is None or len(bumper_boxes) == 0:
                    continue

                for b_box in bumper_boxes:
                    bx1, by1, bx2, by2 = map(int, b_box.xyxy[0].tolist())

                    # translate crop-relative coords back to full frame
                    fx1 = x1 + bx1; fy1 = y1 + by1
                    fx2 = x1 + bx2; fy2 = y1 + by2

                    conf  = float(b_box.conf[0])
                    b_label = f"Number {conf:.2f}"

                    cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
                    cv2.putText(
                        annotated, b_label,
                        (fx1, fy1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                    )

            out.write(annotated)

        cap.release()
        out.release()

        print(colored(f"\nSaved to {output_path}", "green"))

    def train(self) -> None:
        pass