from typing import Any
from ultralytics import YOLO, settings
import cv2, torch, os
from termcolor import colored

from cli.parser import add_defaults
from cli.types import BaseModelConfig, Action
from bumper.ocr import OCR
from bumper.similarity import visual_similarity
from bumper.track import *


class BumperReader(BaseModelConfig):
    def build(self, action: Action, options: dict[str, Any]) -> BaseModelConfig:
        self.action = action
        self.options = add_defaults(options)
        settings.update({"wandb": False})
        return self

    def _load_models(self) -> tuple[YOLO, YOLO, OCR]:
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
        print(colored("Loading OCR model", "green"))
        ocr = OCR(gpu=(self._device == "cuda"))

        return robot_model, bumper_model, ocr

    def validate(self) -> None:
        options = self.options
        robot_model, bumper_model, ocr = self._load_models()

        video_path = options["data"]
        if not os.path.isfile(video_path):
            raise ValueError(colored("Video file does not exist", "red"))
        if not video_path.endswith(".mp4"):
            raise ValueError(colored("Video path must end with .mp4", "red"))

        destination_dir = options["destination"]
        os.makedirs(destination_dir, exist_ok=True)
        output_path = os.path.join(destination_dir, f"{options['project']}.mp4")

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        teams = [8825, 935, 9410, 3160, 1987, 5801]

        # track_id -> { team_number -> cumulative similarity score }
        track_votes: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))

        # frame_number -> list of robot annotation dicts (boxes, labels, bumpers)
        frame_data: dict[int, list[dict]] = {}

        # ── Pass 1: inference + vote accumulation ─────────────────────────────────
        print(colored("\nPass 1: collecting detections", "green"))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"\r{colored(f'Frame: {frame_number}/{max_frames}', 'blue')}", end="", flush=True)

            robot_results = robot_model.track(
                frame,
                tracker=options["tracker"] + ".yaml",
                persist=options["persist"],
                verbose=False,
                device=self._device,
            )

            frame_robots: list[dict] = []
            boxes = robot_results[0].boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1 = max(0, x1);
                    y1 = max(0, y1)
                    x2 = min(width, x2);
                    y2 = min(height, y2)

                    track_id = int(box.id[0]) if box.id is not None else -1
                    robot_crop = frame[y1:y2, x1:x2]

                    bumper_detections: list[dict] = []

                    if robot_crop.size > 0:
                        bumper_results = bumper_model.predict(
                            robot_crop, verbose=False, device=self._device
                        )
                        bumper_boxes = bumper_results[0].boxes

                        if bumper_boxes is not None and len(bumper_boxes) > 0:
                            for b_box in bumper_boxes:
                                bx1, by1, bx2, by2 = map(int, b_box.xyxy[0].tolist())
                                fx1 = x1 + bx1;
                                fy1 = y1 + by1
                                fx2 = x1 + bx2;
                                fy2 = y1 + by2
                                conf = float(b_box.conf[0])

                                bumper_crop = robot_crop[by1:by2, bx1:bx2]
                                ocr_results = ocr.read(bumper_crop) if bumper_crop.size > 0 else []

                                team_number = None
                                b_label = f"? ({conf:.2f})"

                                if ocr_results:
                                    team_number, ocr_conf = ocr_results[0]
                                    b_label = f"#{team_number} ({ocr_conf:.2f})"

                                # accumulate votes for this track across all frames
                                if team_number is not None and track_id != -1:
                                    scores = {
                                        t: visual_similarity(t, team_number)
                                        for t in teams
                                    }
                                    best_team = max(scores, key=scores.get)
                                    track_votes[track_id][best_team] += scores[best_team]

                                bumper_detections.append({
                                    "box": (fx1, fy1, fx2, fy2),
                                    "label": b_label,
                                })

                    frame_robots.append({
                        "track_id": track_id,
                        "box": (x1, y1, x2, y2),
                        "bumpers": bumper_detections,
                    })

            # suppress duplicate boxes before storing
            frame_data[frame_number] = suppress_duplicate_boxes(frame_robots)

        cap.release()

        # ── Merge re-identified tracks ────────────────────────────────────────────
        print(colored("\nMerging re-identified tracks...", "green"))

        mapping = merge_lost_tracks(track_votes, frame_data)
        track_votes = apply_track_merge(track_votes, mapping)

        print(colored(f"Track merge map: {mapping}", "cyan"))

        # ── Resolve: assign one unique team per canonical track_id ────────────────
        print(colored("Resolving team assignments...", "green"))

        track_best = {
            tid: max(votes.items(), key=lambda x: x[1])
            for tid, votes in track_votes.items()
        }
        sorted_tracks = sorted(track_best.items(), key=lambda x: x[1][1], reverse=True)

        team_assignments: dict[int, int] = {}
        assigned_teams: set[int] = set()

        for track_id, (best_team, _) in sorted_tracks:
            for team, _ in sorted(track_votes[track_id].items(), key=lambda x: x[1], reverse=True):
                if team not in assigned_teams:
                    team_assignments[track_id] = team
                    assigned_teams.add(team)
                    break

        print(colored(f"Final assignments: {team_assignments}", "cyan"))

        # ── Pass 2: re-read video and draw with resolved labels ───────────────────
        print(colored("Pass 2: writing annotated video", "green"))

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"\r{colored(f'Frame: {frame_number}/{max_frames}', 'blue')}", end="", flush=True)

            annotated = frame.copy()

            for robot in frame_data.get(frame_number, []):
                track_id = robot["track_id"]
                x1, y1, x2, y2 = robot["box"]

                # remap to canonical track_id before looking up team
                canonical = mapping.get(track_id, track_id)
                team = team_assignments.get(canonical)

                if team is not None:
                    label = f"Robot {canonical}: #{team}"
                elif canonical != -1:
                    label = f"Robot {canonical}"
                else:
                    label = "Robot"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

                for bumper in robot["bumpers"]:
                    bx1, by1, bx2, by2 = bumper["box"]
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    cv2.putText(
                        annotated, bumper["label"], (bx1, by1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                    )

            out.write(annotated)

        cap.release()
        out.release()

        print(colored(f"\nSaved to {output_path}", "green"))

    def train(self) -> None:
        pass