"""Stage 6: Run through video with models to gather data."""

import tempfile
import cv2
from typing import Any

from robot_detection_lib.robot import Robot
from robot_detection_lib.bumper import Bumper
from robot_detection_lib.reid import ReID
from robot_detection_lib.ocr import OCR
from robot_detection_lib.utils import suppress_duplicate_boxes


def execute(
    previous_artifacts: dict[str, Any]
) -> dict[str, Any]:
    """Run models and detect robots and bumpers."""
    # TODO: Implement robot and bumper models

    robot, bumper, reid, ocr = _load_models()

    corrected_feeds = ["camera_02.mp4", "camera_04.mp4"]
    corrected_feeds: list[str]

    print(f"Starting models inferences for job TEST")

    video_data: dict[int, dict] = {}

    with tempfile.TemporaryDirectory() as temp_dir:

        for i, feed in enumerate(corrected_feeds):

            cap = cv2.VideoCapture(feed)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            output_path = f"./annotated_{i}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_data: dict[int, list[dict]] = {}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if frame_number % 1 != 0:
                    continue

                print(f"Processing feed {i+1}/{len(corrected_feeds)} - Frame {frame_number}/{max_frames}")

                robot_results = robot.track(frame)

                frame_robots: list[dict] = []
                boxes = robot_results[0].boxes

                annotated = frame.copy()

                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)

                        track_id = int(box.id[0]) if box.id is not None else -1
                        robot_crop = frame[y1:y2, x1:x2]

                        label = f"Robot {track_id}" if track_id != -1 else "Robot"
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            annotated, label,
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                        )

                        bumper_detections: list[dict] = []

                        if robot_crop.size > 0:
                            bumper_results = bumper.predict(robot_crop)

                            bumper_boxes = bumper_results[0].boxes

                            annotated_frame = bumper_results[0].plot()

                            if bumper_boxes is not None and len(bumper_boxes) > 0:
                                for b_box in bumper_boxes:
                                    bx1, by1, bx2, by2 = map(int, b_box.xyxy[0].tolist())
                                    fx1 = x1 + bx1
                                    fy1 = y1 + by1
                                    fx2 = x1 + bx2
                                    fy2 = y1 + by2

                                    bumper_crop = robot_crop[by1:by2, bx1:bx2]
                                    ocr_results = ocr.read(bumper_crop) if bumper_crop.size > 0 else []

                                    if ocr_results:
                                        team_number, ocr_conf = ocr_results[0]
                                        bumper_detections.append({
                                            "box": (fx1, fy1, fx2, fy2),
                                            "number": team_number,
                                            "conf": ocr_conf
                                        })

                                        conf = float(b_box.conf[0])
                                        b_label = f"Number {conf:.2f}: {team_number} {ocr_conf:.2f}"

                                        cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
                                        cv2.putText(
                                            annotated, b_label,
                                            (fx1, fy1 - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                                        )
                                    else:
                                        conf = float(b_box.conf[0])
                                        b_label = f"Number {conf:.2f}: ?"

                                        cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
                                        cv2.putText(
                                            annotated, b_label,
                                            (fx1, fy1 - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                                        )

                        frame_robots.append({
                            "track_id": track_id,
                            "box": (x1, y1, x2, y2),
                            "bumpers": suppress_duplicate_boxes(bumper_detections),
                            "embedding": reid.embed(robot_crop).tolist(),
                        })

                # suppress duplicate boxes before storing
                frame_data[frame_number] = suppress_duplicate_boxes(frame_robots)

                out.write(annotated)

            video_data[i + 1] = frame_data

            cap.release()
            out.release()

    return {
        "video_data": video_data
    }

def _load_models() -> tuple[Robot, Bumper, ReID, OCR]:
    lib_path = "./robot_detection_lib/"
    robot = Robot(lib_path + "robot_best.pt", "cuda")
    bumper = Bumper(lib_path + "bumper_best.pt", "cuda")
    reid = ReID(lib_path + "reid_best.pth", "cuda")
    ocr = OCR("cuda")

    return robot, bumper, reid, ocr
