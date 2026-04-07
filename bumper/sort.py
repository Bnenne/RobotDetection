import os
import cv2
import torch
from collections import defaultdict
from ultralytics import YOLO
from ocr import OCR
from similarity import visual_similarity

# -----------------------------
# CONFIG
# -----------------------------

VIDEOS = [
    "videos/camera_02(1).mp4",
    "videos/camera_04(1).mp4",
]

ROBOT_MODEL = "robot_best.pt"
BUMPER_MODEL = "bumper_best.pt"

DESTINATION = "new/reid_dataset"

TEAMS = [7072, 5125, 3140, 2190, 5005, 8772]

FRAME_SKIP = 10          # sample every N frames
MIN_OCR_CONF = 0.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(DESTINATION, exist_ok=True)


# -----------------------------
# LOAD MODELS
# -----------------------------

print("Loading robot model")
robot_model = YOLO(ROBOT_MODEL)

print("Loading bumper model")
bumper_model = YOLO(BUMPER_MODEL)

print("Loading OCR")
ocr = OCR(gpu=(DEVICE == "cuda"))


# -----------------------------
# UTILS
# -----------------------------

def save_robot(team, image, counter):
    folder = os.path.join(DESTINATION, str(team))
    os.makedirs(folder, exist_ok=True)

    filename = os.path.join(folder, f"{team}_{counter:06d}.jpg")
    cv2.imwrite(filename, image)


# -----------------------------
# PROCESS VIDEOS
# -----------------------------

image_counter = defaultdict(int)

for video_path in VIDEOS:

    print(f"\nProcessing {video_path}")

    cap = cv2.VideoCapture(video_path)

    frame_id = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        if frame_id % FRAME_SKIP != 0:
            continue

        # -----------------------------
        # ROBOT DETECTION + TRACKING
        # -----------------------------

        robot_results = robot_model.track(
            frame,
            persist=True,
            device=DEVICE,
            verbose=False
        )

        boxes = robot_results[0].boxes

        if boxes is None:
            continue

        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            robot_crop = frame[y1:y2, x1:x2]

            if robot_crop.size == 0:
                continue

            # -----------------------------
            # BUMPER DETECTION
            # -----------------------------

            bumper_results = bumper_model.predict(
                robot_crop,
                device=DEVICE,
                verbose=False
            )

            bumper_boxes = bumper_results[0].boxes

            team_guess = None
            best_conf = 0

            if bumper_boxes is not None:

                for b in bumper_boxes:

                    bx1, by1, bx2, by2 = map(int, b.xyxy[0].tolist())

                    bumper_crop = robot_crop[by1:by2, bx1:bx2]

                    if bumper_crop.size == 0:
                        continue

                    ocr_results = ocr.read(bumper_crop)

                    if not ocr_results:
                        continue

                    team_number, conf = ocr_results[0]

                    if conf < MIN_OCR_CONF:
                        continue

                    scores = {
                        t: visual_similarity(t, team_number)
                        for t in TEAMS
                    }

                    best_team = max(scores, key=scores.get)

                    if scores[best_team] > best_conf:
                        best_conf = scores[best_team]
                        team_guess = best_team

            # -----------------------------
            # SAVE IMAGE
            # -----------------------------

            if team_guess is None:
                team_guess = "unknown"

            image_counter[team_guess] += 1

            save_robot(
                team_guess,
                robot_crop,
                image_counter[team_guess]
            )

    cap.release()


print("\nFinished dataset creation.")