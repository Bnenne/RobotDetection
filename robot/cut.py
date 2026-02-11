from ultralytics import YOLO
import cv2, os

directory_path = '../videos'

entries = os.listdir(directory_path)

videos = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]

model = YOLO("../archive/RobotDetection_v2.pt")

for video in videos:
    video_path = f"videos/{video}"

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    x = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if x % 300 == 0:
            # Run inference
            results = model.track(
                frame,
                tracker="bytetrack.yaml",  # enables ByteTrack
                persist=True,
                verbose=False
            )

            for i, result in enumerate(results):
                boxes = result.boxes
                for j, box in enumerate(boxes):
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x_min, y_min, x_max, y_max = coords

                    crop_img = frame[y_min:y_max, x_min:x_max]

                    crop_filename = f"{video}_{x}_{i}_{j}.png"
                    cv2.imwrite(f"cropped/{crop_filename}", crop_img)
                    print(f"Saved {crop_filename}")
        x += 1

    cap.release()