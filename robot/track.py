from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

video_path = "../archive/test_track.mp4"
output_path = "test_track_tracked.mp4"

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
        tracker="bytetrack.yaml", # enables ByteTrack
        persist=True,
        verbose=False
    )

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

cap.release()
out.release()

print(f"Saved to {output_path}")