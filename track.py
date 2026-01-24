from ultralytics import YOLO
import cv2

# Loading trained model
model = YOLO("RobotDetection_v1.pt")

# Video paths
video_path = "test_track.mp4"
output_path = "test_track_botsort.mp4"

cap = cv2.VideoCapture(video_path)

# Get video info
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.track(
        frame,
        tracker="botsort.yaml",
        persist=True,
        verbose=False
    )

    # Annotate frame
    annotated_frame = results[0].plot()

    # Write the annotated frame
    out.write(annotated_frame)

# Stop video and tracking
cap.release()
out.release()