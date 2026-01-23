from ultralytics import YOLO
import cv2

# Load your trained YOLO26n model
model = YOLO("best.pt")

# Video source
video_path = "video path"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.track(
        frame,
        tracker="bytetrack.yaml",  # enables ByteTrack
        persist=True
    )

    # Annotate frame
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()