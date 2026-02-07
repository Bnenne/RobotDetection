from ultralytics import YOLO
import torch

def train():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model = YOLO("yolo26s.pt")

        model.train(
            data="dataset.yaml",
            epochs=50,
            imgsz=768,
            batch=8,
            device=device,
            project="runs/detect",
            name="robot_detector",
            workers=8,
            patience=50,
            pretrained=True,
            verbose=True
        )

        model.val()

        print("Training completed")
    except KeyboardInterrupt:
        print("Training interrupted by user")

train()