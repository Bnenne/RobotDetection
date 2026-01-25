from ultralytics import YOLO
import torch

def train():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Load a pretrained YOLO model
        model = YOLO("yolo26s.pt")

        # Train the model
        model.train(
            data="dataset.yaml", # path to dataset yaml
            epochs=50,
            imgsz=768,
            batch=8,
            device=device,
            project="runs/detect", # output directory
            name="robot_detector", # experiment name
            workers=8,
            patience=50,
            pretrained=True,
            verbose=True
        )

        # Evaluate on validation and test set
        model.val()
    except KeyboardInterrupt:
        print("Training interrupted by user.")

train()