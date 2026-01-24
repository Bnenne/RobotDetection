from ultralytics import YOLO
import torch

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load a pretrained YOLO model
    model = YOLO("yolo26n.pt")

    # Train the model
    model.train(
        data="dataset.yaml", # path to dataset yaml
        epochs=3, # increase if dataset is small
        imgsz=640,
        batch=16, # lower if you get OOM errors
        device=device,
        project="runs/detect", # output directory
        name="robot_detector", # experiment name
        workers=4,
        patience=20,
        pretrained=False,
        verbose=True
    )

    # Evaluate on validation and test set
    model.val()

train()