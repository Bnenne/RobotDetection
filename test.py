from ultralytics import YOLO
import torch

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load a pretrained YOLO model
    model = YOLO("best.pt")

    # Evaluate on validation + test set
    model.

test()