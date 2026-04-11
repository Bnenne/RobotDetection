from typing import Any
from numpy import ndarray, dtype
import torch
import torch.nn.functional as F
from torchvision import transforms

from robot_detection_lib.model_base import BaseModel, ReIDModel


class ReID(BaseModel):
    def __init__(self, model_path: str, device: str):
        super().__init__(model_path, device)
        self.model: ReIDModel | None = None

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self._load()

    def _load(self):
        weights = torch.load(self.model_path, map_location=self.device)

        if "model_state_dict" in weights:
            state = weights["model_state_dict"]
        else:
            state = weights

        num_classes = state["classifier.weight"].shape[0]

        self.model = ReIDModel(num_classes=num_classes)
        self.model.load_state_dict(state)
        self.model.to(self.device)

    def embed(self, image: ndarray[tuple[Any, ...], dtype[Any]]) -> torch.Tensor:

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding, _ = self.model(tensor)

        embedding = embedding.squeeze(0)

        # normalize for cosine similarity
        embedding = F.normalize(embedding, dim=0)

        return embedding

    def embed_batch(self, images: list[ndarray]) -> torch.Tensor:

        tensors = [self.transform(img) for img in images]
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            embeddings, _ = self.model(batch)

        embeddings = F.normalize(embeddings, dim=1)

        return embeddings

    def similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:

        return float(F.cosine_similarity(emb1, emb2, dim=0))

    def update_vector(
        self,
        current_vector: torch.Tensor,
        new_vector: torch.Tensor,
        count: int
    ) -> torch.Tensor:

        updated = (current_vector * count + new_vector) / (count + 1)

        return F.normalize(updated, dim=0)