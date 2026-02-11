import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Model(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.embedding = nn.Linear(2048, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)

        emb = self.embedding(x)
        emb = F.normalize(emb, dim=1)

        logits = self.classifier(emb)
        return emb, logits
