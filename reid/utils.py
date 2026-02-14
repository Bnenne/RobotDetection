import torch
import torch.nn.functional as functional
import numpy as np
from redis.redis import Redis

def extract_embeddings(model, loader, device):
    model.eval()
    all_emb = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            emb, _ = model(imgs)
            emb = functional.normalize(emb, dim=1)

            all_emb.append(emb.cpu())
            all_labels.append(labels)

    return torch.cat(all_emb), torch.cat(all_labels)

def rank1_accuracy(emb, labels):
    sim = torch.mm(emb, emb.t())

    correct = 0
    for i in range(len(labels)):
        sim[i, i] = -1
        best = sim[i].argmax()
        correct += (labels[i] == labels[best]).item()

    return correct / len(labels)

def cache_embeddings(emb, labels, dataset, ttl=600):
    redis = Redis()

    emb_np = emb.numpy().astype(np.float32)
    labels_np = labels.numpy()

    redis.store_vector(emb_np, labels_np, dataset, ttl=ttl)

def validate(model, loader, device, cache_ttl=600):
    emb, labels = extract_embeddings(model, loader, device)
    rank1 = rank1_accuracy(emb, labels)

    return rank1, (emb, labels, cache_ttl)
