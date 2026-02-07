import torch
import torch.nn.functional as functional

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
