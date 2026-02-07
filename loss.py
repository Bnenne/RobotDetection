import torch
import torch.nn.functional as functional

def batch_hard_triplet_loss(embeddings, labels, margin=0.3):
    dist = torch.cdist(embeddings, embeddings)

    loss = 0.0
    for i in range(len(labels)):
        pos = dist[i][labels == labels[i]]
        neg = dist[i][labels != labels[i]]

        hardest_pos = pos.max()
        hardest_neg = neg.min()

        loss += functional.relu(hardest_pos - hardest_neg + margin)

    return loss / len(labels)
