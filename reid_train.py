import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from reid import Model
from loss import batch_hard_triplet_loss
from sampler import PKSampler
from validate import extract_embeddings, rank1_accuracy
from dataset import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

train_ds, val_ds = load_dataset()

sampler = PKSampler(train_ds.targets, p=5, k=4)
train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

model = Model(num_classes=len(train_ds.classes)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(30):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        emb, logits = model(imgs)

        loss_id = F.cross_entropy(logits, labels)
        loss_tri = batch_hard_triplet_loss(emb, labels)
        loss = loss_id + loss_tri

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    val_emb, val_labels = extract_embeddings(model, val_loader, device)
    rank1 = rank1_accuracy(val_emb, val_labels)

    print(
        f"Epoch {epoch}: "
        f"train_loss={total_loss:.3f} "
        f"rank1={rank1:.3f}"
    )

torch.save(model.state_dict(), "robot_reid.pth")
print("Saved model to robot_reid.pth")
