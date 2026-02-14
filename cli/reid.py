from typing import Any
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader
from termcolor import colored

from .parser import add_defaults
from reid.reid import Model
from reid.loss import batch_hard_triplet_loss
from reid.sampler import PKSampler
from reid.utils import validate, cache_embeddings
from reid.dataset import load_dataset
from redis.redis import Redis


from cli.types import BaseModelConfig, Action

class ReID(BaseModelConfig):
    def build(self, action: Action, options: dict[str, Any]):
        self.action = action
        self.options = add_defaults(options)

    def train(self):
        options = self.options

        device = torch.device(options["device"])

        train_ds, val_ds = load_dataset(options["data"])

        sampler = PKSampler(train_ds.targets, p=options["p"], k=options["k"])
        train_loader = DataLoader(train_ds, batch_size=options["batch"], sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=options["batch"], shuffle=False)

        model = Model(num_classes=len(train_ds.classes)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        best_rank1 = 0.0

        redis = Redis()
        redis.flushdb()

        for epoch in range(options["epochs"]):
            model.train()
            total_loss = 0

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                emb, logits = model(imgs)

                loss_id = functional.cross_entropy(logits, labels)
                loss_tri = batch_hard_triplet_loss(emb, labels)
                loss = loss_id + loss_tri

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            rank1, emb = validate(
                model,
                val_loader,
                device
            )

            if rank1 > best_rank1:
                best_rank1 = rank1
                cache_embeddings(emb[0], emb[1], val_ds, emb[2])

            print(
                colored(f"Epoch {epoch}: ", "blue"),
                colored(f"train_loss={total_loss:.3f} ", "yellow"),
                colored(f"rank1={rank1:.3f}", "green")
            )

        torch.save(model.state_dict(), f"../{options["destination"]}/{options["project"]}.pth")
        print(colored(f"Saved model to {options["destination"]}/{options["project"]}.pth", "green"))