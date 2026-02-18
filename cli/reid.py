from typing import Any
import torch
from torch.utils.data import DataLoader
from termcolor import colored

from .parser import add_defaults
from reid.reid import Model
from reid.loss import calculate_loss
from reid.sampler import PKSampler
from reid.utils import validate
from reid.dataset import load_dataset

from cli.types import BaseModelConfig, Action

class ReID(BaseModelConfig):
    def build(self, action: Action, options: dict[str, Any]) -> None:
        self.action = action
        self.options = add_defaults(options)

    def train(self) -> None:
        options = self.options
        device = torch.device(options["device"])

        train_ds, val_ds = load_dataset(options["data"])

        sampler = PKSampler(train_ds.targets, p=options["p"], k=options["k"])
        train_loader = DataLoader(
            train_ds,
            batch_size=options["batch"],
            sampler=sampler
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=options["batch"],
            shuffle=False
        )

        model = Model(num_classes=len(train_ds.classes)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        best_rank1 = 0.0
        best_state = None

        for epoch in range(options["epochs"]):
            model.train()
            total_loss = 0

            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                emb, logits = model(imgs)
                loss = calculate_loss(logits, emb, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            rank1, _ = validate(model, val_loader, device)

            if rank1 > best_rank1:
                best_rank1 = rank1
                best_state = model.state_dict()

            print(
                colored(f"Epoch {epoch}: ", "blue"),
                colored(f"train_loss={total_loss:.3f} ", "yellow"),
                colored(f"rank1={rank1:.3f}", "green")
            )

        # Save best model
        save_path = f"../{options['destination']}/{options['project']}.pth"

        if best_state is not None:
            torch.save(best_state, save_path)
        else:
            torch.save(model.state_dict(), save_path)

        print(colored(f"Saved model to {save_path}", "green"))
