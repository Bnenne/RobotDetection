from typing import Any
import torch, wandb, os
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

    def train(self) -> dict[str, Any]:
        global rank1, avg_loss, epoch
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
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=options.get("lr", 3e-4)
        )

        best_rank1 = 0.0
        best_state = None

        for epoch in range(options["epochs"]):
            model.train()
            total_loss = 0.0

            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                emb, logits = model(imgs)
                loss = calculate_loss(logits, emb, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            rank1, _ = validate(model, val_loader, device)

            if rank1 > best_rank1:
                best_rank1 = rank1
                best_state = model.state_dict()

            print(
                colored(f"Epoch {epoch}: ", "blue"),
                colored(f"train_loss={avg_loss:.4f} ", "yellow"),
                colored(f"rank1={rank1:.4f}", "green")
            )

        save_dir = f"../{options['destination']}"
        os.makedirs(save_dir, exist_ok=True)

        best_path = os.path.join(save_dir, f"{options['project']}_best.pth")
        last_path = os.path.join(save_dir, f"{options['project']}_last.pth")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_rank1": best_rank1
        }, last_path)

        if best_state is not None:
            torch.save(best_state, best_path)
        else:
            torch.save(model.state_dict(), best_path)

        print(colored(f"Saved best model to {best_path}", "green"))
        print(colored(f"Saved last model to {last_path}", "green"))

        best_artifact = wandb.Artifact(
            name=f"{options['project']}-best",
            type="model"
        )
        best_artifact.add_file(best_path)
        wandb.log_artifact(best_artifact)

        last_artifact = wandb.Artifact(
            name=f"{options['project']}-last",
            type="model"
        )
        last_artifact.add_file(last_path)
        wandb.log_artifact(last_artifact)

        return {
            "final_rank1": rank1,
            "best_rank1": best_rank1,
            "final_train_loss": avg_loss
        }