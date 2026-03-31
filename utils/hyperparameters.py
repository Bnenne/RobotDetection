import wandb

def get_hyperparameters(run_path: str) -> dict[str, any]:
    api = wandb.Api()

    run = api.run(run_path)

    config = run.config

    return {
        "lr0": config.get("lr0"),
        "weight_decay": config.get("weight_decay"),
        "momentum": config.get("momentum"),
        "batch": config.get("batch"),
        "optimizer": config.get("optimizer"),
        "images": config.get("imgsz")
    }