import subprocess
import sys
import os
import wandb
from pathlib import Path
import csv

run = wandb.init()
c = run.config

# Disable wandb inside the subprocess so YOLO doesn't call wandb.init() again
child_env = os.environ.copy()
child_env["WANDB_DISABLED"] = "true"
child_env["WANDB_MODE"] = "disabled"

cmd = [
    sys.executable, "main.py",
    "robot", "train",
    "-m", "yolo26s.pt",
    "-s", "dataset.yaml",
    "-lr", str(c.lr0),
    "-wd", str(c.weight_decay),
    "-mm", str(c.momentum),
    "-b",  str(c.batch),
    "-opt", str(c.optimizer),
    "-dv", str(c.device),
    "-i",  str(c.imgsz),
    "-e",  str(c.epochs),
    "-w",  str(c.workers),
    "-p",  str(c.patience),
    "-n", str(c.project),
    "-v", "-cl"
]
result = subprocess.run(cmd, env=child_env)

# YOLO saves results to runs/<project>/train/results.csv — log final metrics to wandb
results_path = Path("runs") / str(c.project) / "train" / "results.csv"
if results_path.exists():
    with open(results_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if rows:
        # Log all rows as steps, then summary
        for i, row in enumerate(rows):
            run.log({k.strip(): float(v) for k, v in row.items() if v.strip()}, step=i)

run.finish()
sys.exit(result.returncode)