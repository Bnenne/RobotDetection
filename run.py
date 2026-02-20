import subprocess
import sys
import os
import wandb
from pathlib import Path
import csv

run = wandb.init()
c = run.config

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

result = subprocess.run(cmd)

results_path = Path("runs") / str(c.get("project")) / "train" / "results.csv"

if results_path.exists():
    with open(results_path) as f:
        reader = csv.DictReader(f)
        for step, row in enumerate(reader):
            log_dict = {}
            for k, v in row.items():
                v = v.strip()
                if not v:
                    continue
                try:
                    log_dict[k.strip()] = float(v)
                except ValueError:
                    continue
            if log_dict:
                run.log(log_dict, step=step)

run.finish()
sys.exit(result.returncode)