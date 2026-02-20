import subprocess
import sys
import wandb

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
    "-pr", str(c.project),
    "-v", "-cl",
]

result = subprocess.run(cmd)
sys.exit(result.returncode)