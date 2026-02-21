from termcolor import colored
import wandb, sys

from cli.detector import Detector
from cli.parser import parse
from cli.reid import ReID
from cli.types import Model, BaseModelConfig, Action

project = sys.argv[-1]

run = wandb.init(project=project)
c = run.config

if str(c.model) == "robot":
    args = [
        "robot", "train",
        "-m", "yolo26s.pt",
        "-s", str(c.dataset),
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
elif str(c.model) == "reid":
    args = [
        "reid", "train",
        "-s", str(c.dataset),
        "-b", str(c.batch),
        "-e", str(c.epochs),
        "-lr", str(c.lr),
        "-pk", str(c.p), str(c.k),
        "-dv", str(c.device),
        "-n", str(c.project),
        "-d", str(c.destination),
    ]
else:
    raise ValueError(colored("Invalid model", "red"))

parsed = parse(args)

model = parsed.model
action = parsed.action
options = parsed.options

if model == Model.robot:
    model_config = Detector()
elif model == Model.reid:
    model_config = ReID()
else:
    raise ValueError(colored("Invalid model", "red"))

print(colored(f"Running {action.label} on {model.label}", "green"))

if not isinstance(model_config, BaseModelConfig):
    raise ValueError(colored("Invalid model config", "red"))

model_config.build(action, options)

if action == Action.train:
    metrics = model_config.train()
elif action == Action.val:
    metrics = model_config.validate()
else:
    raise ValueError(colored("Invalid action", "red"))

wandb.log(metrics)
run.finish(0)