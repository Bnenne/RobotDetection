import json
from termcolor import colored

from .types import *


def parse(args: list[str]) -> ParseResult:
    model: Model
    action: Action

    config = {}

    if len(args) < 2:
        raise ValueError(colored("Not enough arguments", "red"))

    match args[0]:
        case "robot": model = Model.robot
        case "reid": model = Model.reid
        case _: raise ValueError(colored("Invalid model", "red"))

    match args[1]:
        case "train": action = Action.train
        case "val": action = Action.val
        case _: raise ValueError(colored("Invalid function", "red"))

    options = args[2:]

    if len(options) > 0:
        i = 0
        while i < len(options):
            arg = options[i]
            match arg:
                # Destination directory
                case "-d":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError(colored("Destination directory is required", "red"))
                    config["destination"] = options[i + 1]
                    i += 2

                # Training set directory
                case "-s":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError(colored("Data set directory is required", "red"))
                    config["data"] = options[i + 1]
                    i += 2

                # Epochs
                case "-e":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError(colored("Epochs is required", "red"))
                    try:
                        config["epochs"] = int(options[i + 1])
                    except ValueError:
                        raise ValueError(colored("Epochs must be an integer", "red"))
                    i += 2

                # Images
                case "-i":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError(colored("Images is required", "red"))
                    try:
                        config["images"] = int(options[i + 1])
                    except ValueError:
                        raise ValueError(colored("Images must be an integer", "red"))
                    i += 2

                # Batch
                case "-b":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError(colored("Batch is required", "red"))
                    try:
                        config["batch"] = int(options[i + 1])
                    except ValueError:
                        raise ValueError(colored("Batch must be an integer", "red"))
                    i += 2

                # Device
                case "-dv":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError(colored("Device is required", "red"))
                    config["device"] = options[i + 1]
                    i += 2

                # Pretrained
                case "-pr":
                    config["pretrained"] = True
                    i += 1

                # Verbose
                case "-v":
                    config["verbose"] = True
                    i += 1

                # Cosine learning rate
                case "-cl":
                    config["cos_lr"] = True
                    i += 1

                # Workers
                case "-w":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError(colored("Workers is required", "red"))
                    try:
                        config["workers"] = int(options[i + 1])
                    except ValueError:
                        raise ValueError(colored("Workers must be an integer", "red"))
                    i += 2

                # Patience
                case "-p":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError(colored("Patience is required", "red"))
                    try:
                        config["patience"] = int(options[i + 1])
                    except ValueError:
                        raise ValueError(colored("Patience must be an integer", "red"))
                    i += 2

                # Project name
                case "-n":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError(colored("Project name is required", "red"))
                    config["project"] = options[i + 1]
                    i += 2

                # Model (yolo26s, yolo11n, etc.)
                case "-m":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError(colored("Model is required", "red"))
                    config["model"] = options[i + 1]
                    i += 2

                # p and k values for PKSampler
                case "-pk":
                    if i + 2 >= len(options) or options[i + 1].startswith("-") or options[i + 2].startswith("-"):
                        raise ValueError(colored("p and k values are required", "red"))
                    config["p"] = int(options[i + 1])
                    config["k"] = int(options[i + 2])
                    i += 3

                # Persist
                case "-ps":
                    config["persist"] = True
                    i += 1

                # Tracker ("botsort" or "bytetrack")
                case "-tr":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError(colored("Tracker is required", "red"))
                    config["tracker"] = options[i + 1]
                    i += 2

                # Learning rate
                case "-lr":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError(colored("Learning rate is required", "red"))
                    try:
                        config["lr0"] = float(options[i + 1])
                    except ValueError:
                        raise ValueError(colored("Learning rate must be an number", "red"))
                    i += 2

                # Weighted decay
                case "-wd":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError(colored("Weight decay is required", "red"))
                    try:
                        config["weight_decay"] = float(options[i + 1])
                    except ValueError:
                        raise ValueError(colored("Weight decay must be an number", "red"))
                    i += 2

                # Weighted decay
                case "-mm":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError(colored("Momentum is required", "red"))
                    try:
                        config["momentum"] = float(options[i + 1])
                    except ValueError:
                        raise ValueError(colored("Momentum must be an number", "red"))
                    i += 2

                # Optimizer
                case "-opt":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError(colored("Optimizer is required", "red"))
                    config["optimizer"] = options[i + 1]
                    i += 2

                # Error case
                case _:
                    raise ValueError(colored(f"Invalid argument {arg}", "red"))

    return ParseResult(
        model=model,
        action=action,
        options=config,
    )

def add_defaults(config: dict[str, Any]) -> dict[str, Any]:
    with open("cli/default.json", "r") as f:
        defaults = json.load(f)

    for key in defaults:
        if key not in config:
            config[key] = defaults[key]
        print(colored(f"{key}:", "blue"), colored(f"{config[key]}", "green"))

    return config