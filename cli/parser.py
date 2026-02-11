from sympy.strategies.core import switch

from cli.types import *


def parse(args: list[str]) -> ParseResult:

    model: Model
    action: Action

    config = {}

    if len(args) < 2:
        raise ValueError("Not enough arguments")

    match args[0]:
        case "robot": model = Model.robot
        case "reid": model = Model.reid
        case _: raise ValueError("Invalid model")

    match args[1]:
        case "train": action = Action.train
        case "val": action = Action.val
        case _: raise ValueError("Invalid function")

    options = args[2:]

    if len(options) > 0:
        i = 0
        while i < len(options):
            arg = options[i]
            match arg:
                # Destination directory
                case "-d":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError("Destination directory is required")
                    config["destination"] = options[i + 1]
                    i += 2

                # Training set directory
                case "-s":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError("Data set directory is required")
                    config["data"] = options[i + 1]
                    i += 2

                # Epochs
                case "-e":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError("Epochs is required")
                    try:
                        config["epochs"] = int(options[i + 1])
                    except ValueError:
                        raise ValueError("Epochs must be an integer")
                    i += 2

                # Images
                case "-i":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError("Images is required")
                    try:
                        config["images"] = int(options[i + 1])
                    except ValueError:
                        raise ValueError("Images must be an integer")
                    i += 2

                # Batch
                case "-b":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError("Batch is required")
                    try:
                        config["batch"] = int(options[i + 1])
                    except ValueError:
                        raise ValueError("Batch must be an integer")
                    i += 2

                # Device
                case "-dv":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError("Device is required")
                    config["device"] = options[i + 1]
                    i += 2

                # Pretrained
                case "-p":
                    config["pretrained"] = True
                    i += 1

                # Verbose
                case "-v":
                    config["verbose"] = True
                    i += 1

                # Workers
                case "-w":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError("Workers is required")
                    try:
                        config["workers"] = int(options[i + 1])
                    except ValueError:
                        raise ValueError("Workers must be an integer")
                    i += 2

                # Patience
                case "-p":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError("Patience is required")
                    try:
                        config["patience"] = int(options[i + 1])
                    except ValueError:
                        raise ValueError("Patience must be an integer")
                    i += 2

                # Project name
                case "-n":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError("Project name is required")
                    config["project"] = options[i + 1]
                    i += 2

                case "-m":
                    if i + 1 >= len(options) or options[i + 1].startswith("-"):
                        raise ValueError("Model is required")
                    config["model"] = options[i + 1]

                # Error case
                case _:
                    raise ValueError(f"Invalid argument {arg}")

    return ParseResult(
        model=model,
        action=action,
        options=config,
    )
