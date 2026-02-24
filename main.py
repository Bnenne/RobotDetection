import sys
from termcolor import colored

from cli.parser import parse
from cli.types import *
from cli.detector import Detector
from cli.reid import ReID
from help import HELP_TEXT
from hyperparameters import get_hyperparameters

def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_TEXT)
        sys.exit(0)

    args = sys.argv[1:]

    hyperparameters = None

    if "--wandb" in args:
        wandb_index = args.index("--wandb")

        if wandb_index + 1 >= len(args):
            raise ValueError(colored("WandB run path is required", "red"))

        run_path = args[wandb_index + 1]
        hyperparameters = get_hyperparameters(run_path)

        del args[wandb_index:wandb_index + 2]

    parsed = parse(args)

    model = parsed.model
    action = parsed.action
    options = parsed.options

    if hyperparameters is not None:
        options.update(hyperparameters)

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
        model_config.train()
    elif action == Action.val:
        model_config.validate()
    else:
        raise ValueError(colored("Invalid action", "red"))

if __name__ == "__main__":
    main()