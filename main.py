import sys
from termcolor import colored

from cli.parser import parse
from cli.types import *
from cli.detector import Detector
from cli.reid import ReID

args = sys.argv[1:]

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

if not isinstance(model_config, BaseModelConfig):
    raise ValueError(colored("Invalid model config", "red"))

model_config.build(action, options)

if action == Action.train:
    model_config.train()
elif action == Action.val:
    model_config.validate()
else:
    raise ValueError(colored("Invalid action", "red"))