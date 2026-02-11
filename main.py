import sys

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
    model_config = Detector
elif model == Model.reid:
    model_config = ReID
else:
    raise ValueError("Invalid model")

if not isinstance(model_config, BaseModelConfig):
    raise ValueError("Invalid model config")

model_config.build(action, options)