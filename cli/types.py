from enum import Enum
from typing import Any
from pydantic import BaseModel

class Action(Enum):
    train = (1, "Training")
    val = (2, "Validating")

    def __new__(cls, code: int, label: str):
        obj = object.__new__(cls)
        obj._value_ = code
        obj.label = label
        return obj

class Model(Enum):
    robot = (1, "RobotDetection")
    reid = (2, "ReID")

    def __new__(cls, code: int, label: str):
        obj = object.__new__(cls)
        obj._value_ = code
        obj.label = label
        return obj

class ParseResult(BaseModel):
    model: Model
    action: Action
    options: dict[str, Any]

class BaseModelConfig:
    action: Action = Action.train
    options: dict[str, Any] = {}

    def build(self, action: Action, options: dict[str, Any]):
        pass

    def train(self):
        pass

    def validate(self):
        pass