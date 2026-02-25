from enum import Enum
from typing import Any
from pydantic import BaseModel

class Action(Enum):
    train = (1, "Training")
    val = (2, "Validation")

    def __new__(cls, code: int, label: str) -> "Action":
        obj = object.__new__(cls)
        obj._value_ = code
        obj.label = label
        return obj

class Model(Enum):
    robot = (1, "RobotDetection")
    reid = (2, "ReID")

    def __new__(cls, code: int, label: str) -> "Model":
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

    def train(self) -> dict[str, Any]:
        pass

    def validate(self) -> None:
        pass