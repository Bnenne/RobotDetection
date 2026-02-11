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
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 10

    def build(self, action, options):
        pass