from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str = "logistic_regression"
    max_iter: int = 200
