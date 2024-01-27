from abc import ABC, abstractmethod

import torch
from torch import nn

__all__ = ["layer_of", "AbstractFeatureExtractor"]


def layer_of(module: nn.Module, pattern: str) -> nn.Module:
    parts = pattern.split(".")
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


class AbstractFeatureExtractor(ABC):
    def __init__(self, *modules: nn.Module) -> None:
        for module in modules:
            module.requires_grad_(False)

    @abstractmethod
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract(x)
