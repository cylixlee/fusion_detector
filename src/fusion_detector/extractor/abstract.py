from abc import ABC, abstractmethod
from typing import *

import torch
from torch import nn

from ..misc.hook import LayerOutputCollectedModuleWrapper

__all__ = [
    "AbstractFeatureExtractor",
    "IntermediateLayerFeatureExtractor",
]


class AbstractFeatureExtractor(ABC):
    @abstractmethod
    def extract(self, x: torch.Tensor) -> torch.Tensor: ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract(x)


class IntermediateLayerFeatureExtractor(AbstractFeatureExtractor):
    def __init__(self, module: nn.Module, layer_pattern: str) -> None:
        self.module_wrapper = LayerOutputCollectedModuleWrapper(module, layer_pattern)
        self.module_wrapper.underlying.requires_grad_(False)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        self.module_wrapper(x)
        return self.module_wrapper.intermediate
