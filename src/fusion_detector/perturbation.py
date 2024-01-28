from enum import Enum
from typing import *

import numpy as np
import torch
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from cleverhans.torch.attacks.sparse_l1_descent import sparse_l1_descent
from torch import nn

__all__ = [
    "TensorTransformAction",
    "AvailableNorm",
    "fgsm",
    "pgd",
    "sparsel1",
]

TensorTransformAction = Callable[..., torch.Tensor]


class AvailableNorm(Enum):
    L1 = 1
    L2 = 2
    INFINITY = np.inf


def fgsm(
    module: nn.Module,
    epsilon: float = 0.1,
    norm: AvailableNorm = AvailableNorm.INFINITY,
) -> TensorTransformAction:
    def _wrapper(x: torch.Tensor) -> torch.Tensor:
        return fast_gradient_method(module, x, epsilon, norm.value)

    return _wrapper


def pgd(
    module: nn.Module,
    epsilon: float = 0.1,
    step_epsilon: float = 0.05,
    iterations: int = 10,
    norm: AvailableNorm = AvailableNorm.INFINITY,
) -> TensorTransformAction:
    def _wrapper(x: torch.Tensor) -> torch.Tensor:
        return projected_gradient_descent(
            module, x, epsilon, step_epsilon, iterations, norm.value
        )

    return _wrapper


def sparsel1(
    module: nn.Module,
    epsilon: float,
    step_epsilon: float,
    iterations: int,
) -> TensorTransformAction:
    def _wrapper(x: torch.Tensor) -> torch.Tensor:
        return sparse_l1_descent(module, x, epsilon, step_epsilon, iterations)

    return _wrapper