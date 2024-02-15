from typing import *

import torch

__all__ = ["denormalize", "Denormalize"]


def denormalize(
    x: torch.Tensor,
    mean: Iterable[float],
    std: Iterable[float],
) -> torch.Tensor:
    # 3, H, W, B
    ten = x.clone().detach().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)


class Denormalize(object):
    def __init__(self, mean: Iterable[float], std: Iterable[float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return denormalize(x, self.mean, self.std)
