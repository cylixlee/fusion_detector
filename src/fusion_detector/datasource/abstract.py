import pathlib
from abc import ABC, abstractmethod
from typing import *

import torch
from torch.utils.data.dataloader import DataLoader

__all__ = ["AbstractDataSource"]


class AbstractDataSource(ABC):
    @property
    @abstractmethod
    def trainset(self) -> DataLoader: ...

    @property
    @abstractmethod
    def testset(self) -> DataLoader: ...
