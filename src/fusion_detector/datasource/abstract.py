import math
import pathlib
from abc import ABC, abstractmethod
from typing import *

import torch
from PIL.Image import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

__all__ = [
    "AbstractDataSource",
    "CommonDataSourceTemplate",
    "JaggedDataset",
    "RepeatDataset",
]


class AbstractDataSource(ABC):
    @property
    @abstractmethod
    def trainset(self) -> DataLoader: ...

    @property
    @abstractmethod
    def testset(self) -> DataLoader: ...


class CommonDataSourceTemplate(AbstractDataSource):
    def __init__(
        self,
        dataset_class: type,
        batch_size: int,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        num_workers: int = 0,
    ) -> None:
        raw_trainset = dataset_class(
            train=True,
            transform=transform,
            target_transform=target_transform,
        )
        raw_testset = dataset_class(
            train=False,
            transform=transform,
            target_transform=target_transform,
        )
        self._trainloader = DataLoader(
            raw_trainset,
            batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        self._testloader = DataLoader(
            raw_testset,
            batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    @property
    def trainset(self) -> DataLoader:
        return self._trainloader

    @property
    def testset(self) -> DataLoader:
        return self._testloader


class JaggedDataset(Dataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        self._underlying = sorted(datasets, key=len)
        self._underlying_count = len(self._underlying)
        self._len = sum(map(len, self._underlying))

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Tuple[Image, int]:
        return self._itemof(0, index)

    def _itemof(self, target: int, index: int) -> Tuple[Image, int]:
        minlen = len(self._underlying[target])
        remaining_targets = self._underlying_count - target
        if index < minlen * remaining_targets:
            prefix = index // remaining_targets
            suffix = index % remaining_targets
            return self._underlying[target + suffix][prefix]
        return self._itemof(target + 1, index - minlen)


class RepeatDataset(Dataset):
    def __init__(self, dataset: Dataset, repeat: int) -> None:
        self._underlying = dataset
        self._repeat = repeat
        self._realen = len(dataset)
        self._len = len(dataset) * repeat

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Tuple[Image, int]:
        return self._underlying[index % self._realen]
