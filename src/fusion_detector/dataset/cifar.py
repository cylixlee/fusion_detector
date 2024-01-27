import math
import os
import pathlib
from typing import *

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

from .abstract import AbstractDataset

__all__ = ["CifarDataset", "CifarBatchDataset"]

SCRIPT_DIRECTORY = pathlib.PurePath(__file__).parent
INNER_PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
SOURCE_DIRECTORY = INNER_PROJECT_DIRECTORY.parent
PROJECT_DIRECTORY = SOURCE_DIRECTORY.parent
DATASET_DIRECTORY = PROJECT_DIRECTORY / "data" / "datasets"


class CifarDataset(AbstractDataset):
    """Thirdparty project `pytorch_cifar10`-compatible CIFAR10 dataset."""

    def __init__(
        self,
        batch_size: int,
        root: str = str(DATASET_DIRECTORY / "CIFAR10"),
        num_workers: int = math.floor(os.cpu_count() / 2),
    ) -> None:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        raw_trainset = datasets.CIFAR10(
            root,
            train=True,
            transform=train_transform,
            download=True,
        )
        raw_testset = datasets.CIFAR10(
            root,
            train=False,
            transform=test_transform,
            download=True,
        )
        self._trainloader = DataLoader(
            raw_trainset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
        )
        self._testloader = DataLoader(
            raw_testset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
        )

    @property
    def trainset(self) -> DataLoader:
        return self._trainloader

    @property
    def testset(self) -> DataLoader:
        return self._testloader


class _CifarBatch(Dataset):
    def __init__(
        self,
        root: str = str(DATASET_DIRECTORY / "CIFAR10Batch" / "batch.pt"),
        map_location: Any = "cpu",
    ) -> None:
        batch = torch.load(root, map_location=map_location)
        self.x, self.label = batch

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.label[index]

    def __len__(self) -> int:
        return len(self.label)


class CifarBatchDataset(AbstractDataset):
    def __init__(self, batch_size: int = 32, map_location: Any = "cpu") -> None:
        self._batch = _CifarBatch(map_location=map_location)
        self._dataloader = DataLoader(self._batch, batch_size=batch_size)

    @property
    def trainset(self) -> DataLoader:
        return self._dataloader

    @property
    def testset(self) -> DataLoader:
        return self._dataloader

    @property
    def batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return next(iter(self._dataloader))
