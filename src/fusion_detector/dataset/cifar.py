import pathlib
from typing import *

from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

from .abstract import AbstractDataset

__all__ = [
    "CifarDataset",
]

SCRIPT_DIRECTORY = pathlib.PurePath(__file__).parent
INNER_PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
SOURCE_DIRECTORY = INNER_PROJECT_DIRECTORY.parent
PROJECT_DIRECTORY = SOURCE_DIRECTORY.parent
DATASET_DIRECTORY = PROJECT_DIRECTORY / "data" / "datasets"


class CifarDataset(AbstractDataset):
    def __init__(
        self,
        batch_size: int,
        root: str = str(DATASET_DIRECTORY / "CIFAR10"),
        num_workers: int = 0,
    ) -> None:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        raw_trainset = datasets.CIFAR10(
            root,
            train=True,
            transform=transforms.ToTensor(),
            download=True,
        )
        raw_testset = datasets.CIFAR10(
            root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            download=True,
        )
        self._trainloader = DataLoader(
            raw_trainset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        self._testloader = DataLoader(
            raw_testset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    @property
    def trainset(self) -> DataLoader:
        return self._trainloader

    @property
    def testset(self) -> DataLoader:
        return self._testloader
