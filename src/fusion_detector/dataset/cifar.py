import math
import os

from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

from .abstract import AbstractDataset

__all__ = ["CifarDataset"]


class CifarDataset(AbstractDataset):
    """ImageNet-compatible CIFAR-10 Dataset.

    This dataset is applied with transforms to match ImageNet's image shape.
    """

    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int = math.floor(os.cpu_count() / 2),
    ) -> None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(0, 1),
                transforms.Resize((224, 224), antialias=True),
            ]
        )
        raw_trainset = datasets.CIFAR10(
            root,
            train=True,
            transform=transform,
            download=True,
        )
        raw_testset = datasets.CIFAR10(
            root,
            train=False,
            transform=transform,
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
