import math
import os

from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

from .abstract import AbstractDataset

__all__ = ["CifarDataset"]


class CifarDataset(AbstractDataset):
    """Thirdparty project `pytorch_cifar10`-compatible CIFAR10 dataset."""

    def __init__(
        self,
        root: str,
        batch_size: int,
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
