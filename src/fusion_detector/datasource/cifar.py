import pathlib
import pickle
from typing import *

from PIL.Image import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

from .abstract import AbstractDataSource

__all__ = [
    "CifarDataSource",
    "CifarBatchDataSource",
]

SCRIPT_DIRECTORY = pathlib.Path(__file__).parent
INNER_PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
SOURCE_DIRECTORY = INNER_PROJECT_DIRECTORY.parent
PROJECT_DIRECTORY = SOURCE_DIRECTORY.parent
DATASET_DIRECTORY = PROJECT_DIRECTORY / "data" / "datasets"


class CifarDataSource(AbstractDataSource):
    def __init__(
        self,
        batch_size: int,
        root: pathlib.Path = DATASET_DIRECTORY / "CIFAR10",
        num_workers: int = 0,
    ) -> None:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        raw_trainset = datasets.CIFAR10(
            str(root),
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            download=True,
        )
        raw_testset = datasets.CIFAR10(
            str(root),
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


class _CifarBatchDataset(Dataset):
    def __init__(
        self,
        root: pathlib.Path = DATASET_DIRECTORY / "CIFAR10Batch",
        transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        with open(root / "batch.pickle", "rb") as file:
            self.pairs: List[Tuple[Image, int]] = pickle.load(file)

    def __getitem__(self, index: int) -> Tuple[Image, int]:
        if self.transform is not None:
            x, y = self.pairs[index]
            x = self.transform(x)
            return x, y
        return self.pairs[index]

    def __len__(self) -> int:
        return len(self.pairs)


class CifarBatchDataSource(AbstractDataSource):
    def __init__(self, batch_size: int) -> None:
        self.batchsize = batch_size
        self._dataset = _CifarBatchDataset(transform=transforms.ToTensor())
        self._dataloader = DataLoader(self._dataset, batch_size)
        self._iter = iter(self._dataloader)

    @property
    def trainset(self) -> DataLoader:
        return self._dataloader

    @property
    def testset(self) -> DataLoader:
        return self._dataloader

    def batch(self) -> List[Tuple[Image, int]]:
        return next(self._iter)
