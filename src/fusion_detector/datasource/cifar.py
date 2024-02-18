import pathlib
import pickle
from typing import *

import torch
from PIL.Image import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset, Dataset
from torchvision import datasets, transforms

from .. import misc
from .abstract import AbstractDataSource

__all__ = [
    "CIFAR_MEAN",
    "CIFAR_STD",
    "denormalize_cifar",
    "CifarDataSource",
    "CifarBatchDataSource",
    "SegmentedAdversarialCifarDataset",
    "ApgdAdversarialCifarDataSource",
    "CWAdversarialCifarDataSource",
    "RFgsmAdversarialCifarDataSource",
    "AdversarialCifarDataSource",
]

SCRIPT_DIRECTORY = pathlib.Path(__file__).parent
INNER_PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
SOURCE_DIRECTORY = INNER_PROJECT_DIRECTORY.parent
PROJECT_DIRECTORY = SOURCE_DIRECTORY.parent
DATASET_DIRECTORY = PROJECT_DIRECTORY / "data" / "datasets"

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2471, 0.2435, 0.2616)


def denormalize_cifar(x: torch.Tensor) -> torch.Tensor:
    return misc.denormalize(x, CIFAR_MEAN, CIFAR_STD)


class CifarDataSource(AbstractDataSource):
    def __init__(
        self,
        batch_size: int,
        root: pathlib.Path = DATASET_DIRECTORY / "CIFAR10",
        num_workers: int = 0,
    ) -> None:
        raw_trainset = datasets.CIFAR10(
            str(root),
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
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
                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
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
        self._dataset = _CifarBatchDataset(
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                ]
            )
        )
        self._dataloader = DataLoader(self._dataset, batch_size)
        self._iter = iter(self._dataloader)

    @property
    def trainset(self) -> DataLoader:
        return self._dataloader

    @property
    def testset(self) -> DataLoader:
        return self._dataloader

    def batch(self) -> List[torch.Tensor]:
        return next(self._iter)


class SegmentedAdversarialCifarDataset(Dataset):
    DIRECTORY = PROJECT_DIRECTORY / "data" / "datasets" / "CIFAR10Adversarial"

    def __init__(
        self,
        attack: str,
        victim: str,
        train: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.transform = transform
        self.lst = misc.SegmentedSerializableList[Tuple[Image, int]](
            self.__class__.DIRECTORY / attack / victim, "train" if train else "test"
        )

    def __getitem__(self, index) -> Tuple[Image, int]:
        if self.transform is not None:
            x, label = self.lst[index]
            return self.transform(x), label
        return self.lst[index]

    def __len__(self) -> int:
        return len(self.lst)


class _SpecifiedAttackCifarDataset(ConcatDataset):
    VICTIMS = ["densenet121", "googlenet", "mobilenet_v2", "resnet50"]

    def __init__(
        self,
        attack: str,
        train: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__(
            (
                SegmentedAdversarialCifarDataset(attack, victim, train, transform)
                for victim in self.__class__.VICTIMS
            )
        )


class _CommonDataSourceTemplate(AbstractDataSource):
    def __init__(
        self,
        dataset_class: type,
        batch_size: int,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        num_workers: int = 0,
    ) -> None:
        raw_trainset = dataset_class(train=True, transform=transform)
        raw_testset = dataset_class(train=False, transform=transform)
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


class _ApgdAdversarialCifarDataset(_SpecifiedAttackCifarDataset):
    def __init__(
        self,
        train: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__("apgd", train, transform)


class ApgdAdversarialCifarDataSource(_CommonDataSourceTemplate):
    def __init__(
        self,
        batch_size: int,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            _ApgdAdversarialCifarDataset,
            batch_size,
            transform,
            num_workers,
        )


class _CWAdversarialCifarDataset(_SpecifiedAttackCifarDataset):
    def __init__(
        self,
        train: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__("cw", train, transform)


class CWAdversarialCifarDataSource(_CommonDataSourceTemplate):
    def __init__(
        self,
        batch_size: int,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            _CWAdversarialCifarDataset,
            batch_size,
            transform,
            num_workers,
        )


class _RFgsmAdversarialCifarDataset(_SpecifiedAttackCifarDataset):
    def __init__(
        self,
        train: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__("rfgsm", train, transform)


class RFgsmAdversarialCifarDataSource(_CommonDataSourceTemplate):
    def __init__(
        self,
        batch_size: int,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            _RFgsmAdversarialCifarDataset,
            batch_size,
            transform,
            num_workers,
        )


class _AdversarialCifarDataset(ConcatDataset):
    def __init__(
        self,
        train: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__(
            (
                _ApgdAdversarialCifarDataset(train, transform),
                _CWAdversarialCifarDataset(train, transform),
                _RFgsmAdversarialCifarDataset(train, transform),
            )
        )


class AdversarialCifarDataSource(_CommonDataSourceTemplate):
    def __init__(
        self,
        batch_size: int,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            _AdversarialCifarDataset,
            batch_size,
            transform,
            num_workers,
        )
