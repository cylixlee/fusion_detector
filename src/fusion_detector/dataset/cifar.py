import math
import os
import pathlib
from math import floor
from os import PathLike, cpu_count
from typing import *
from typing import Any

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

from .abstract import AbstractDataset

__all__ = [
    "CifarDataset",
    "NormalizedCifarDataset",
    "NormalizedCifarBatchDataset",
    "CifarFgsmAdversarialDataSource",
    "CifarPgdAdversarialDataSource",
    "CifarFgsmAdversarialDataset",
    "CifarPgdAdversarialDataset",
    "CifarAdversarialBatchDataset",
]

SCRIPT_DIRECTORY = pathlib.PurePath(__file__).parent
INNER_PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
SOURCE_DIRECTORY = INNER_PROJECT_DIRECTORY.parent
PROJECT_DIRECTORY = SOURCE_DIRECTORY.parent
DATASET_DIRECTORY = PROJECT_DIRECTORY / "data" / "datasets"

DEFAULT_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"


class CifarDataset(AbstractDataset):
    def __init__(
        self,
        batch_size: int,
        root: str = str(DATASET_DIRECTORY / "CIFAR10"),
        num_workers: int = 0,
    ) -> None:
        raw_trainset = datasets.CIFAR10(
            root,
            train=True,
            transform=transforms.ToTensor(),
            target_transform=lambda label: torch.zeros_like(label),
            download=True,
        )
        raw_testset = datasets.CIFAR10(
            root,
            train=False,
            transform=transforms.ToTensor(),
            target_transform=lambda label: torch.zeros_like(label),
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


class NormalizedCifarDataset(AbstractDataset):
    """Thirdparty project `pytorch_cifar10`-compatible CIFAR10 dataset."""

    def __init__(
        self,
        batch_size: int,
        root: str = str(DATASET_DIRECTORY / "CIFAR10"),
        num_workers: int = 0,
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


class _NormalizedCifarBatch(Dataset):
    def __init__(
        self,
        root: str = str(DATASET_DIRECTORY / "CIFAR10Batch" / "batch.pt"),
        map_location: Any = DEFAULT_DEVICE,
    ) -> None:
        batch = torch.load(root, map_location=map_location)
        self.x, self.label = batch

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.label[index]

    def __len__(self) -> int:
        return len(self.label)


class NormalizedCifarBatchDataset(AbstractDataset):
    def __init__(
        self,
        batch_size: int = 32,
        map_location: Any = DEFAULT_DEVICE,
    ) -> None:
        self._batch = _NormalizedCifarBatch(map_location=map_location)
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


class _CifarSingleAdversarialAttackDataSource(Dataset):
    IMAGES_PER_SEGMENT = 10000
    TRAIN_SEGMENTS = 5

    def __init__(
        self,
        root: os.PathLike,
        train: bool = True,
        transform: Optional[Callable] = None,
        map_location: Any = DEFAULT_DEVICE,
    ) -> None:
        if not isinstance(root, pathlib.Path):
            root = pathlib.Path(root)
        assert root.exists()
        self.train = train
        self.transform = transform
        self.map_location = map_location
        self.victims: List[pathlib.Path] = []
        for entry in root.iterdir():
            assert entry.is_dir()
            self.victims.append(entry)
        self.root = root
        self.current_victim: Optional[str] = None
        self.segment: Optional[List[torch.Tensor]] = None

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        segment_index = index // self.__class__.IMAGES_PER_SEGMENT
        example_index = index % self.__class__.IMAGES_PER_SEGMENT
        if not self.train:
            if self.current_victim != self.victims[segment_index]:
                self.current_victim = self.victims[segment_index]
                self.segment = torch.load(
                    str(self.root / self.current_victim / "testset.pt"),
                    map_location=self.map_location,
                )
            x, label = self.segment[0][example_index], self.segment[1][example_index]
            if self.transform is not None:
                # return self.transform(x), label
                return self.transform(x), torch.scalar_tensor(1.0)
            # return x, label
            return x, torch.scalar_tensor(1.0)
        victim_index = segment_index // self.__class__.TRAIN_SEGMENTS
        segment_index = segment_index % self.__class__.TRAIN_SEGMENTS
        if self.current_victim != self.victims[victim_index]:
            self.current_victim = self.victims[victim_index]
            self.segment = torch.load(
                str(self.root / self.current_victim / f"trainset_{segment_index}.pt"),
                map_location=self.map_location,
            )
        x, label = self.segment[0][example_index], self.segment[1][example_index]
        if self.transform is not None:
            return self.transform(x), torch.scalar_tensor(1.0)
        return x, torch.scalar_tensor(1.0)

    def __len__(self) -> int:
        if self.train:
            return (
                self.__class__.TRAIN_SEGMENTS
                * self.__class__.IMAGES_PER_SEGMENT
                * len(self.victims)
            )
        return self.__class__.IMAGES_PER_SEGMENT * len(self.victims)


class CifarFgsmAdversarialDataSource(_CifarSingleAdversarialAttackDataSource):
    def __init__(
        self,
        root: os.PathLike = DATASET_DIRECTORY / "CIFAR10Adversarial",
        train: bool = True,
        transform: Optional[Callable] = None,
        map_location: Any = DEFAULT_DEVICE,
    ) -> None:
        if not isinstance(root, pathlib.Path):
            root = pathlib.Path(root)
        super().__init__(root / "FGSM", train, transform, map_location)


class CifarPgdAdversarialDataSource(_CifarSingleAdversarialAttackDataSource):
    def __init__(
        self,
        root: os.PathLike = DATASET_DIRECTORY / "CIFAR10Adversarial",
        train: bool = True,
        transform: Optional[Callable] = None,
        map_location: Any = DEFAULT_DEVICE,
    ) -> None:
        if not isinstance(root, pathlib.Path):
            root = pathlib.Path(root)
        super().__init__(root / "PGD", train, transform, map_location)


class _CifarTemplateAdversarialDataset(AbstractDataset):
    def __init__(
        self,
        underlying: Any,
        batch_size: int,
        root: os.PathLike = DATASET_DIRECTORY / "CIFAR10Adversarial",
        map_location: Any = DEFAULT_DEVICE,
        num_workers: int = 0,
    ) -> None:
        raw_trainset = underlying(
            root,
            train=True,
            map_location=map_location,
        )
        raw_testset = underlying(
            root,
            train=False,
            map_location=map_location,
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


class CifarFgsmAdversarialDataset(_CifarTemplateAdversarialDataset):
    def __init__(
        self,
        batch_size: int,
        root: os.PathLike = DATASET_DIRECTORY / "CIFAR10Adversarial",
        map_location: Any = DEFAULT_DEVICE,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            CifarFgsmAdversarialDataSource,
            batch_size,
            root,
            map_location,
            num_workers,
        )


class CifarPgdAdversarialDataset(_CifarTemplateAdversarialDataset):
    def __init__(
        self,
        batch_size: int,
        root: PathLike = DATASET_DIRECTORY / "CIFAR10Adversarial",
        map_location: Any = DEFAULT_DEVICE,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            CifarPgdAdversarialDataSource,
            batch_size,
            root,
            map_location,
            num_workers,
        )


class _CifarAdversarialBatch(Dataset):
    def __init__(
        self,
        root: str = str(DATASET_DIRECTORY / "CIFAR10AdversarialBatch" / "batch.pt"),
        map_location: Any = DEFAULT_DEVICE,
    ) -> None:
        batch = torch.load(root, map_location=map_location)
        self.x, self.label = batch

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.label[index]

    def __len__(self) -> int:
        return len(self.label)


class CifarAdversarialBatchDataset(AbstractDataset):
    def __init__(
        self,
        batch_size: int = 32,
        map_location: Any = DEFAULT_DEVICE,
    ) -> None:
        self._batch = _CifarAdversarialBatch(map_location=map_location)
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
