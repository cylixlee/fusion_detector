import pathlib
import pickle
from typing import *

import torch
from PIL.Image import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset, Dataset
from torchvision import datasets, transforms

from .. import misc
from .abstract import (
    AbstractDataSource,
    CommonDataSourceTemplate,
    JaggedDataset,
    RepeatDataset,
)

__all__ = [
    "CIFAR_MEAN",
    "CIFAR_STD",
    "CIFAR_TRANSFORM",
    "NORMAL_TARGET_TRANSFORM",
    "ADVERSARIAL_TARGET_TRANSFORM",
    "denormalize_cifar",
    "CifarDataSource",
    "CifarBatchDataSource",
    "SegmentedAdversarialCifarDataset",
    "AdversarialCifarDataSource",
    "HybridCifarDataset",
    "HybridCifarDataSource",
    "StrongAdversarialCifarDataset",
    "StrongAdversarialCifarDataSource",
    "HybridStrongCifarDataset",
    "HybridStrongCifarDataSource",
]

SCRIPT_DIRECTORY = pathlib.Path(__file__).parent
INNER_PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
SOURCE_DIRECTORY = INNER_PROJECT_DIRECTORY.parent
PROJECT_DIRECTORY = SOURCE_DIRECTORY.parent
DATASET_DIRECTORY = PROJECT_DIRECTORY / "data" / "datasets"

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2471, 0.2435, 0.2616)
CIFAR_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
)
NORMAL_TARGET_TRANSFORM = lambda _: 0.0  # Labels of adversarial examples are zeroes.
ADVERSARIAL_TARGET_TRANSFORM = lambda _: 1.0  # Labels of normal examples are ones.


def denormalize_cifar(x: torch.Tensor) -> torch.Tensor:
    return misc.denormalize(x, CIFAR_MEAN, CIFAR_STD)


def _cifar_dataset_wrapper(
    train: bool = True,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Callable[..., Dataset]:
    return datasets.CIFAR10(
        str(DATASET_DIRECTORY / "CIFAR10"),
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=True,
    )


class CifarDataSource(CommonDataSourceTemplate):
    def __init__(
        self,
        batch_size: int,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = CIFAR_TRANSFORM,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            _cifar_dataset_wrapper,
            batch_size,
            transform,
            target_transform,
            num_workers,
        )


class _CifarBatchDataset(Dataset):
    def __init__(
        self,
        root: pathlib.Path = DATASET_DIRECTORY / "CIFAR10Batch",
        transform: Optional[Callable] = CIFAR_TRANSFORM,
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
        self._dataset = _CifarBatchDataset(transform=CIFAR_TRANSFORM)
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
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = CIFAR_TRANSFORM,
        target_transform: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = ADVERSARIAL_TARGET_TRANSFORM,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        self.lst = misc.SegmentedSerializableList[Tuple[Image, int]](
            self.__class__.DIRECTORY / attack / victim, "train" if train else "test"
        )

    def __getitem__(self, index) -> Tuple[Image, int]:
        x, label = self.lst[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return x, label

    def __len__(self) -> int:
        return len(self.lst)


class _AdversarialCifarDataset(ConcatDataset):
    ATTACKS = ["apgd", "cw", "rfgsm"]
    VICTIMS = ["densenet121", "googlenet", "mobilenet_v2", "resnet50"]

    def __init__(
        self,
        train: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = CIFAR_TRANSFORM,
        target_transform: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = ADVERSARIAL_TARGET_TRANSFORM,
    ) -> None:
        super().__init__(
            (
                SegmentedAdversarialCifarDataset(
                    attack,
                    victim,
                    train,
                    transform,
                    target_transform,
                )
                for attack in self.__class__.ATTACKS
                for victim in self.__class__.VICTIMS
            )
        )


class AdversarialCifarDataSource(CommonDataSourceTemplate):
    def __init__(
        self,
        batch_size: int,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = CIFAR_TRANSFORM,
        target_transform: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = ADVERSARIAL_TARGET_TRANSFORM,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            _AdversarialCifarDataset,
            batch_size,
            transform,
            target_transform,
            num_workers,
        )


class HybridCifarDataset(JaggedDataset):
    def __init__(
        self,
        train: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = CIFAR_TRANSFORM,
        # Target transform cannot be specified in hybrid dataset.
        # This argument is left unused, in order to make compiler happy :)
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__(
            (
                _AdversarialCifarDataset(
                    train, transform, ADVERSARIAL_TARGET_TRANSFORM
                ),
                RepeatDataset(
                    _cifar_dataset_wrapper(train, transform, NORMAL_TARGET_TRANSFORM),
                    (
                        len(_AdversarialCifarDataset.ATTACKS)
                        * len(_AdversarialCifarDataset.VICTIMS)
                    ),
                ),
            )
        )


class HybridCifarDataSource(CommonDataSourceTemplate):
    def __init__(
        self,
        batch_size: int,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = CIFAR_TRANSFORM,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            HybridCifarDataset,
            batch_size,
            transform,
            None,
            num_workers,
        )


class StrongAdversarialCifarDataset(Dataset):
    def __init__(
        self,
        train: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = CIFAR_TRANSFORM,
        target_transform: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = ADVERSARIAL_TARGET_TRANSFORM,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        self.lst = misc.SegmentedSerializableList[Tuple[Image, int]](
            DATASET_DIRECTORY / "CIFAR10StrongAdversarial", "train" if train else "test"
        )

    def __getitem__(self, index) -> Tuple[Image, int]:
        x, label = self.lst[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return x, label

    def __len__(self) -> int:
        return len(self.lst)


class StrongAdversarialCifarDataSource(CommonDataSourceTemplate):
    def __init__(
        self,
        batch_size: int,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = CIFAR_TRANSFORM,
        target_transform: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = ADVERSARIAL_TARGET_TRANSFORM,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            StrongAdversarialCifarDataset,
            batch_size,
            transform,
            target_transform,
            num_workers,
        )


class HybridStrongCifarDataset(JaggedDataset):
    def __init__(
        self,
        train: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = CIFAR_TRANSFORM,
        # Target transform cannot be specified in hybrid dataset.
        # This argument is left unused, in order to make compiler happy :)
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__(
            (
                StrongAdversarialCifarDataset(
                    train, transform, ADVERSARIAL_TARGET_TRANSFORM
                ),
                RepeatDataset(
                    _cifar_dataset_wrapper(train, transform, NORMAL_TARGET_TRANSFORM),
                    2,
                ),
            )
        )


class HybridStrongCifarDataSource(CommonDataSourceTemplate):
    def __init__(
        self,
        batch_size: int,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = CIFAR_TRANSFORM,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            HybridStrongCifarDataset,
            batch_size,
            transform,
            None,
            num_workers,
        )
