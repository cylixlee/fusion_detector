import pathlib
import pickle
import random
import sys
from typing import *

import torch
from PIL.Image import Image
from tools_common import PROJECT_DIRECTORY
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from tqdm import tqdm, trange

project_path = str(PROJECT_DIRECTORY)
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from src.fusion_detector.dataset import (
    CifarFgsmAdversarialDataSource,
    CifarPgdAdversarialDataSource,
)

ATTACKS = ["FGSM", "PGD"]
VICTIMS = [
    "densenet121",
    "googlenet",
    "inception_v3",
    "mobilenet_v2",
    "resnet50",
    "vgg19_bn",
]


class NonAdversarialCifarDataset(Dataset):
    def __init__(self, train: bool = True) -> None:
        self.underlying = datasets.CIFAR10(
            str(PROJECT_DIRECTORY / "data" / "datasets" / "CIFAR10"),
            download=True,
            train=train,
        )

    def __getitem__(self, index):
        x, _ = self.underlying[index % len(self.underlying)]
        return x, 0.0

    def __len__(self) -> int:
        return 6 * len(self.underlying)


SEGMENT_SIZE = 60000


def save_segment(
    datasources: List[Dataset],
    dir: pathlib.Path,
    name: str,
):
    assert SEGMENT_SIZE % len(datasources) == 0
    length = len(datasources[0])
    for datasource in datasources[1:]:
        assert len(datasource) == length

    dumped_index = 0
    segment = []
    for index in trange(length, desc=name):
        datapieces = [datasource[index] for datasource in datasources]
        segment.extend(datapieces)
        if len(segment) == SEGMENT_SIZE:
            random.shuffle(segment)
            with open(dir / f"{name}_{dumped_index}.segment", "wb") as file:
                pickle.dump(segment, file)
            dumped_index += 1
            segment = []
    if len(segment) > 0:
        with open(dir / f"{name}_{dumped_index}.segment", "wb") as file:
            pickle.dump(segment, file)


def main():
    savepath = PROJECT_DIRECTORY / "data" / "datasets" / "CIFAR10Hybrid"
    if not savepath.exists():
        savepath.mkdir(parents=True)

    save_segment(
        [
            CifarFgsmAdversarialDataSource(train=True),
            CifarPgdAdversarialDataSource(train=True),
            NonAdversarialCifarDataset(train=True),
        ],
        savepath,
        "trainset",
    )
    save_segment(
        [
            CifarFgsmAdversarialDataSource(train=False),
            CifarPgdAdversarialDataSource(train=False),
            NonAdversarialCifarDataset(train=False),
        ],
        savepath,
        "testset",
    )


# Guideline recommended Main Guard
if __name__ == "__main__":
    main()
