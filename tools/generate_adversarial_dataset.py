import math
import os
import pathlib
import sys
from typing import *

import matplotlib.pyplot as plt
import torch
from tools_common import PROJECT_DIRECTORY
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

project_path = str(PROJECT_DIRECTORY)
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import src.fusion_detector.thirdparty.pytorch_cifar10.module as M
from src.fusion_detector import dataset, perturbation

PRETRAINED_MODULE_CONSTRUCTORS = {
    "vgg19_bn": M.vgg19_bn,
    "resnet50": M.resnet50,
    "densenet121": M.densenet121,
    "mobilenet_v2": M.mobilenet_v2,
    "googlenet": M.googlenet,
    "inception_v3": M.inception_v3,
}

PERTURBATION_CONSTRUCTORS = {
    "FGSM": perturbation.fgsm,
    "PGD": perturbation.pgd,
}

CIFAR_ROOT = PROJECT_DIRECTORY / "data" / "datasets" / "CIFAR10"
CIFAR_ADVERSARIAL_ROOT = PROJECT_DIRECTORY / "data" / "datasets" / "CIFAR10Adversarial"
SEGMENT_SIZE = 10000
BATCH_SIZE = 25

CUDA_AVAILABLE = torch.cuda.is_available()
NUM_WORKERS = math.floor(os.cpu_count() / 2)


def use_cuda(x):
    if CUDA_AVAILABLE:
        return x.cuda()
    return x


def load_cifar() -> Tuple[DataLoader, DataLoader]:
    trainset = datasets.CIFAR10(
        CIFAR_ROOT,
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )
    trainloader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )
    testset = datasets.CIFAR10(
        CIFAR_ROOT,
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )
    testloader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )
    return trainloader, testloader


def save_segment(
    adversarial_list: List[torch.Tensor],
    label_list: List[torch.Tensor],
    path: pathlib.PurePath,
) -> None:
    adversarials = torch.cat(adversarial_list)
    labels = torch.cat(label_list)
    savepath = pathlib.Path(path) if not isinstance(path, pathlib.Path) else path
    save_directory = savepath.parent
    if not save_directory.exists():
        save_directory.mkdir(parents=True)
    torch.save([adversarials, labels], str(savepath))


def generate_dataset(victims: List[nn.Module], data: DataLoader, desc: str) -> None:
    dataset_size = data.batch_size * len(data)
    single_segment = dataset_size <= SEGMENT_SIZE
    for victim_name, victim in tqdm(
        victims.items(),
        desc=f"Generating {desc}",
        leave=False,
    ):
        for attack_name, constructor in tqdm(
            PERTURBATION_CONSTRUCTORS.items(),
            desc="Attacks",
            leave=False,
        ):
            save_directory = CIFAR_ADVERSARIAL_ROOT / attack_name / victim_name
            attack = constructor(victim)
            adversarial_list = []
            label_list = []
            count = 0
            segment = 0
            for x, label in tqdm(data, desc=desc, leave=False):
                x = use_cuda(x)
                label = use_cuda(label)
                adversarial = attack(x).clone().detach()
                adversarial_list.append(adversarial)
                label_list.append(label.clone().detach())
                count += len(label)
                if count == min(dataset_size, SEGMENT_SIZE):
                    if single_segment:
                        save_segment(
                            adversarial_list,
                            label_list,
                            save_directory / f"{desc}.pt",
                        )
                        return
                    save_segment(
                        adversarial_list,
                        label_list,
                        save_directory / f"{desc}_{segment}.pt",
                    )
                    segment += 1
                    count = 0
                    adversarial_list.clear()
                    label_list.clear()
            if len(adversarial_list) > 0:
                assert len(adversarial_list) == len(label_list)
                assert len(adversarial_list) < SEGMENT_SIZE
                save_segment(
                    adversarial_list,
                    label_list,
                    save_directory / f"{desc}_{segment}.pt",
                )


def main():
    assert SEGMENT_SIZE % BATCH_SIZE == 0
    pretrained_modules = {
        name: use_cuda(constructor(pretrained=True))
        for name, constructor in PRETRAINED_MODULE_CONSTRUCTORS.items()
    }
    trainloader, testloader = load_cifar()
    generate_dataset(pretrained_modules, testloader, "testset")
    generate_dataset(pretrained_modules, trainloader, "trainset")


# Guideline recommended Main Guard
if __name__ == "__main__":
    main()
