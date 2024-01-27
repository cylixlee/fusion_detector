import pathlib
from typing import *
from typing import Any

import lightning
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from fusion_detector import dataset
from fusion_detector.misc import AccuracyRecorder
from thirdparty.pytorch_cifar10 import module as M

SCRIPT_DIRECTORY = pathlib.PurePath(__file__).parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent

ALL_PRETRAINED_MODULES = {
    "vgg11_bn": M.vgg11_bn,
    "vgg13_bn": M.vgg13_bn,
    "vgg16_bn": M.vgg16_bn,
    "vgg19_bn": M.vgg19_bn,
    "resnet18": M.resnet18,
    "resnet34": M.resnet34,
    "resnet50": M.resnet50,
    "densenet121": M.densenet121,
    "densenet161": M.densenet161,
    "densenet169": M.densenet169,
    "mobilenet_v2": M.mobilenet_v2,
    "googlenet": M.googlenet,
    "inception_v3": M.inception_v3,
}


class BaselineWrapper(lightning.LightningModule):
    def __init__(self, underlying: nn.Module) -> None:
        super().__init__()
        self.underlying = underlying
        self.accuracy = AccuracyRecorder()

    def test_step(self, batch) -> STEP_OUTPUT:
        x, label = batch
        y = self.underlying(x)

        predict = torch.argmax(y, dim=1)
        correct = torch.eq(predict, label).sum().item()
        total = len(label)
        self.accuracy.push(correct, total)


def test_baseline(
    constructor: Callable[..., nn.Module],
    data: dataset.AbstractDataset,
) -> float:
    wrapper = BaselineWrapper(constructor(pretrained=True))
    trainer = lightning.Trainer(
        enable_checkpointing=False,
        logger=False,
    )
    trainer.test(wrapper, data.testset)
    accuracy = wrapper.accuracy.pop()
    print(accuracy)
    return accuracy


def main():
    cifar = dataset.CifarDataset(
        PROJECT_DIRECTORY / "data" / "datasets" / "CIFAR10",
        batch_size=16,
    )
    accuracies = {}
    for name, constructor in ALL_PRETRAINED_MODULES.items():
        accuracies[name] = test_baseline(constructor, cifar)
    for name, accuracy in accuracies.items():
        print(name, "accuracy", accuracy)


# Guideline recommended Main Guard
if __name__ == "__main__":
    main()
