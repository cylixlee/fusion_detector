import pathlib
from typing import *
from typing import Any

import lightning
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, optim

from constants import PARTICIPATED_MODULES
from fusion_detector.dataset import CifarDataset

SCRIPT_PATH = pathlib.PurePath(__file__).parent  # The 'src' directory
PROJECT_PATH = SCRIPT_PATH.parent  # The outer `fusion_detector` directory
DATA_PATH = PROJECT_PATH / "data"

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.1


class BaselineWrapper(lightning.LightningModule):
    def __init__(self, underlying: nn.Module):
        super().__init__()
        self.underlying = underlying
        self.criteria = F.cross_entropy
        self.correct = 0
        self.total = 0

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.SGD(self.parameters(), lr=LEARNING_RATE)

    def training_step(self, batch: Any) -> STEP_OUTPUT:
        x, label = batch
        y = self.underlying(x)
        loss = self.criteria(y, label)
        return loss

    def test_step(self, batch: Any) -> STEP_OUTPUT:
        x, label = batch
        y = self.underlying(x)
        loss = self.criteria(y, label)

        predict = torch.argmax(y, dim=1)
        self.correct += torch.eq(predict, label).sum().item()
        self.total += len(label)
        return loss


def main():
    # Initialize dataset.
    cifar = CifarDataset(DATA_PATH / "datasets" / "CIFAR10", batch_size=BATCH_SIZE)

    # Initialize modules.
    module_names = list(PARTICIPATED_MODULES.keys())
    modules = [BaselineWrapper(module) for module in PARTICIPATED_MODULES.values()]

    # Train, save and test modules.
    for index, module in enumerate(modules):
        trainer = lightning.Trainer(
            enable_checkpointing=False,
            max_epochs=EPOCHS,
            logger=False,
        )
        trainer.fit(module, cifar.trainset)
        trainer.save_checkpoint(
            DATA_PATH / "modules" / "CIFAR10" / f"{module_names[index]}.ckpt"
        )
        trainer.test(module, cifar.testset)
        print(
            f"Evaluated module {module_names[index]} "
            f"with {module.correct / module.total} accuracy."
        )


# Guideline recommended Main Guard.
if __name__ == "__main__":
    main()
