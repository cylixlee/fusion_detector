import pathlib
from typing import *
from typing import Any

import lightning
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import optim

from fusion_detector.classifier import LinearAdversarialClassifier
from fusion_detector.extractor import MobileResExtractor, ResNetKind

SCRIPT_PATH = pathlib.Path(__file__).parent
PROJECT_PATH = SCRIPT_PATH.parent
LOGGER_PATH = PROJECT_PATH / "log"


class MobileResLinearAdversarialDetector(lightning.LightningModule):
    def __init__(self, learning_rate: float = 0.01) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.extractor = MobileResExtractor(ResNetKind.RESNET_50)
        self.classifier = LinearAdversarialClassifier(2048 + 1280)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.SGD(self.classifier.parameters(), self.learning_rate)

    def training_step(self, batch: Any) -> STEP_OUTPUT:
        x, label = batch
        # Feature extraction
        x = self.extractor(x)
        # Binary classification (adversarial, or non-adversarial)
        y = self.classifier(x).view(-1)
        # Return loss to lightning framework.
        loss = F.binary_cross_entropy_with_logits(y, label)
        self.log("train loss", loss.item(), on_step=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any) -> STEP_OUTPUT:
        x, label = batch
        # Feature extraction
        x = self.extractor(x)
        # Binary classification (adversarial, or non-adversarial)
        y = self.classifier(x).view(-1)
        predict = torch.floor(y + 0.5)
        # Collect statistics
        correct = torch.eq(predict, label).sum().item()
        total = len(label)
        self.log(
            "test accuracy", correct / total, prog_bar=True, on_epoch=True, on_step=True
        )


BATCH_SIZE = 32
NUM_WORKERS = 2


def main():
    savepath = PROJECT_PATH / "data" / "modules" / "CIFAR10Adversarial"
    if not savepath.exists():
        savepath.mkdir(parents=True)

    detector = MobileResLinearAdversarialDetector.load_from_checkpoint(
        savepath / "MobileResLinear.ckpt"
    )

    trainer = lightning.Trainer(
        # max_epochs=100,
        logger=TensorBoardLogger(LOGGER_PATH),
        enable_checkpointing=False,
    )

    # test_data = dataset.CifarHybridDataSource(
    #     train=False, transform=transforms.ToTensor()
    # )

    # train_data = dataset.CifarHybridDataSource(
    #     train=True, transform=transforms.ToTensor()
    # )

    # trainloader = DataLoader(
    #     train_data,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     persistent_workers=NUM_WORKERS > 0,
    # )

    # testloader = DataLoader(
    #     test_data,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     persistent_workers=NUM_WORKERS > 0,
    # )

    # trainer.fit(detector, trainloader)
    # trainer.save_checkpoint(savepath / "MobileResLinear.ckpt")
    # trainer.test(detector, testloader)


# Guideline recommended Main Guard
if __name__ == "__main__":
    main()
