import pathlib
from typing import *
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, optim

from fusion_detector import datasource
from fusion_detector.classifier import LinearAdversarialClassifier
from fusion_detector.extractor import (
    AbstractFeatureExtractor,
    MobileResConvExtractor,
    MobileResLinearExtractor,
    ResNetKind,
)

SCRIPT_PATH = pathlib.Path(__file__).parent
PROJECT_PATH = SCRIPT_PATH.parent
# LOGGER_PATH = PROJECT_PATH / "log"
LOGGER_PATH = "/root/tf-logs"


class FusionDetectorTemplate(L.LightningModule):
    def __init__(
        self,
        extractor: AbstractFeatureExtractor,
        classifier: nn.Module,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.classifier = classifier
        self.learning_rate = learning_rate

    def _trainable_parameters(self) -> Iterable[nn.Parameter]:
        extractor_parameters = self.extractor.trainable_parameters()
        if extractor_parameters is not None:
            return (*self.classifier.parameters(), *extractor_parameters)
        return self.classifier.parameters()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.SGD(self._trainable_parameters(), self.learning_rate)

    def training_step(self, batch: Any) -> STEP_OUTPUT:
        x, label = batch
        # Feature extraction
        x = self.extractor(x)
        # Binary classification (adversarial, or non-adversarial)
        y = self.classifier(x).view(-1)
        y = F.sigmoid(y)
        predict = torch.floor(y + 0.5)
        # Collect statistics
        correct = torch.eq(predict, label).sum().item()
        total = len(label)
        # Return loss to lightning framework.
        loss = F.binary_cross_entropy_with_logits(y, label)
        self.log("loss", loss.item(), prog_bar=True, on_epoch=True, on_step=True)
        self.log("acc", correct / total, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch: Any) -> STEP_OUTPUT:
        x, label = batch
        # Feature extraction
        x = self.extractor(x)
        # Binary classification (adversarial, or non-adversarial)
        y = self.classifier(x).view(-1)
        y = F.sigmoid(y)
        predict = torch.floor(y + 0.5)
        # Collect statistics
        correct = torch.eq(predict, label).sum().item()
        total = len(label)
        self.log("acc", correct / total, prog_bar=True, on_epoch=True, on_step=True)


class MobileResLinearDetector(FusionDetectorTemplate):
    def __init__(
        self,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__(
            MobileResLinearExtractor(ResNetKind.RESNET_50),
            LinearAdversarialClassifier(2048 + 1280),
            learning_rate,
        )


class MobileResConvDetector(FusionDetectorTemplate):
    def __init__(
        self,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__(
            MobileResConvExtractor(ResNetKind.RESNET_50_CONV),
            LinearAdversarialClassifier(8192),
            learning_rate,
        )


BATCH_SIZE = 256
NUM_WORKERS = 12
LEARNING_RATE = 0.01


def main():
    savepath = PROJECT_PATH / "data" / "modules" / "CIFAR10Adversarial"
    if not savepath.exists():
        savepath.mkdir(parents=True)

    detector = MobileResConvDetector(LEARNING_RATE)
    # detector = MobileResConvDetector(LEARNING_RATE)

    # detector = MobileResConvDetector.load_from_checkpoint(
    #     savepath / "MobileResConv.ckpt"
    # )

    # detector = MobileResLinearDetector.load_from_checkpoint(
    #     savepath / "MobileResLinear.ckpt"
    # )

    trainer = L.Trainer(
        max_epochs=100, logger=TensorBoardLogger(LOGGER_PATH), default_root_dir=savepath
    )

    # trainer = L.Trainer(logger=False, enable_checkpointing=False)

    # source = datasource.HybridCifarDataSource(BATCH_SIZE, num_workers=NUM_WORKERS)
    source = datasource.HybridStrongCifarDataSource(BATCH_SIZE, num_workers=NUM_WORKERS)

    trainer.fit(detector, source.trainset)
    # trainer.save_checkpoint(savepath / "MobileResLinear.ckpt")
    trainer.test(detector, source.testset, ckpt_path="best")


# Guideline recommended Main Guard
if __name__ == "__main__":
    main()
