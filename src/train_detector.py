import pathlib
from typing import *

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, optim

from fusion_detector import datasource
from fusion_detector.classifier import (
    ConvAdversarialClassifier,
    LinearAdversarialClassifier,
)
from fusion_detector.extractor import (
    AbstractFeatureExtractor,
    MobileResConcatExtractor,
    MobileResSeparateExtractor,
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

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.RMSprop(self.classifier.parameters(), self.learning_rate)

    def training_step(self, batch: Any) -> STEP_OUTPUT:
        x, label = batch
        # Feature extraction
        x = self.extractor(x)
        # Binary classification (adversarial, or non-adversarial)
        y = self.classifier(x).view(-1)
        # Collect statistics
        predict = torch.floor(F.sigmoid(y) + 0.5)
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
        # Collect statistics
        predict = torch.floor(F.sigmoid(y) + 0.5)
        correct = torch.eq(predict, label).sum().item()
        total = len(label)
        self.log("acc", correct / total, prog_bar=True, on_epoch=True, on_step=True)


class MobileResLinearDetector(FusionDetectorTemplate):
    def __init__(
        self,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__(
            MobileResConcatExtractor(ResNetKind.RESNET_50),
            LinearAdversarialClassifier(512 + 192),
            learning_rate,
        )


class MobileResConvDetector(FusionDetectorTemplate):
    def __init__(
        self,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__(
            MobileResSeparateExtractor(ResNetKind.RESNET_50_CONV),
            ConvAdversarialClassifier((512, 8), (192, 16), out_channels=1024),
            learning_rate,
        )


BATCH_SIZE = 512
NUM_WORKERS = 12
LEARNING_RATE = 0.01


def main():
    savepath = PROJECT_PATH / "data" / "modules" / "CIFAR10Adversarial"
    if not savepath.exists():
        savepath.mkdir(parents=True)

    # detector = MobileResLinearDetector(LEARNING_RATE)
    detector = MobileResConvDetector(LEARNING_RATE)

    # detector = MobileResLinearDetector.load_from_checkpoint(
    #     savepath / "MobileResLinear.ckpt"
    # )

    # detector = MobileResConvDetector.load_from_checkpoint(
    #     savepath / "MobileResConv.ckpt"
    # )
    # detector.train(False)
    # detector.eval()

    trainer = L.Trainer(max_epochs=100, logger=TensorBoardLogger(LOGGER_PATH))

    # trainer = L.Trainer(logger=False, enable_checkpointing=False)

    source = datasource.HybridStrongCifarDataSource(BATCH_SIZE, num_workers=NUM_WORKERS)

    trainer.fit(detector, source.trainset)
    trainer.test(detector, source.testset, ckpt_path="best")

    # trainer.test(detector, source.testset)

    # trainer.save_checkpoint(savepath / "MobileResLinear.ckpt")
    trainer.save_checkpoint(savepath / "MobileResConv.ckpt")


# Guideline recommended Main Guard
if __name__ == "__main__":
    main()
    # detector = MobileResConvDetector(LEARNING_RATE)
    # print(detector)
