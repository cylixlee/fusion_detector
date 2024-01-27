from typing import *
from typing import Any

import lightning
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, optim

from fusion_detector.classifier import LinearAdversarialClassifier
from fusion_detector.extractor import MobileResExtractor, ResNetKind
from fusion_detector.misc import AccuracyRecorder


class MobileResLinearAdversarialDetector(lightning.LightningModule):
    def __init__(self, learning_rate: float) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.extractor = MobileResExtractor(ResNetKind.RESNET_50)
        self.classifier = LinearAdversarialClassifier(2048 + 1280)
        self.accuracy = AccuracyRecorder()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.SGD(self.classifier.parameters(), self.learning_rate)

    def training_step(self, batch: Any) -> STEP_OUTPUT:
        x, label = batch
        # Feature extraction
        x = self.extractor(x)
        # Binary classification (adversarial, or non-adversarial)
        y = self.classifier(x)
        # Return loss to lightning framework.
        loss = F.cross_entropy(y, label)
        return loss

    def test_step(self, batch: Any) -> STEP_OUTPUT:
        x, label = batch
        # Feature extraction
        x = self.extractor(x)
        # Binary classification (adversarial, or non-adversarial)
        y = self.classifier(x)
        predict = torch.argmax(y, dim=1)
        # Collect statistics
        correct = torch.eq(predict, label).sum().item()
        total = len(label)
        self.accuracy.push(correct, total)


def main():
    ...


# Guideline recommended Main Guard
if __name__ == "__main__":
    main()
