import pathlib
from typing import *
from typing import Any

import lightning
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

from fusion_detector import dataset
from fusion_detector.classifier import LinearAdversarialClassifier
from fusion_detector.extractor import MobileResExtractor, ResNetKind
from fusion_detector.misc import AccuracyRecorder

SCRIPT_PATH = pathlib.Path(__file__).parent
PROJECT_PATH = SCRIPT_PATH.parent
LOGGER_PATH = PROJECT_PATH / "log"


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
        self.accuracy.push(correct, total)


class CompoundCifarDataSource(Dataset):
    def __init__(self, train: bool = True) -> None:
        super().__init__()
        self.fgsm = dataset.CifarFgsmAdversarialDataSource(train=train)
        self.pgd = dataset.CifarPgdAdversarialDataSource(train=train)
        self.normal = datasets.CIFAR10(
            str(PROJECT_PATH / "data" / "datasets" / "CIFAR10"),
            train=train,
            transform=transforms.ToTensor(),
            target_transform=lambda _: torch.scalar_tensor(0.0),
            download=True,
        )
        self.fgsmlen = len(self.fgsm)
        self.pgdlen = len(self.pgd)
        self.normallen = len(self.normal)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if index < self.fgsmlen:
            return self.fgsm.__getitem__(index)
        elif index < self.fgsmlen + self.pgdlen:
            return self.pgd.__getitem__(index - self.fgsmlen)
        return self.normal.__getitem__(index - self.fgsmlen - self.pgdlen)

    def __len__(self) -> int:
        return self.fgsmlen + self.pgdlen + self.normallen


BATCH_SIZE = 32
NUM_WORKERS = 0


def main():
    detector = MobileResLinearAdversarialDetector(0.01)
    trainer = lightning.Trainer(
        max_epochs=100,
        logger=TensorBoardLogger(LOGGER_PATH),
        enable_checkpointing=False,
    )

    train_datasource = CompoundCifarDataSource()
    test_datasource = CompoundCifarDataSource(train=False)
    trainloader = DataLoader(
        train_datasource,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
    )
    testloader = DataLoader(
        test_datasource,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
    )

    # trainer.fit(detector, trainloader)
    trainer.test(detector, testloader)
    print(f"Tested with {detector.accuracy.push()} accuracy.")

    savepath = PROJECT_PATH / "data" / "modules" / "CIFAR10Adversarial"
    if not savepath.exists():
        savepath.mkdir(parents=True)
    trainer.save_checkpoint(savepath / "MobileResLinear.ckpt")


# Guideline recommended Main Guard
if __name__ == "__main__":
    main()
