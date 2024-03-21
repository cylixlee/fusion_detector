from typing import *

import inclconf
import torch
from torch import nn
from tqdm import tqdm

inclconf.configure_includepath()

from src.fusion_detector import datasource
from src.fusion_detector.thirdparty.pytorch_cifar10 import module as M

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS: Dict[str, nn.Module] = {
    "resnet50": M.resnet50(pretrained=True, device=DEFAULT_DEVICE),
    "densenet121": M.densenet121(pretrained=True, device=DEFAULT_DEVICE),
    "mobilenet_v2": M.mobilenet_v2(pretrained=True, device=DEFAULT_DEVICE),
    "googlenet": M.googlenet(pretrained=True, device=DEFAULT_DEVICE),
}
BATCH_SIZE = 256


def main():
    data = datasource.StrongAdversarialCifarDataSource(
        BATCH_SIZE, target_transform=None
    )

    for name, model in tqdm(MODELS.items(), desc="Model", leave=False):
        model = model.to(DEFAULT_DEVICE)
        correct = 0
        total = 0
        with tqdm(total=len(data.testset), desc="Dataset", leave=False) as progress:
            for x, labels in data.testset:
                x, labels = x.to(DEFAULT_DEVICE), labels.to(DEFAULT_DEVICE)
                y = model(x)
                predict = torch.argmax(y, dim=1)
                correct += torch.eq(predict, labels).sum().item()
                total += len(labels)
                progress.set_postfix({"acc": correct / total})
                progress.update()
            print(f"{name=}, acc={correct / total}")


if __name__ == "__main__":
    main()
