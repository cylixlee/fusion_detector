import time
from typing import *

import inclconf
import pandas as pd
import torch
import torchattacks
from inclconf import PROJECT_DIRECTORY
from torch import nn
from torchattacks.attack import Attack
from tqdm import tqdm

inclconf.configure_includepath()

from src.fusion_detector import datasource
from src.fusion_detector.thirdparty.pytorch_cifar10 import module as M


class BenchmarkReportCard(object):
    def __init__(self) -> None:
        self.bidimensional: Dict[str, Dict[str, float]] = dict()

    def add(self, victim: str, measure: str, value: float) -> None:
        if victim not in self.bidimensional.keys():
            self.bidimensional[victim] = dict()
        self.bidimensional[victim][measure] = value

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.bidimensional)


AnyAttackConstructor = Callable[..., Attack]
CanonicalAttackConstructor = Callable[[nn.Module], Attack]


def canonicalized(
    attack_constructor: AnyAttackConstructor, *args, **kwargs
) -> CanonicalAttackConstructor:
    def _wrapper(module: nn.Module) -> Attack:
        attack = attack_constructor(module, *args, **kwargs)
        attack.set_normalization_used(datasource.CIFAR_MEAN, datasource.CIFAR_STD)
        return attack

    return _wrapper


ATTACKS: Dict[str, CanonicalAttackConstructor] = {
    "rfgsm default": canonicalized(torchattacks.RFGSM),
    "apgd default": canonicalized(torchattacks.APGD),
    "cw default": canonicalized(torchattacks.CW),
    "pixle default": canonicalized(torchattacks.Pixle),
}
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS: Dict[str, nn.Module] = {
    "vgg19": M.vgg19_bn(pretrained=True, device=DEFAULT_DEVICE),
    "resnet50": M.resnet50(pretrained=True, device=DEFAULT_DEVICE),
    "densenet121": M.densenet121(pretrained=True, device=DEFAULT_DEVICE),
    "mobilenet_v2": M.mobilenet_v2(pretrained=True, device=DEFAULT_DEVICE),
    "googlenet": M.googlenet(pretrained=True, device=DEFAULT_DEVICE),
    "inception_v3": M.inception_v3(pretrained=True, device=DEFAULT_DEVICE),
}

BENCHMARK_SAVE_DIRECTORY = PROJECT_DIRECTORY / "metadata" / "benchmark" / "CIFAR10"


def main():
    timestamp = int(time.time())
    x, label = datasource.CifarBatchDataSource(32).batch()
    x, label = x.to(DEFAULT_DEVICE), label.to(DEFAULT_DEVICE)
    card = BenchmarkReportCard()
    for attack_name, attack_constructor in tqdm(
        ATTACKS.items(), desc="Attack", leave=False
    ):
        for model_name, model in tqdm(MODELS.items(), desc="Model", leave=False):
            model.eval()
            attack = attack_constructor(model)
            adversarial = attack(x, label)
            y = model(adversarial)

            # Collect statistics
            correct = torch.eq(torch.argmax(y, dim=1), label).sum().item()
            total = len(label)
            card.add(model_name, attack_name, correct / total)
    dataframe = card.to_dataframe()
    print(dataframe)
    savepath = BENCHMARK_SAVE_DIRECTORY / str(timestamp)
    if not savepath.exists():
        savepath.mkdir(parents=True)
    dataframe.to_csv(savepath / "benchmark.csv")


# Guideline Recommended Main Guard
if __name__ == "__main__":
    main()
