from typing import *

import inclconf
import pandas as pd
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

inclconf.configure_includepath()

import src.fusion_detector.thirdparty.pytorch_cifar10.module as M
from src.fusion_detector import datasource


class BenchmarkReportCard(object):
    def __init__(self) -> None:
        self.bidimensional: Dict[str, Dict[str, float]] = dict()

    def add(self, victim: str, measure: str, value: float) -> None:
        if victim not in self.bidimensional.keys():
            self.bidimensional[victim] = dict()
        self.bidimensional[victim][measure] = value

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.bidimensional)


def create_dataset(attack: str, victim: str) -> Dataset:
    return datasource.SegmentedAdversarialCifarDataset(
        attack,
        victim,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(datasource.CIFAR_MEAN, datasource.CIFAR_STD),
            ]
        ),
    )


DATASOURCES: Dict[str, Dataset] = {
    "apgd (densenet121)": create_dataset("apgd", "densenet121"),
    "apgd (googlenet)": create_dataset("apgd", "googlenet"),
    "apgd (mobilenet_v2)": create_dataset("apgd", "mobilenet_v2"),
    "apgd (resnet50)": create_dataset("apgd", "resnet50"),
    "cw (densenet121)": create_dataset("cw", "densenet121"),
    "cw (googlenet)": create_dataset("cw", "googlenet"),
    "cw (mobilenet_v2)": create_dataset("cw", "mobilenet_v2"),
    "cw (resnet50)": create_dataset("cw", "resnet50"),
    "rfgsm (densenet121)": create_dataset("rfgsm", "densenet121"),
    "rfgsm (googlenet)": create_dataset("rfgsm", "googlenet"),
    "rfgsm (mobilenet_v2)": create_dataset("rfgsm", "mobilenet_v2"),
    "rfgsm (resnet50)": create_dataset("rfgsm", "resnet50"),
}
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS: Dict[str, nn.Module] = {
    "densenet121": M.densenet121(pretrained=True, device=DEFAULT_DEVICE),
    "googlenet": M.googlenet(pretrained=True, device=DEFAULT_DEVICE),
    "mobilenet_v2": M.mobilenet_v2(pretrained=True, device=DEFAULT_DEVICE),
    "resnet50": M.resnet50(pretrained=True, device=DEFAULT_DEVICE),
}
SAVE_DIRECTORY = inclconf.PROJECT_DIRECTORY / "metadata" / "benchmark_transferability"


def main():
    card = BenchmarkReportCard()
    for datasource_name, dataset in tqdm(
        DATASOURCES.items(), desc="DataSource", leave=False
    ):
        loader = DataLoader(dataset, 32, num_workers=16, persistent_workers=True)
        for model_name, model in tqdm(MODELS.items(), desc="Model", leave=False):
            model.to(DEFAULT_DEVICE).eval()
            correct = 0
            total = 0
            with tqdm(total=len(loader), desc="Benchmark", leave=False) as progress:
                for x, label in loader:
                    x, label = x.to(DEFAULT_DEVICE), label.to(DEFAULT_DEVICE)
                    y = model(x)
                    predict = torch.argmax(y, dim=1)
                    c = torch.eq(predict, label).sum().item()
                    t = len(label)
                    correct += c
                    total += t
                    progress.set_postfix({"acc": correct / total})
                    progress.update()
            card.add(model_name, datasource_name, correct / total)
    dataframe = card.to_dataframe()
    print(dataframe)
    if not SAVE_DIRECTORY.exists():
        SAVE_DIRECTORY.mkdir(parents=True)
    savepath = SAVE_DIRECTORY / "benchmark.csv"
    dataframe.to_csv(savepath)


# Guideline Recommended Main Guard
if __name__ == "__main__":
    main()
