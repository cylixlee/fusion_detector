import pathlib
import sys

import torch
from tools_common import PROJECT_DIRECTORY

project_path = str(PROJECT_DIRECTORY)
if project_path not in sys.path:
    sys.path.insert(0, project_path)


from src.fusion_detector import dataset

CIFAR_DATASET_ROOT = PROJECT_DIRECTORY / "data" / "datasets" / "CIFAR10"
CIFAR_BATCH_ROOT = PROJECT_DIRECTORY / "data" / "datasets" / "CIFAR10Batch"


def main():
    cifar = dataset.NormalizedCifarDataset(32)
    batch = next(iter(cifar.testset))
    savepath = pathlib.Path(CIFAR_BATCH_ROOT)
    if not savepath.exists():
        savepath.mkdir(parents=True)
    torch.save(batch, str(savepath / "batch.pt"))


# Guideline recommended Main Guard
if __name__ == "__main__":
    main()
