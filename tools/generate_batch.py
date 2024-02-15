import pickle
from typing import *

import inclconf
import torch
import torchvision.transforms.functional as V
from inclconf import PROJECT_DIRECTORY
from PIL.Image import Image

inclconf.configure_includepath()

from src.fusion_detector import datasource, misc

BATCHSIZE = 32
SAVEPATH = PROJECT_DIRECTORY / "data" / "datasets" / "CIFAR10Batch"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2471, 0.2435, 0.2616)


# Guideline Recommended Main Guard
def main():
    if not SAVEPATH.exists():
        SAVEPATH.mkdir(parents=True)
    with open(SAVEPATH / "batch.pickle", "wb") as file:
        source = datasource.CifarDataSource(BATCHSIZE)
        batch = next(iter(source.testset))
        del source  # Release memory ASAP

        batch[0] = misc.denormalize(batch[0], MEAN, STD)
        pairs: List[Tuple[Image, int]] = []
        for index in range(BATCHSIZE):
            x: torch.Tensor = batch[0][index]
            y: torch.Tensor = batch[1][index]
            image = V.to_pil_image(x)
            category = int(y.item())
            pairs.append((image, category))
        pickle.dump(pairs, file)


if __name__ == "__main__":
    main()
