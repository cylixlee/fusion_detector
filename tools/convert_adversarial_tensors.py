import pathlib
import pickle
import sys
from typing import *

import torch
from PIL.Image import Image
from tools_common import PROJECT_DIRECTORY
from torchvision import transforms
from tqdm import tqdm

project_path = str(PROJECT_DIRECTORY)
if project_path not in sys.path:
    sys.path.insert(0, project_path)


def convert(pt: List[torch.Tensor]) -> List[Tuple[Image, int]]:
    imagefy = transforms.ToPILImage()
    image_tensor = pt[0]
    label_tensor = pt[1]

    data: List[Tuple[Image, int]] = []
    for index in range(image_tensor.shape[0]):
        single_image = imagefy(image_tensor[index])
        single_label = int(label_tensor[index])
        data.append((single_image, single_label))
    return data


def convert_save_directory(source: pathlib.Path, dest: pathlib.Path) -> None:
    if not dest.exists():
        dest.mkdir(parents=True)
    for entry in source.iterdir():
        assert entry.is_file()
        with open((dest / entry.stem).with_suffix(".segment"), "wb") as file:
            pt = torch.load(entry, map_location="cpu")
            segment = convert(pt)
            pickle.dump(segment, file)


ATTACKS = ["FGSM", "PGD"]
VICTIMS = [
    "densenet121",
    "googlenet",
    "inception_v3",
    "mobilenet_v2",
    "resnet50",
    "vgg19_bn",
]


def main():
    tensorset_path = PROJECT_DIRECTORY / "data" / "datasets" / "CIFAR10Adversarial"
    imageset_path = (
        PROJECT_DIRECTORY / "data" / "datasets" / "CIFAR10AdversarialImageSet"
    )
    for attack in tqdm(ATTACKS, desc="Attacks"):
        for victim in tqdm(VICTIMS, desc="Victim"):
            convert_save_directory(
                tensorset_path / attack / victim,
                imageset_path / attack / victim,
            )


if __name__ == "__main__":
    main()
