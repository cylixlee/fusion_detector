from typing import *

import inclconf
import torch
import torchvision.transforms.functional as V
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

inclconf.configure_includepath()

from src.fusion_detector import datasource, misc
from src.fusion_detector.thirdparty.pytorch_cifar10 import module as M

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS: Dict[str, nn.Module] = {
    "resnet50": M.resnet50(pretrained=True, device=DEFAULT_DEVICE),
    "densenet121": M.densenet121(pretrained=True, device=DEFAULT_DEVICE),
    "mobilenet_v2": M.mobilenet_v2(pretrained=True, device=DEFAULT_DEVICE),
    "googlenet": M.googlenet(pretrained=True, device=DEFAULT_DEVICE),
}
BATCH_SIZE = 256
SAVE_DIRECTORY = (
    inclconf.PROJECT_DIRECTORY / "data" / "datasets" / "CIFAR10StrongAdversarial"
)
THRESHOLD = 10000


def main():
    data = datasource.AdversarialCifarDataSource(
        BATCH_SIZE, target_transform=None, num_workers=8
    )
    emit_filtered(data.trainset, "train")
    emit_filtered(data.testset, "test")


def emit_filtered(loader: DataLoader, prefix: str):
    lst = misc.SegmentedSerializableList(SAVE_DIRECTORY, prefix, THRESHOLD)
    for x, labels in tqdm(loader, desc="Dataset", leave=False):
        x, labels = x.to(DEFAULT_DEVICE), labels.to(DEFAULT_DEVICE)
        adopted = [True for _ in range(len(labels))]
        for model in MODELS.values():
            model.to(DEFAULT_DEVICE)
            y = model(x)
            predict = torch.argmax(y, dim=1)
            equality = torch.eq(predict, labels)
            for i, equal in enumerate(equality):
                if equal:
                    adopted[i] = False
        for i, adopt in enumerate(adopted):
            if adopt:
                adversarial_image = V.to_pil_image(
                    datasource.denormalize_cifar(x[i].unsqueeze(0)).squeeze()
                )
                category = labels[i].item()
                lst.add((adversarial_image, category))
    lst.save()
    print("Saved ", prefix)


# Guideline recommended Main Guard.
if __name__ == "__main__":
    main()
