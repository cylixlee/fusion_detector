# Selected Attack Methods:
#   R-FGSM (default)
#   APGD (default)
#   CW (lr=0.05)
#   Pixle (pixel_mapping=similarity_random)
#
# Selected Victims:
#   Resnet50, Densenet121, MobileNetV2, GoogLeNet

from typing import *

import inclconf
import torch
import torchattacks
import torchvision.transforms.functional as V
from torch import nn
from torchattacks.attack import Attack
from tqdm import tqdm

inclconf.configure_includepath()

import src.fusion_detector.thirdparty.pytorch_cifar10.module as M
from src.fusion_detector import datasource, misc

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
    # "vanilla": canonicalized(torchattacks.VANILA),
    "rfgsm": canonicalized(torchattacks.RFGSM),
    "apgd": canonicalized(torchattacks.APGD),
    "cw": canonicalized(torchattacks.CW, lr=0.05),
    "pixle": canonicalized(torchattacks.Pixle, pixel_mapping="similarity_random"),
}
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS: Dict[str, nn.Module] = {
    "resnet50": M.resnet50(pretrained=True, device=DEFAULT_DEVICE),
    "densenet121": M.densenet121(pretrained=True, device=DEFAULT_DEVICE),
    "mobilenet_v2": M.mobilenet_v2(pretrained=True, device=DEFAULT_DEVICE),
    "googlenet": M.googlenet(pretrained=True, device=DEFAULT_DEVICE),
}

SAVE_DIRECTORY = inclconf.PROJECT_DIRECTORY / "data" / "datasets" / "CIFAR10Adversarial"


def main():
    data = datasource.CifarDataSource(32)
    for attack_name, attack_constructor in tqdm(
        ATTACKS.items(), desc="Attack", leave=False
    ):
        for model_name, model in tqdm(MODELS.items(), desc="Model", leave=False):
            model.to(DEFAULT_DEVICE).eval()
            attack = attack_constructor(model)

            # Save adversarial train set.
            lst = misc.SegmentedSerializableList(
                SAVE_DIRECTORY / attack_name / model_name, "train", 10000
            )
            for x, label in tqdm(data.trainset, desc="Train set", leave=False):
                x, label = x.to(DEFAULT_DEVICE), label.to(DEFAULT_DEVICE)
                adversarial = attack(x, label)
                for index in range(adversarial.shape[0]):
                    adversarial_image = V.to_pil_image(
                        datasource.denormalize_cifar(
                            adversarial[index].unsqueeze(0)
                        ).squeeze()
                    )
                    category = label[index].item()
                    lst.add((adversarial_image, category))
            lst.save()

            # Save adversarial test set.
            lst = misc.SegmentedSerializableList(
                SAVE_DIRECTORY / attack_name / model_name, "test", 10000
            )
            for x, label in tqdm(data.testset, desc="Test set", leave=False):
                x, label = x.to(DEFAULT_DEVICE), label.to(DEFAULT_DEVICE)
                adversarial = attack(x, label)
                for index in range(adversarial.shape[0]):
                    adversarial_image = V.to_pil_image(
                        datasource.denormalize_cifar(
                            adversarial[index].unsqueeze(0)
                        ).squeeze()
                    )
                    category = label[index].item()
                    lst.add((adversarial_image, category))
            lst.save()


# Guideline Recommended Main Guard
if __name__ == "__main__":
    main()
