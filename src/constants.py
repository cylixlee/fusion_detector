from typing import *

from torch import nn
from torchvision import models

PARTICIPATED_MODULES: Dict[str, nn.Module] = {
    "resnet18": models.resnet18(),
    "vgg11": models.vgg11(),
    "mobilenet_v2": models.mobilenet_v2(),
    "alexnet": models.alexnet(),
}
