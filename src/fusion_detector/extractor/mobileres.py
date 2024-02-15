from enum import Enum
from typing import *

import torch
from torch import nn

from ..thirdparty.pytorch_cifar10 import module as M
from .abstract import AbstractFeatureExtractor, IntermediateLayerFeatureExtractor

__all__ = ["ResNetKind", "MobileResExtractor"]

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ResNetKind(Enum):
    RESNET_18 = (M.resnet18, "avgpool")  # [-1, 512, 1, 1]
    RESNET_34 = (M.resnet34, "avgpool")  # [-1, 512, 1, 1]
    RESNET_50 = (M.resnet50, "avgpool")  # [-1, 2048, 1, 1]


# MobileNetV2 -- ReLU6 -- features.18.2 -- [-1, 1280, 28, 28]
class MobileResExtractor(AbstractFeatureExtractor):
    MOBILENET_V2_LAYER = "features.18.2"

    def __init__(self, resnet_kind: ResNetKind) -> None:
        constructor, pattern = resnet_kind.value
        self.resnet = IntermediateLayerFeatureExtractor(
            constructor(pretrained=True, device=DEFAULT_DEVICE),
            pattern,
        )
        self.mobilenet = IntermediateLayerFeatureExtractor(
            M.mobilenet_v2(pretrained=True, device=DEFAULT_DEVICE),
            MobileResExtractor.MOBILENET_V2_LAYER,
        )
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        # [-1, 2048, 1, 1]
        resnet_features = self.resnet(x)

        # [-1, 1280, 28, 28] if ResNet50
        mobilenet_features = self.mobilenet(x)
        # [-1, 1280, 1, 1]
        mobilenet_features = self.average_pool(mobilenet_features)

        # [-1, 2048+1280, 1, 1] if ResNet50
        return torch.cat((resnet_features, mobilenet_features), dim=1)
