from enum import Enum
from typing import *

import torch
from torch import nn

from ..thirdparty.pytorch_cifar10 import module as M
from .abstract import AbstractFeatureExtractor, IntermediateLayerFeatureExtractor

__all__ = [
    "ResNetKind",
    "MobileResConcatExtractor",
    "MobileResSeparateExtractor",
]

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ResNetKind(Enum):
    # RESNET_18 = (M.resnet18, "avgpool")  # [-1, 512, 1, 1]
    # RESNET_34 = (M.resnet34, "avgpool")  # [-1, 512, 1, 1]
    RESNET_50 = (M.resnet50, "layer2.3")  # [-1, 512, 8, 8]
    RESNET_50_CONV = (M.resnet50, "layer2.3")  # [-1, 512, 8, 8]


class MobileResConcatExtractor(AbstractFeatureExtractor):
    MOBILENET_V2_LAYER = "features.7.conv.0.1"  # [-1, 192, 16, 16]

    def __init__(self, resnet_kind: ResNetKind) -> None:
        constructor, pattern = resnet_kind.value
        self.resnet = IntermediateLayerFeatureExtractor(
            constructor(pretrained=True, device=DEFAULT_DEVICE).to(DEFAULT_DEVICE),
            pattern,
        )
        self.mobilenet = IntermediateLayerFeatureExtractor(
            M.mobilenet_v2(pretrained=True, device=DEFAULT_DEVICE).to(DEFAULT_DEVICE),
            self.__class__.MOBILENET_V2_LAYER,
        )
        self.resnet_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mobilenet_avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        # [-1, 512, 8, 8]
        resnet_features = self.resnet(x)
        # [-1, 512, 1, 1]
        resnet_features = self.resnet_avgpool(resnet_features)

        # [-1, 192, 16, 16]
        mobilenet_features = self.mobilenet(x)
        # [-1, 192, 1, 1]
        mobilenet_features = self.mobilenet_avgpool(mobilenet_features)

        # [-1, 512+192, 1, 1]
        return torch.cat((resnet_features, mobilenet_features), dim=1)


class MobileResSeparateExtractor(AbstractFeatureExtractor):
    MOBILENET_V2_LAYER = "features.7.conv.0.1"  # [-1, 192, 16, 16]

    def __init__(self, resnet_kind: ResNetKind) -> None:
        constructor, pattern = resnet_kind.value
        self.resnet = IntermediateLayerFeatureExtractor(
            constructor(pretrained=True, device=DEFAULT_DEVICE).to(DEFAULT_DEVICE),
            pattern,
        )  # [-1, 512, 8, 8]
        self.mobilenet = IntermediateLayerFeatureExtractor(
            M.mobilenet_v2(pretrained=True, device=DEFAULT_DEVICE).to(DEFAULT_DEVICE),
            self.__class__.MOBILENET_V2_LAYER,
        )  # [-1, 192, 16, 16]

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        # [-1, 512, 8, 8]
        resnet_features = self.resnet(x)

        # [-1, 192, 16, 16]
        mobilenet_features = self.mobilenet(x)

        return (resnet_features, mobilenet_features)
