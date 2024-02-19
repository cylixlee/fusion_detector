from enum import Enum
from typing import *

import torch
from torch import nn

from ..thirdparty.pytorch_cifar10 import module as M
from .abstract import AbstractFeatureExtractor, IntermediateLayerFeatureExtractor

__all__ = [
    "ResNetKind",
    "MobileResLinearExtractor",
    "MobileResConvExtractor",
]

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ResNetKind(Enum):
    RESNET_18 = (M.resnet18, "avgpool")  # [-1, 512, 1, 1]
    RESNET_34 = (M.resnet34, "avgpool")  # [-1, 512, 1, 1]
    RESNET_50 = (M.resnet50, "avgpool")  # [-1, 2048, 1, 1]
    RESNET_50_CONV = (M.resnet50, "layer4")  # [-1, 2048, 2, 2]


class MobileResLinearExtractor(AbstractFeatureExtractor):
    MOBILENET_V2_LAYER = "features.18.2"

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
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        # [-1, 2048, 1, 1]
        resnet_features = self.resnet(x)

        # [-1, 1280, 4, 4]
        mobilenet_features = self.mobilenet(x)
        # [-1, 1280, 1, 1]
        mobilenet_features = self.average_pool(mobilenet_features)

        # [-1, 2048+1280, 1, 1]
        return torch.cat((resnet_features, mobilenet_features), dim=1)


class MobileResConvExtractor(AbstractFeatureExtractor):
    MOBILENET_V2_LAYER = "features.18.2"

    def __init__(self, resnet_kind: ResNetKind) -> None:
        constructor, pattern = resnet_kind.value
        self.resnet = IntermediateLayerFeatureExtractor(
            constructor(pretrained=True, device=DEFAULT_DEVICE).to(DEFAULT_DEVICE),
            pattern,
        )
        self.resnetconv = nn.Conv2d(2048, 640, 2).to(
            DEFAULT_DEVICE
        )  # to [-1, 640, 1, 1]
        self.mobilenet = IntermediateLayerFeatureExtractor(
            M.mobilenet_v2(pretrained=True, device=DEFAULT_DEVICE).to(DEFAULT_DEVICE),
            self.__class__.MOBILENET_V2_LAYER,
        )
        self.mobileconv = nn.Conv2d(1280, 640, 4).to(
            DEFAULT_DEVICE
        )  # to [-1, 640, 1, 1]

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (
            *self.resnetconv.parameters(),
            *self.mobileconv.parameters(),
        )

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        # [-1, 2048, 2, 2]
        resnet_features = self.resnet(x)
        # [-1, 640, 1, 1]
        resnet_features = self.resnetconv(resnet_features)

        # [-1, 1280, 4, 4]
        mobilenet_features = self.mobilenet(x)
        # [-1, 640, 1, 1]
        mobilenet_features = self.mobileconv(mobilenet_features)

        # [-1, 1280, 1, 1]
        return torch.cat((resnet_features, mobilenet_features), dim=1)
