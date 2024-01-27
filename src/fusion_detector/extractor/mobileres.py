from enum import Enum
from typing import *

import torch
from torch import nn

from .. import misc
from ..thirdparty.pytorch_cifar10 import module as M
from .abstract import AbstractFeatureExtractor, layer_of

__all__ = ["ResNetKind", "MobileResExtractor"]


class ResNetKind(Enum):
    RESNET_18 = (M.resnet18, "avgpool")  # [-1, 512, 1, 1]
    RESNET_34 = (M.resnet34, "avgpool")  # [-1, 512, 1, 1]
    RESNET_50 = (M.resnet50, "avgpool")  # [-1, 2048, 1, 1]


# MobileNetV2 -- ReLU6 -- features.18.2 -- [-1, 1280, 28, 28]
class MobileResExtractor(AbstractFeatureExtractor):
    MOBILENET_V2_LAYER = "features.18.2"

    def __init__(self, resnet_kind: ResNetKind) -> None:
        constructor, pattern = resnet_kind.value
        self.resnet = constructor(pretrained=True)
        self.resnet_layer = layer_of(self.resnet, pattern)
        self.mobilenet = M.mobilenet_v2(pretrained=True)
        self.mobilenet_layer = layer_of(
            self.mobilenet,
            MobileResExtractor.MOBILENET_V2_LAYER,
        )
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Freeze the model so that its parameters won't be updated.
        super().__init__(self.resnet, self.mobilenet)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        # [-1, 2048, 1, 1]
        resnet_features: torch.Tensor
        with misc.LayerOutputValueCollector(self.resnet_layer) as resnet_collector:
            self.resnet(x)
            resnet_features = resnet_collector.value

        # [-1, 1280, 28, 28]
        mobilenet_features: torch.Tensor
        with misc.LayerOutputValueCollector(
            self.mobilenet_layer
        ) as mobilenet_collector:
            self.mobilenet(x)
            mobilenet_features = mobilenet_collector.value
        # [-1, 1280, 1, 1]
        mobilenet_features = self.average_pool(mobilenet_features)
        # [-1, 2048+1280, 1, 1]
        return torch.cat((resnet_features, mobilenet_features), dim=1)
