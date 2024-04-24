from typing import *

import torch
from torch import nn

__all__ = ["LinearAdversarialClassifier", "ConvAdversarialClassifier"]


class LinearAdversarialClassifier(nn.Module):
    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.fc = nn.Sequential(
            nn.Linear(feature_channels, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(-1, self.feature_channels))


class ConvAdversarialClassifier(nn.Module):
    def __init__(self, *shapes: Tuple[int, int], out_channels: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.feature_channels = len(shapes) * out_channels
        for in_channels, kernel in shapes:
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel))
        self.classifier = LinearAdversarialClassifier(self.feature_channels)

    def forward(self, x: Iterable[torch.Tensor]) -> torch.Tensor:
        flattened = []
        for index, feature_graph in enumerate(x):
            flattened.append(self.convs[index](feature_graph))
        fusioned = torch.concat(flattened, dim=1)
        return self.classifier(fusioned)
