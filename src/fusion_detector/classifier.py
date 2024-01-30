import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["LinearAdversarialClassifier"]


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
        x = x.view(-1, self.feature_channels)
        binary = self.fc(x)
        return binary
