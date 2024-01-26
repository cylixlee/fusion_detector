from abc import ABC, abstractmethod

from torch.utils.data.dataloader import DataLoader

__all__ = ["AbstractDataset"]


class AbstractDataset(ABC):
    @property
    @abstractmethod
    def trainset(self) -> DataLoader:
        ...

    @property
    @abstractmethod
    def testset(self) -> DataLoader:
        ...
