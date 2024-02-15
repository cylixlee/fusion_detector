from abc import ABC, abstractmethod

from torch.utils.data.dataloader import DataLoader

__all__ = ["AbstractDataSource"]


class AbstractDataSource(ABC):
    @property
    @abstractmethod
    def trainset(self) -> DataLoader: ...

    @property
    @abstractmethod
    def testset(self) -> DataLoader: ...
