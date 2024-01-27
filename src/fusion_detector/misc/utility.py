import os
import pathlib
import sys
from typing import *

import torch

__all__ = [
    "ListOrElement",
    "TensorTransformAction",
    "attribute_of",
    "ListOrElementProxy",
    "PossibleRedirectStream",
    "AccuracyRecorder",
]

TEXT_FILE_SUFFIX = ".txt"

# Generic type variables
_T = TypeVar("_T")

# Type aliases
ListOrElement = Union[_T, List[_T]]
TensorTransformAction = Callable[[torch.Tensor], torch.Tensor]


def attribute_of(obj: Any, *chained_attributes: str) -> Any:
    """A utility function to get type-unknown object's chained attribute.

    Just making the compiler happy :)
    """
    for attribute in chained_attributes:
        obj = getattr(obj, attribute)
    return obj


class ListOrElementProxy(Generic[_T]):
    def __init__(self, data: ListOrElement[_T]) -> None:
        self.data: ListOrElement[_T] = data

    def __len__(self) -> int:
        if isinstance(self.data, List):
            return len(self.data)
        return 1

    def assert_possible_length(self, length: int) -> None:
        if isinstance(self.data, List):
            assert len(self.data) == length

    def assert_length_compatible(self, target: Sized) -> None:
        self.assert_possible_length(len(target))

    def decide(self, index: int) -> _T:
        if isinstance(self.data, List):
            return self.data[index]
        return self.data

    def tolist(self) -> List[_T]:
        if isinstance(self.data, List):
            return self.data
        return [self.data]


class PossibleRedirectStream(object):
    def __init__(
        self,
        filename: Optional[os.PathLike],
    ) -> None:
        self.path: Optional[pathlib.Path] = None
        if filename is None:
            return
        self.path = pathlib.Path(filename).with_suffix(TEXT_FILE_SUFFIX)

    def __enter__(self) -> None:
        if self.path is None:
            return
        # Save the original stream.
        self.standard_output = sys.stdout
        # Create parent directory if any.
        parent_directory = self.path.parent
        if not parent_directory.exists():
            parent_directory.mkdir(parents=True)
        self.file = self.path.open("w")
        # Redirect stream to the file.
        sys.stdout = self.file

    def __exit__(self, *args: Any) -> None:
        if self.path is None:
            return
        # Close file.
        self.file.close()
        # Restore system stream.
        sys.stdout = self.standard_output


class AccuracyRecorder(object):
    def __init__(self) -> None:
        self._correct: int = 0
        self._total: int = 0

    @property
    def correct(self) -> int:
        return self._correct

    @property
    def total(self) -> int:
        return self._total

    def push(self, correct: int, total: int) -> None:
        self._correct += correct
        self._total += total

    def pop(self) -> float:
        accuracy = self._correct / self._total
        self._correct = self._total = 0
        return accuracy
