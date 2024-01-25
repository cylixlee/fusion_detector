import os
import sys
from pathlib import Path
from typing import *

import torch

__all__ = [
    "ListOrElement",
    "TensorTransformAction",
    "attribute_of",
    "ListOrElementProxy",
    "RedirectStream",
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


class RedirectStream(object):
    def __init__(
        self,
        filename: Optional[os.PathLike],
    ) -> None:
        self.path: Optional[Path] = None
        if filename is None:
            return
        self.path = Path(filename).with_suffix(TEXT_FILE_SUFFIX)

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
