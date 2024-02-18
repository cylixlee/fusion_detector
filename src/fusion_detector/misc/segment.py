import pathlib
import pickle
from typing import *

__all__ = ["SegmentedSerializableList"]

T = TypeVar("T", covariant=True)


class SegmentedSerializableList(Generic[T]):
    def __init__(
        self,
        directory: pathlib.Path,
        prefix: str,
        threshold: Optional[int] = None,
    ) -> None:
        self._directory: pathlib.Path = directory
        self._prefix: str = prefix
        self._threshold: Optional[int] = threshold
        self._segments: int = 0
        self._total: int = 0
        self._immutable = False
        if self.metafile_path.exists():
            self._immutable = True
            with self.metafile_path.open("rb") as file:
                (
                    self._threshold,
                    self._segments,
                    self._total,
                ) = pickle.load(file)
        if self._threshold is None and not self._immutable:
            raise RuntimeError(
                f"requires threshold when initializing a mutable {self.__class__.__name__}"
            )
        self._data: List[T] = []
        self._current_segment_index = None

    def __getitem__(self, index: int) -> T:
        if not self._immutable:
            raise RuntimeError(
                f"cannot get items from a mutable {self.__class__.__name__}."
            )
        if index >= self._total:
            raise IndexError(f"index {index} out of range {self._total}")
        element_index = index % self._threshold
        segment_index = index // self._threshold
        if self._current_segment_index != segment_index:
            self._load_segment(segment_index)
        return self._data[element_index]

    def __len__(self) -> int:
        if not self._immutable:
            raise RuntimeError(
                f"cannot get length of a mutable {self.__class__.__name__}."
            )
        return self._total

    @property
    def immutable(self) -> bool:
        return self._immutable

    @property
    def metafile_path(self) -> pathlib.Path:
        return self._directory / f"{self._prefix}.meta.pickle"

    def add(self, data: T) -> None:
        if self._immutable:
            raise RuntimeError(
                f"immutable {self.__class__.__name__} can not be modified."
            )
        self._data.append(data)
        self._total += 1
        if len(self._data) == self._threshold:
            self._save_segment()

    def save(self) -> None:
        if self._immutable:
            raise RuntimeError(f"cannot re-save immutable {self.__class__.__name__}.")
        if len(self._data) > 0:
            self._save_segment()
        self._save_meta()

    def _save_segment(self) -> None:
        if not self._directory.exists():
            self._directory.mkdir(parents=True)
        savepath = self._directory / f"{self._prefix}_{self._segments}.pickle"
        with savepath.open("wb") as file:
            pickle.dump(self._data, file)
        self._data.clear()
        self._segments += 1

    def _save_meta(self) -> None:
        with self.metafile_path.open("wb") as file:
            pickle.dump((self._threshold, self._segments, self._total), file)

    def _load_segment(self, index: int) -> None:
        loadpath = self._directory / f"{self._prefix}_{index}.pickle"
        with loadpath.open("rb") as file:
            self._data = pickle.load(file)
        self._current_segment_index = index
