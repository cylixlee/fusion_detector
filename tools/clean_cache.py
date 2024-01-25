import os
import pathlib
from typing import *

import colorama
from colorama import Fore, Back

colorama.init(autoreset=True)

SCRIPT_PATH = pathlib.PurePath(__file__).parent
PROJECT_PATH = SCRIPT_PATH.parent
RECURSIVE_CLEAN_TARGETS = ["src", "tests"]


def _background(color: str, text: str) -> str:
    return Fore.BLACK + color + f" {text} "


def _print_enter(text: str) -> None:
    print(_background(Back.WHITE, " ENTER "), text)


def _print_clean(text: str) -> None:
    print(_background(Back.GREEN, " CLEAN "), text)


def _recursive_remove_directory(directory: Union[pathlib.Path, os.PathLike]) -> None:
    if not isinstance(directory, pathlib.Path):
        directory = pathlib.Path(directory)
    for entry in directory.iterdir():
        if entry.is_dir():
            _recursive_remove_directory(entry)
        else:
            entry.unlink()
    directory.rmdir()


def _clean(directory: Union[pathlib.Path, os.PathLike]) -> None:
    if not isinstance(directory, pathlib.Path):
        directory = pathlib.Path(directory)
    assert directory.is_dir()
    for entry in directory.iterdir():
        if entry.is_dir() and entry.name == "__pycache__":
            _recursive_remove_directory(entry)
            _print_clean(str(entry))


def _recursive_clean(directory: Union[pathlib.Path, os.PathLike]) -> None:
    if not isinstance(directory, pathlib.Path):
        directory = pathlib.Path(directory)
    for entry in directory.iterdir():
        if entry.is_dir():
            if entry.name == "__pycache__":
                _recursive_remove_directory(entry)
                _print_clean(str(entry))
            else:
                _recursive_clean(entry)


def main():
    _print_enter("==SCRIPT== path " + str(SCRIPT_PATH))
    _clean(SCRIPT_PATH)
    _print_enter("==PROJECT== path " + str(PROJECT_PATH))
    _clean(PROJECT_PATH)
    for target in RECURSIVE_CLEAN_TARGETS:
        _print_enter(f"==PROJECT==/{target}")
        _recursive_clean(PROJECT_PATH / target)


if __name__ == '__main__':
    main()
