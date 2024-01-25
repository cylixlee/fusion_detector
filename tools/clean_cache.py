if __name__ != "__main__":
    import pathlib

    filename = pathlib.PurePath(__file__).name
    raise ModuleNotFoundError(f"`{filename}` is a tool script, not a lib.")

import os
import pathlib
from typing import *

import colorama
from colorama import Back, Fore

colorama.init(autoreset=True)

SCRIPT_DIRECTORY = pathlib.PurePath(__file__).parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
RECURSIVE_CLEAN_TARGETS = ["src", "tests"]
PYTHON_CACHE_DIRECTORY_NAME = "__pycache__"


def highlight(back: str, text: str) -> str:
    return Fore.BLACK + back + f" {text} "


def entering(*text: Any) -> None:
    print(highlight(Back.WHITE, " ENTER "), *text)


def cleaning(*text: Any) -> None:
    print(highlight(Back.GREEN, " CLEAN "), *text)


def recursive_remove_directory(directory: Union[pathlib.Path, os.PathLike]) -> None:
    if not isinstance(directory, pathlib.Path):
        directory = pathlib.Path(directory)
    for entry in directory.iterdir():
        if entry.is_dir():
            recursive_remove_directory(entry)
        else:
            entry.unlink()
    directory.rmdir()


def clean_directory(directory: Union[pathlib.Path, os.PathLike]) -> None:
    if not isinstance(directory, pathlib.Path):
        directory = pathlib.Path(directory)
    assert directory.is_dir()
    for entry in directory.iterdir():
        if entry.is_dir() and entry.name == PYTHON_CACHE_DIRECTORY_NAME:
            recursive_remove_directory(entry)
            cleaning(entry)


def recursive_clean_directory(directory: Union[pathlib.Path, os.PathLike]) -> None:
    if not isinstance(directory, pathlib.Path):
        directory = pathlib.Path(directory)
    for entry in directory.iterdir():
        if entry.is_dir():
            if entry.name == PYTHON_CACHE_DIRECTORY_NAME:
                recursive_remove_directory(entry)
                cleaning(entry)
            else:
                recursive_clean_directory(entry)


def main():
    entering("==SCRIPT== path", SCRIPT_DIRECTORY)
    clean_directory(SCRIPT_DIRECTORY)
    entering("==PROJECT== path", PROJECT_DIRECTORY)
    clean_directory(PROJECT_DIRECTORY)
    for target in RECURSIVE_CLEAN_TARGETS:
        entering(f"==PROJECT==/{target}")
        recursive_clean_directory(PROJECT_DIRECTORY / target)


if __name__ == "__main__":
    main()
