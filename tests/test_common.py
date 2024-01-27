import pathlib

__all__ = ["SCRIPT_DIRECTORY", "PROJECT_DIRECTORY"]

SCRIPT_DIRECTORY = pathlib.PurePath(__file__).parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
