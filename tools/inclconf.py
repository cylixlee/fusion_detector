# Project Includepath Configuration

import pathlib
import sys

SCRIPT_DIRECTORY = pathlib.Path(__file__).parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent


def configure_includepath():
    include_path = str(PROJECT_DIRECTORY)
    if include_path not in sys.path:
        sys.path.insert(0, include_path)
