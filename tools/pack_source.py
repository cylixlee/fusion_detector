# Pack sources to deploy on cloud GPUs. Following types of files are ignored:
#   - Private files (starting with `.`)
#   - Data files
#   - Metadata files (ignored when pack, but updated to GitHub.)
#   - Tests
#   - Notebooks
#   - Log (local log information is not needed when using cloud GPU.)
#   - Markdown documents
# Additionally, Python compilation cache (anything inside a `__pycache__` directory) is
# ignored.

import pathlib
import tarfile
from typing import *

SCRIPT_DIRECTORY = pathlib.PurePath(__file__).parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent

PACKED_SOURCE_PATH = PROJECT_DIRECTORY.with_suffix(".tar")

EXCLUDSIVE_ENTRIES = [
    ".*",
    "data",
    "metadata",
    "tests",
    "notebooks",
    "log",
    "*.md",
]


def pycache_filter(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
    if "__pycache__" in info.name:
        print("Skip:", info.name)
        return None
    return info


def main():
    with tarfile.TarFile(PACKED_SOURCE_PATH, mode="w") as tar:
        project_path = pathlib.Path(PROJECT_DIRECTORY)
        for entry in project_path.glob("*"):
            relative_path = entry.relative_to(project_path)
            skip = False
            for exclusive in EXCLUDSIVE_ENTRIES:
                if entry.match(exclusive):
                    skip = True
                    print("Skip (Entry):", relative_path)
                    break
            if not skip:
                print("Pack:", relative_path)
                tar.add(relative_path, filter=pycache_filter)


# Guideline recommended Main Guard.
if __name__ == "__main__":
    main()
