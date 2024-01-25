import pathlib

if __name__ != "__main__":
    filename = pathlib.PurePath(__file__).name
    raise ModuleNotFoundError(f"`{filename}` is a tool script, not a lib.")

import sys

SCRIPT_DIRECTORY = pathlib.PurePath(__file__).parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent

# Add project source directory into sys.path, to import modules without messing up
# with source files.
source_path = str(PROJECT_DIRECTORY / "src")
if source_path not in sys.path:
    sys.path.insert(0, source_path)

from fusion_detector.constants import PARTICIPATED_MODULES
from fusion_detector.misc.module import ModuleProxy

SUMMARY_DIRECTORY = PROJECT_DIRECTORY / "metadata" / "summary"
STRUCTURE_DIRECTORY = PROJECT_DIRECTORY / "metadata" / "structure"
FILE_SUFFIX = ".txt"


def main():
    for name, module in PARTICIPATED_MODULES.items():
        if not isinstance(module, ModuleProxy):
            module = ModuleProxy(module)
        summary_path = (SUMMARY_DIRECTORY / name).with_suffix(FILE_SUFFIX)
        structure_path = (STRUCTURE_DIRECTORY / name).with_suffix(FILE_SUFFIX)
        module.summary(path=summary_path)
        module.structure(path=structure_path)


# Main guard
if __name__ == "__main__":
    main()
