# Generate module metadata to inspect the structure and parameters.

import sys

import torchsummary
from tools_common import PROJECT_DIRECTORY

# Add project source directory into sys.path, to import modules without messing up
# with source files.
project_path = str(PROJECT_DIRECTORY)
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from src.constants import PARTICIPATED_MODULES
from src.fusion_detector import misc

SUMMARY_DIRECTORY = PROJECT_DIRECTORY / "metadata" / "summary"
STRUCTURE_DIRECTORY = PROJECT_DIRECTORY / "metadata" / "structure"
FILE_SUFFIX = ".txt"


def main():
    for name, module in PARTICIPATED_MODULES.items():
        summary_path = (SUMMARY_DIRECTORY / name).with_suffix(FILE_SUFFIX)
        with misc.PossibleRedirectStream(summary_path):
            torchsummary.summary(module, input_size=(3, 224, 224))
        structure_path = (STRUCTURE_DIRECTORY / name).with_suffix(FILE_SUFFIX)
        with misc.PossibleRedirectStream(structure_path):
            print(module)


# Main guard
if __name__ == "__main__":
    main()
