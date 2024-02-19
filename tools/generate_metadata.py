# Generate module metadata to inspect the structure and parameters.

import os
import pathlib
import sys
from typing import *

import inclconf
import torchsummary
from tqdm import tqdm

inclconf.configure_includepath()

import src.fusion_detector.thirdparty.pytorch_cifar10.module as M

SUMMARY_DIRECTORY = inclconf.PROJECT_DIRECTORY / "metadata" / "summary"
STRUCTURE_DIRECTORY = inclconf.PROJECT_DIRECTORY / "metadata" / "structure"
FILE_SUFFIX = ".txt"


class PossibleRedirectStream(ContextManager):
    def __init__(
        self,
        filename: Optional[os.PathLike],
    ) -> None:
        self.path: Optional[pathlib.Path] = None
        if filename is None:
            return
        self.path = pathlib.Path(filename).with_suffix(FILE_SUFFIX)

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


def main():
    for name, module in tqdm(
        M.all_classifiers.items(), desc="Generate Metadata", leave=False
    ):
        summary_path = (SUMMARY_DIRECTORY / name).with_suffix(FILE_SUFFIX)
        with PossibleRedirectStream(summary_path):
            torchsummary.summary(module, input_size=(3, 32, 32))
        structure_path = (STRUCTURE_DIRECTORY / name).with_suffix(FILE_SUFFIX)
        with PossibleRedirectStream(structure_path):
            print(module)


# Guideline Recommended Main Guard
if __name__ == "__main__":
    main()
