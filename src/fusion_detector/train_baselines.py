# This module should not be imported.
if __name__ != "__main__":
    import pathlib

    filename = pathlib.PurePath(__file__).name
    raise ModuleNotFoundError(f"`{filename}` is a script, not a lib.")

from typing import *

import constants
import misc


def main():
    modules: List[misc.ModuleProxy] = []
    for module in constants.PARTICIPATED_MODULES:
        if not isinstance(module, misc.ModuleProxy):
            module = misc.ModuleProxy(module)
        modules.append(module)


# Guideline recommended Main Guard.
if __name__ == "__main__":
    main()
