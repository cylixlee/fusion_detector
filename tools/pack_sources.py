# Pack sources to deploy on cloud GPUs.

import pathlib
import tarfile

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


def main():
    # Clean cache first.
    import clean_cache

    clean_cache.main()

    with tarfile.TarFile(PACKED_SOURCE_PATH, mode="w") as tar:
        project_path = pathlib.Path(PROJECT_DIRECTORY)
        for entry in project_path.glob("*"):
            skip = False
            for exclusive in EXCLUDSIVE_ENTRIES:
                if entry.match(exclusive):
                    skip = True
                    break
            if not skip:
                tar.add(entry.relative_to(project_path))


# Guideline recommended Main Guard.
if __name__ == "__main__":
    main()
