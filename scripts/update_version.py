#!/usr/bin/env python3
"""Update version in pyproject.toml"""

import sys
import tomllib
import tomli_w
from pathlib import Path


def update_version(new_version: str) -> None:
    """Update the version in pyproject.toml.

    Args:
        new_version: The new version string to set
    """
    pyproject_path = Path("pyproject.toml")

    # Read current config
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    # Update version
    data["project"]["version"] = new_version

    # Write back
    with open(pyproject_path, "wb") as f:
        tomli_w.dump(data, f)

    print(f"Updated version to {new_version}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_version.py VERSION")
        sys.exit(1)

    update_version(sys.argv[1])
