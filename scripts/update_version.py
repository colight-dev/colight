#!/usr/bin/env python3
"""Update version in all package pyproject.toml files"""

import sys
import tomllib
import tomli_w
from pathlib import Path


def update_version(new_version: str) -> None:
    """Update the version in all package pyproject.toml files.

    Args:
        new_version: The new version string to set
    """
    packages_dir = Path("packages")
    updated_files = []

    if not packages_dir.exists():
        print("Error: packages/ directory not found")
        sys.exit(1)

    for package_dir in packages_dir.iterdir():
        if package_dir.is_dir():
            pyproject_path = package_dir / "pyproject.toml"

            if pyproject_path.exists():
                try:
                    # Read current config
                    with open(pyproject_path, "rb") as f:
                        data = tomllib.load(f)

                    # Update version if project section exists
                    if "project" in data and "version" in data["project"]:
                        data["project"]["version"] = new_version

                        # Write back
                        with open(pyproject_path, "wb") as f:
                            tomli_w.dump(data, f)

                        updated_files.append(str(pyproject_path))
                        print(f"Updated {pyproject_path} to version {new_version}")
                    else:
                        print(f"Skipped {pyproject_path} (no project.version found)")

                except Exception as e:
                    print(f"Error updating {pyproject_path}: {e}")
            else:
                print(f"Skipped {package_dir.name} (no pyproject.toml found)")

    if not updated_files:
        print("No files were updated")
        sys.exit(1)

    print(f"\nSuccessfully updated {len(updated_files)} files")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_version.py VERSION")
        print("Updates version in all packages/*/pyproject.toml files")
        sys.exit(1)

    update_version(sys.argv[1])
