#!/usr/bin/env python3
"""Script to initialize a new package in the Colight monorepo."""

import argparse
import sys
from pathlib import Path


def create_package(
    name: str,
    description: str,
    author_name: str = "Matthew Huebert",
    author_email: str = "mhuebert@gmail.com",
    python_version: str = ">=3.10",
) -> None:
    """Create a new package in the monorepo.

    Args:
        name: Package name (e.g., 'colight-example')
        description: Package description
        author_name: Author name
        author_email: Author email
        python_version: Python version requirement
    """
    # Convert name to valid Python module name
    module_name = name.replace("-", "_")

    # Create package directory structure
    package_dir = Path(f"packages/{name}")
    src_dir = package_dir / "src" / module_name
    tests_dir = package_dir / "tests"

    # Create directories
    src_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    # Create pyproject.toml
    pyproject_content = f"""[build-system]
requires = [
    "hatchling",
]
build-backend = "hatchling.build"

[project]
name = "{name}"
version = "2025.4.1"
description = "{description}"
authors = [
    {{ name = "{author_name}", email = "{author_email}" }},
]
readme = "README.md"
requires-python = "{python_version}"
dependencies = [
    # Add your dependencies here
]

[project.license]
text = "MIT"

[dependency-groups]
dev = [
    "pytest>=8.4.0",
]

[tool.hatch.build.targets.wheel]
packages = [
    "src/{module_name}",
]

[tool.pyright]
venvPath = "../.."
venv = ".venv"
include = [
    "src",
    "tests",
]
typeCheckingMode = "standard"
reportUnusedExpression = false
reportFunctionMemberAccess = false

[tool.ruff.lint]
ignore = [
    "E402",
]"""

    (package_dir / "pyproject.toml").write_text(pyproject_content)

    # Create README.md
    readme_content = f"""# {name}

{description}

## Installation

```bash
pip install {name}
```

## Development

This package is part of the Colight monorepo. To develop locally:

```bash
uv sync
```

## Testing

```bash
uv run pytest
```"""

    (package_dir / "README.md").write_text(readme_content)

    # Create __init__.py
    init_content = f'''"""{description}"""

__version__ = "2025.4.1"
__all__ = []'''

    (src_dir / "__init__.py").write_text(init_content)

    # Create test file
    test_content = f'''"""Tests for {name}."""

def test_import():
    """Test that the package can be imported."""
    import {module_name}
    assert hasattr({module_name}, "__version__")'''

    (tests_dir / f"test_{module_name}.py").write_text(test_content)

    print(f"✅ Created package '{name}' at packages/{name}")
    print(f"📦 Module name: {module_name}")
    print("\nNext steps:")
    print(
        "1. Update the root pyproject.toml to add this package to [tool.uv.sources] if needed"
    )
    print("2. Run 'uv sync' to install the package")
    print("3. Add your code to src/{module_name}/")


def main():
    parser = argparse.ArgumentParser(
        description="Create a new package in the Colight monorepo"
    )
    parser.add_argument("name", help="Package name (e.g., 'colight-example')")
    parser.add_argument("description", help="Package description")
    parser.add_argument(
        "--author-name",
        default="Matthew Huebert",
        help="Author name (default: Matthew Huebert)",
    )
    parser.add_argument(
        "--author-email",
        default="mhuebert@gmail.com",
        help="Author email (default: mhuebert@gmail.com)",
    )
    parser.add_argument(
        "--python-version",
        default=">=3.10",
        help="Python version requirement (default: >=3.10)",
    )

    args = parser.parse_args()

    # Check if we're in the right directory
    if not Path("packages").exists():
        print("❌ Error: 'packages' directory not found.")
        print("Please run this script from the root of the monorepo.")
        sys.exit(1)

    # Check if package already exists
    if Path(f"packages/{args.name}").exists():
        print(f"❌ Error: Package '{args.name}' already exists.")
        sys.exit(1)

    create_package(
        args.name,
        args.description,
        args.author_name,
        args.author_email,
        args.python_version,
    )


if __name__ == "__main__":
    main()
