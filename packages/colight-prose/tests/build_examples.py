"""Generate test artifacts for manual inspection."""

import pathlib

from colight_prose.static.builder import build_directory


def build_examples():
    """Generate various colight-prose examples in test-artifacts for manual inspection."""

    # Get paths
    test_dir = pathlib.Path(__file__).parent
    examples_dir = test_dir / "examples"
    project_root = test_dir.parent.parent.parent  # Go up 3 levels to project root
    artifacts_dir = project_root / "test-artifacts" / "colight-prose"

    # Create test-artifacts directory
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # # Clean up any existing files
    # for file in artifacts_dir.rglob("*"):
    #     if file.is_file():
    #         file.unlink()
    # for dir in artifacts_dir.rglob("*"):
    #     if dir.is_dir() and dir != artifacts_dir:
    #         try:
    #             dir.rmdir()
    #         except OSError:
    #             pass

    # Use the existing build_directory function to generate both markdown and HTML
    # Don't bypass PEP 723 - let it run properly with uv
    print("Building markdown files...")
    build_directory(examples_dir, artifacts_dir, verbose=True, formats={"markdown"})

    print("Building HTML files...")
    build_directory(examples_dir, artifacts_dir, verbose=True, formats={"html"})

    print(f"\nGenerated test artifacts in: {artifacts_dir}")
    print("Files created:")
    for file in sorted(artifacts_dir.rglob("*")):
        if file.is_file():
            print(f"  {file.relative_to(artifacts_dir)}")


if __name__ == "__main__":
    build_examples()
