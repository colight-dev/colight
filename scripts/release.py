#!/usr/bin/env python3
"""Interactive release script for colight packages.

Usage:
    uv run python scripts/release.py
    yarn release

Requires:
    uv add --dev questionary
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import questionary
import tomllib
import tomli_w
from questionary import Style

# Custom style
style = Style([
    ("qmark", "fg:cyan bold"),
    ("question", "bold"),
    ("answer", "fg:cyan"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
    ("selected", "fg:green"),
])

# Package definitions
# type: "python" = pyproject.toml only, "npm" = package.json only, "both" = both
PACKAGES = {
    "colight": {
        "type": "python",
        "path": "packages/colight",
        "pyproject": "packages/colight/pyproject.toml",
        "versioning": "calver",
        "description": "Python",
    },
    "scene3d": {
        "type": "npm",
        "path": "packages/colight-scene3d",
        "package_json": "packages/colight-scene3d/package.json",
        "npm_name": "@colight/scene3d",
        "versioning": "semver",
        "description": "npm",
    },
    "serde": {
        "type": "both",
        "path": "packages/colight-serde",
        "pyproject": "packages/colight-serde/pyproject.toml",
        "package_json": "packages/colight-serde/package.json",
        "npm_name": "@colight/serde",
        "pypi_name": "colight-serde",
        "versioning": "semver",
        "description": "Python + npm",
    },
}


def check_working_directory():
    """Ensure no unstaged changes (staged changes are allowed and will be included)."""
    result = subprocess.run(
        ["git", "diff", "--name-only"],
        capture_output=True,
        text=True,
    )
    changed_files = [
        f for f in result.stdout.strip().split("\n")
        if f and f != "scripts/release.py"
    ]
    if changed_files:
        questionary.print("Error: There are unstaged changes to tracked files:", style="fg:red bold")
        for f in changed_files:
            questionary.print(f"  {f}", style="fg:red")
        questionary.print("\nStage the changes you want to include, or stash them.", style="fg:yellow")
        sys.exit(1)


def get_current_version(package_key: str) -> str:
    """Get current version from package file."""
    pkg = PACKAGES[package_key]

    # Prefer pyproject.toml if it exists, else package.json
    if "pyproject" in pkg:
        path = Path(pkg["pyproject"])
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return data["project"]["version"]
    else:
        path = Path(pkg["package_json"])
        with open(path) as f:
            data = json.load(f)
        return data["version"]


def get_next_calver() -> str:
    """Get next calver version for colight."""
    today = datetime.now()
    year = today.year
    month = today.month
    year_month = f"{year}.{month}"

    padded_month = f"{month:02d}"
    tags = (
        subprocess.check_output(
            ["git", "tag", "-l", f"v{year}.{month}.*", f"v{year}.{padded_month}.*"]
        )
        .decode()
        .strip()
        .split("\n")
    )

    release_tags = [tag[1:] for tag in tags if tag and not tag.endswith(".dev")]
    regular_versions = [tag for tag in release_tags if "alpha" not in tag]

    if not regular_versions:
        return f"{year_month}.1"
    else:
        patch_numbers = [int(tag.split(".")[-1]) for tag in regular_versions]
        next_patch = max(patch_numbers) + 1
        return f"{year_month}.{next_patch}"


def bump_semver(current: str, bump_type: str) -> str:
    """Bump semver version."""
    parts = current.split(".")
    if len(parts) != 3:
        questionary.print(f"Error: Invalid version format '{current}', expected x.y.z", style="fg:red")
        sys.exit(1)

    major, minor, patch = map(int, parts)

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def select_packages() -> list[str]:
    """Interactive package selection with checkboxes."""
    choices = []
    for key in PACKAGES:
        pkg = PACKAGES[key]
        version = get_current_version(key)
        name = pkg.get("npm_name", key)
        choices.append(questionary.Choice(
            title=f"{name} ({pkg['description']}) - {version}",
            value=key,
        ))

    selected = questionary.checkbox(
        "Select packages to release:",
        choices=choices,
        style=style,
    ).ask()

    if not selected:
        questionary.print("No packages selected.", style="fg:yellow")
        sys.exit(0)

    return selected


def get_bump_type(package_key: str) -> str:
    """Ask for semver bump type."""
    current = get_current_version(package_key)
    pkg = PACKAGES[package_key]
    name = pkg.get("npm_name", package_key)

    bump_type = questionary.select(
        f"{name} ({current}) - bump type:",
        choices=[
            questionary.Choice("patch", value="patch"),
            questionary.Choice("minor", value="minor"),
            questionary.Choice("major", value="major"),
        ],
        style=style,
    ).ask()

    if not bump_type:
        sys.exit(0)

    return bump_type


def update_version(package_key: str, new_version: str) -> list[Path]:
    """Update version in package file(s). Returns list of updated paths."""
    pkg = PACKAGES[package_key]
    updated_paths = []

    # Update pyproject.toml if present
    if "pyproject" in pkg:
        path = Path(pkg["pyproject"])
        with open(path, "rb") as f:
            data = tomllib.load(f)
        data["project"]["version"] = new_version
        with open(path, "wb") as f:
            tomli_w.dump(data, f)
        updated_paths.append(path)

    # Update package.json if present
    if "package_json" in pkg:
        path = Path(pkg["package_json"])
        with open(path) as f:
            data = json.load(f)
        data["version"] = new_version
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        updated_paths.append(path)

    return updated_paths


def update_changelog(new_version: str) -> bool:
    """Generate changelog entry for colight release."""
    try:
        last_tag = (
            subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"])
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        questionary.print("No previous tags found, skipping changelog generation", style="fg:yellow")
        return True

    commit_messages = (
        subprocess.check_output(
            ["git", "log", f"{last_tag}..HEAD", "--pretty=format:%B"]
        )
        .decode()
        .split("\n\n")
    )

    categories = {
        "New Features": "feat:",
        "Bug Fixes": "fix:",
        "Documentation": "docs:",
        "Other Changes": None,
    }

    categorized_commits = {category: [] for category in categories}

    for msg in commit_messages:
        lines = msg.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            categorized = False
            for category, prefix in categories.items():
                if prefix and line.lower().startswith(prefix.lower()):
                    cleaned_msg = line[len(prefix):].strip().lstrip("- •").strip()
                    categorized_commits[category].append(cleaned_msg)
                    categorized = True
                    break
            if not categorized:
                cleaned_line = line.lstrip("- •").strip()
                categorized_commits["Other Changes"].append(cleaned_line)

    changelog_entry = (
        f"### [{new_version}] - {datetime.now().strftime('%b %d, %Y')}\n\n"
    )

    for category, commits in categorized_commits.items():
        if commits:
            changelog_entry += f"#### {category}\n"
            changelog_entry += "\n".join(f"- {commit}" for commit in commits)
            changelog_entry += "\n\n"

    if (
        len([c for c in categorized_commits.values() if c]) == 1
        and categorized_commits["Other Changes"]
    ):
        changelog_entry = changelog_entry.replace("#### Other Changes\n", "")

    changelog_path = Path("CHANGELOG.md")
    if changelog_path.exists():
        with open(changelog_path) as f:
            original_content = f.read()
    else:
        original_content = ""

    with open(changelog_path, "w") as f:
        f.write(changelog_entry + original_content)

    questionary.print("\nNew changelog entry:", style="fg:cyan bold")
    print(changelog_entry)

    if not questionary.confirm("Accept changelog entry?", default=True, style=style).ask():
        with open(changelog_path, "w") as f:
            f.write(original_content)
        questionary.print("Changelog update cancelled.", style="fg:yellow")
        return False

    return True


def has_npm(pkg: dict) -> bool:
    """Check if package publishes to npm."""
    return pkg["type"] in ("npm", "both")


def has_pypi(pkg: dict) -> bool:
    """Check if package publishes to PyPI."""
    return pkg["type"] in ("python", "both")


def main():
    questionary.print("\n  colight release\n", style="fg:cyan bold")

    check_working_directory()

    selected = select_packages()

    # Determine versions for each selected package
    releases = []
    any_npm = False
    releases_colight = False

    for key in selected:
        pkg = PACKAGES[key]
        current = get_current_version(key)

        if pkg["versioning"] == "calver":
            new_version = get_next_calver()
        else:
            bump_type = get_bump_type(key)
            new_version = bump_semver(current, bump_type)

        if has_npm(pkg):
            any_npm = True
        if key == "colight":
            releases_colight = True

        releases.append({
            "key": key,
            "pkg": pkg,
            "current": current,
            "new_version": new_version,
        })

    # Summary
    questionary.print("\nRelease summary:", style="fg:cyan bold")
    for rel in releases:
        name = rel["pkg"].get("npm_name") or rel["pkg"].get("pypi_name") or rel["key"]
        targets = []
        if has_pypi(rel["pkg"]):
            targets.append("PyPI")
        if has_npm(rel["pkg"]):
            targets.append("npm")
        questionary.print(
            f"  {name}: {rel['current']} → {rel['new_version']} ({', '.join(targets)})",
            style="fg:green"
        )
    print()

    if not questionary.confirm("Proceed with release?", default=True, style=style).ask():
        questionary.print("Aborted.", style="fg:yellow")
        sys.exit(0)

    # Update changelog for colight releases
    if releases_colight:
        colight_version = next(r["new_version"] for r in releases if r["key"] == "colight")
        if not update_changelog(colight_version):
            questionary.print("Release cancelled.", style="fg:red")
            sys.exit(1)

    # Update all version files
    files_to_add = []
    for rel in releases:
        paths = update_version(rel["key"], rel["new_version"])
        for path in paths:
            files_to_add.append(str(path))
            questionary.print(f"  Updated {path}", style="fg:green")

    # Add changelog and uv.lock if colight included
    if releases_colight:
        files_to_add.extend(["CHANGELOG.md", "uv.lock"])

    # Build if npm packages included
    if any_npm:
        questionary.print("\nBuilding...", style="fg:cyan")
        result = subprocess.run(["yarn", "build"])
        if result.returncode != 0:
            questionary.print("Build failed!", style="fg:red bold")
            sys.exit(1)

    # Git operations
    subprocess.run(["git", "add"] + files_to_add)
    subprocess.run(["pre-commit", "run", "--all-files"])
    subprocess.run(["git", "add"] + files_to_add)

    # Commit message
    if len(releases) == 1:
        rel = releases[0]
        commit_msg = f"release({rel['key']}): {rel['new_version']}"
    else:
        parts = [f"{r['key']}@{r['new_version']}" for r in releases]
        commit_msg = f"release: {', '.join(parts)}"

    subprocess.run(["git", "commit", "-m", commit_msg])

    # Create tags
    tags = []
    for rel in releases:
        # colight uses v{version}, others use {key}-v{version}
        if rel["key"] == "colight":
            tag = f"v{rel['new_version']}"
        else:
            tag = f"{rel['key']}-v{rel['new_version']}"
        tags.append(tag)
        name = rel["pkg"].get("npm_name") or rel["pkg"].get("pypi_name") or rel["key"]
        subprocess.run(["git", "tag", "-a", tag, "-m", f"Release {name} {rel['new_version']}"])

    questionary.print(f"\nCreated tags: {', '.join(tags)}", style="fg:green")

    # Push
    if questionary.confirm("Push to origin? (triggers CI publish)", default=True, style=style).ask():
        subprocess.run(["git", "push", "origin", "HEAD", "--tags"])
        questionary.print("\nCI will handle publishing to PyPI and npm.", style="fg:yellow")
    else:
        questionary.print("\nTo publish later, run: git push origin HEAD --tags", style="fg:yellow")

    questionary.print("\n  Done!\n", style="fg:cyan bold")


if __name__ == "__main__":
    main()
