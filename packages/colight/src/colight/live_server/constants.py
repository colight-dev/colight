"""Constants for colight.live_server package."""

# Default ignore patterns for file discovery
DEFAULT_IGNORE_PATTERNS = [
    ".*",  # Hidden files/dirs
    "__pycache__",
    "*.pyc",
    "__init__.py",
    "node_modules",
    ".git",
    ".venv",
    "venv",
    ".env",
    "env",
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".coverage",
    "htmlcov",
    ".tox",
    "*.egg-info",
    ".idea",
    ".vscode",
    ".colight_cache",
]
