"""Constants for colight-prose."""

# Default size threshold (in bytes) for inlining .colight files as script tags
# Files smaller than this will be embedded directly, larger files use external references
DEFAULT_INLINE_THRESHOLD = 50000

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
