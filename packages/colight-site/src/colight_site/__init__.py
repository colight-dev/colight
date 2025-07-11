"""Static site generator for Colight visualizations."""

__version__ = "2025.4.1"

# Re-export the public API for easy access
from .api import (
    EvaluatedBlock,
    EvaluatedPython,
    build_directory,
    build_file,
    evaluate_python,
    get_output_path,
    init_project,
)

__all__ = [
    "__version__",
    # Core API
    "evaluate_python",
    "EvaluatedBlock",
    "EvaluatedPython",
    # CLI convenience functions
    "build_file",
    "build_directory",
    "init_project",
    "get_output_path",
]
