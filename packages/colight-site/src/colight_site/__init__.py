"""Static site generator for Colight visualizations."""

__version__ = "2025.4.1"

# Re-export the public API for easy access
from .api import (
    process_colight_file,
    is_colight_file,
    ProcessedForm,
    ProcessingResult,
    build_file,
    build_directory,
    init_project,
    get_output_path,
)

__all__ = [
    "__version__",
    # Core API
    "process_colight_file",
    "is_colight_file",
    "ProcessedForm",
    "ProcessingResult",
    # CLI convenience functions
    "build_file",
    "build_directory",
    "init_project",
    "get_output_path",
]
