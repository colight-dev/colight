"""Static site generator for Colight visualizations."""

__version__ = "2025.4.1"

# Re-export the public API for easy access
from .api import (
    EvaluatedBlock,
    EvaluatedPython,
    evaluate_python,
)

__all__ = [
    "__version__",
    # Core API
    "evaluate_python",
    "EvaluatedBlock",
    "EvaluatedPython",
]
