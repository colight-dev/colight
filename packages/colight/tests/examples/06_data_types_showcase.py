# Data Types Showcase
# This example demonstrates visualization of various Python data types.
# Each value is displayed on its own line and will be automatically visualized.

from datetime import datetime, date, time
import numpy as np

# Basic Python Types
# ==================

# Integer
42

# Float
3.14159

# String
"Hello, World!"

# Boolean
True

# None
None

# Bytes
b"This is binary data \\x00\\x01\\x02"

# Date and Time Types
# ===================

# Datetime
datetime.now()

# Date
date.today()

# Time
time(14, 30, 45)

# Collections
# ===========

# List
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Tuple
(42.3601, -71.0589, "Boston")

# Set
{1, 2, 3, 4, 5}

# Dictionary
{
    "name": "Alice",
    "age": 30,
    "city": "New York",
    "hobbies": ["reading", "hiking", "photography"],
    "contact": {"email": "alice@example.com", "phone": "+1-555-0123"},
}

# Large Collections
# =================

# Large list
list(range(1000))

# Large dictionary
{f"key_{i}": f"value_{i}" for i in range(200)}

# NumPy Arrays
# ============

# 1D array
np.linspace(0, 10, 20)

# 2D array (matrix)
np.random.rand(5, 8)

# 3D array
np.random.rand(3, 4, 5)

# Large array
np.random.rand(10000)

# Different dtypes - integer
np.array([1, 2, 3, 4, 5], dtype=np.int32)

# Different dtypes - boolean
np.array([True, False, True, True, False])

# Nested and Complex Structures
# =============================

{
    "users": [
        {
            "id": 1,
            "name": "John Doe",
            "scores": [95, 87, 92, 88, 91],
            "metadata": {
                "joined": date(2022, 1, 15),
                "last_login": datetime.now(),
                "preferences": {"theme": "dark", "notifications": True},
            },
        },
        {
            "id": 2,
            "name": "Jane Smith",
            "scores": [88, 92, 85, 90, 94],
            "metadata": {
                "joined": date(2022, 3, 20),
                "last_login": datetime.now(),
                "preferences": {"theme": "light", "notifications": False},
            },
        },
    ],
    "statistics": {
        "total_users": 2,
        "average_score": 90.5,
        "data": np.array([[1, 2, 3], [4, 5, 6]]),
    },
}

# Custom Objects
# ==============


class DataPoint:
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label
        self.timestamp = datetime.now()
        self._internal = "hidden"

    def __str__(self):
        return f"DataPoint({self.x}, {self.y}, '{self.label}')"


DataPoint(3.14, 2.71, "important")


class Dataset:
    def __init__(self, name):
        self.name = name
        self.points = [DataPoint(i, i**2, f"point_{i}") for i in range(5)]
        self.metadata = {
            "created": datetime.now(),
            "version": "1.0",
            "description": "Sample dataset with quadratic relationship",
        }

    def __str__(self):
        return f"Dataset('{self.name}', {len(self.points)} points)"


Dataset("Quadratic Example")

# Edge Cases
# ==========

# Empty list
[]

# Empty dict
{}

# Empty set
set()

# Deeply nested structure
{"level1": {"level2": {"level3": {"level4": {"level5": "deep value"}}}}}

# Mixed types in collections
[
    42,
    "string",
    [1, 2, 3],
    {"key": "value"},
    np.array([1, 2, 3]),
    datetime.now(),
    None,
    True,
]

# Special string cases - long string
"This is a very long string. " * 20

# Multiline string
"""This is a
multiline string
with multiple
lines of text"""

# Unicode and special characters
{
    "emoji": "🎉🐍📊",
    "math": "∑∏∫∂∇",
    "languages": {"chinese": "你好", "arabic": "مرحبا", "russian": "Привет"},
}
