# Mixed Content Example
# This example shows various types of content and edge cases.

import numpy as np

# Start with some narrative
# This is just explanatory text that becomes markdown.
#
# We can have multiple paragraphs.

# Simple computation
result = 2 + 2
print(f"2 + 2 = {result}")

# Some data that will be visualized
data = [1, 4, 9, 16, 25, 36]

# Visualize the data
data

# More narrative
# Here we continue with more explanation.
# This demonstrates how narrative and code are interwoven.

# Non-visualizable return values
"This is just a string"

42

True

None

# These won't create .colight files, just return values

# Arrays that will be visualized
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([1, 4, 9, 16, 25])

# Multiple arrays
arr1, arr2

# Dictionary with mixed data
mixed_data = {
    "numbers": [1, 2, 3, 4, 5],
    "squares": [1, 4, 9, 16, 25],
    "description": "Numbers and their squares",
}

# Visualize dictionary
mixed_data

# Final section
# This concludes our mixed content example.
# Notice how both text and code flow together naturally.
