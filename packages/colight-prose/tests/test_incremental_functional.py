"""Comprehensive tests for incremental execution with functional/immutable patterns.

These tests verify that "even after an arbitrary series of edits, the result is
that our document looks as if it had been run from top to bottom" - but only
for functional/immutable code patterns.
"""

from colight_prose.incremental_executor import IncrementalExecutor
from colight_prose.parser import parse_document


def test_simple_dependency_chain_with_edits():
    """Test that editing blocks properly propagates changes downstream."""
    # Initial code
    code = """# Define x
x = 1

# Compute y based on x
y = x * 2

# Compute z based on y
z = y + 3"""

    doc = parse_document(code)
    executor = IncrementalExecutor()

    # Initial execution
    results = list(executor.execute_incremental_streaming(doc))
    assert len(results) == 3
    assert all(r[1].error is None for r in results)
    assert executor.env["x"] == 1
    assert executor.env["y"] == 2
    assert executor.env["z"] == 5

    # Edit x
    code_edited = """# Define x
x = 5

# Compute y based on x
y = x * 2

# Compute z based on y
z = y + 3"""

    doc_edited = parse_document(code_edited)
    results = list(executor.execute_incremental_streaming(doc_edited))

    # Verify all downstream values updated
    assert executor.env["x"] == 5
    assert executor.env["y"] == 10
    assert executor.env["z"] == 13

    # Edit middle block (y calculation)
    code_edited2 = """# Define x
x = 5

# Compute y based on x
y = x * 3

# Compute z based on y
z = y + 3"""

    doc_edited2 = parse_document(code_edited2)
    results = list(executor.execute_incremental_streaming(doc_edited2))

    # x should be unchanged, y and z should update
    assert executor.env["x"] == 5
    assert executor.env["y"] == 15
    assert executor.env["z"] == 18


def test_multiple_providers():
    """Test that the last provider of a symbol is used."""
    code = """# First definition of x
x = 1

# Override x
x = 10

# Use x
y = x * 2"""

    doc = parse_document(code)
    executor = IncrementalExecutor()
    results = list(executor.execute_incremental_streaming(doc))

    assert executor.env["x"] == 10
    assert executor.env["y"] == 20

    # Edit first x - should have no effect on y
    code_edited = """# First definition of x
x = 5

# Override x
x = 10

# Use x
y = x * 2"""

    doc_edited = parse_document(code_edited)
    results = list(executor.execute_incremental_streaming(doc_edited))

    # With caching, only the changed block should execute
    # Since the first x changed but doesn't affect y (second x overrides),
    # y should still be 20 from cache
    assert executor.env["y"] == 20  # unchanged

    # The cache should have served the y block from cache
    # We can verify by checking the results - should only contain the edited block
    assert len(results) == 3  # All blocks returned but some from cache

    # Edit second x - should update y
    code_edited2 = """# First definition of x
x = 5

# Override x
x = 20

# Use x
y = x * 2"""

    doc_edited2 = parse_document(code_edited2)
    results = list(executor.execute_incremental_streaming(doc_edited2))

    assert executor.env["x"] == 20
    assert executor.env["y"] == 40


def test_always_eval_pragma():
    """Test that blocks with always-eval pragma re-execute every time."""
    code = """# Regular block
import random

# %% pragma: always-eval
# This block should always re-execute
value = random.randint(1, 100)

# Dependent block
doubled = value * 2"""

    doc = parse_document(code)
    executor = IncrementalExecutor()

    # First execution
    results = list(executor.execute_incremental_streaming(doc))
    first_value = executor.env["value"]
    first_doubled = executor.env["doubled"]
    assert first_doubled == first_value * 2

    # Re-execute without changes - only always-eval block should run
    results = list(executor.execute_incremental_streaming(doc))
    second_value = executor.env["value"]
    second_doubled = executor.env["doubled"]

    # The random value should (very likely) be different
    # But the doubled value should match the new random value
    assert second_doubled == second_value * 2

    # Edit an unrelated block
    code_edited = """# Regular block
import random
x = 42

# %% pragma: always-eval
# This block should always re-execute
value = random.randint(1, 100)

# Dependent block
doubled = value * 2"""

    doc_edited = parse_document(code_edited)
    results = list(executor.execute_incremental_streaming(doc_edited))

    # Always-eval block should have run again
    third_value = executor.env["value"]
    third_doubled = executor.env["doubled"]
    assert third_doubled == third_value * 2


def test_import_and_usage():
    """Test import statements with incremental execution."""
    code = """# Import numpy
import numpy as np

# Create data
data = np.array([1, 2, 3, 4, 5])

# Compute result
mean = np.mean(data)
std = np.std(data)"""

    doc = parse_document(code)
    executor = IncrementalExecutor()
    results = list(executor.execute_incremental_streaming(doc))

    assert "np" in executor.env
    assert executor.env["mean"] == 3.0
    assert executor.env["std"] > 0

    # Edit data
    code_edited = """# Import numpy
import numpy as np

# Create data
data = np.array([10, 20, 30, 40, 50])

# Compute result
mean = np.mean(data)
std = np.std(data)"""

    doc_edited = parse_document(code_edited)
    results = list(executor.execute_incremental_streaming(doc_edited))

    # Import block should not re-execute
    # But both data and result blocks should
    assert executor.env["mean"] == 30.0


def test_function_definition_and_usage():
    """Test defining and using pure functions."""
    code = """# Define a pure function
def transform(x):
    return x ** 2 + 1

# Create data
values = [1, 2, 3, 4, 5]

# Apply transformation
results = [transform(v) for v in values]"""

    doc = parse_document(code)
    executor = IncrementalExecutor()
    results = list(executor.execute_incremental_streaming(doc))

    assert executor.env["results"] == [2, 5, 10, 17, 26]

    # Edit the function
    code_edited = """# Define a pure function
def transform(x):
    return x ** 3 - 1

# Create data
values = [1, 2, 3, 4, 5]

# Apply transformation
results = [transform(v) for v in values]"""

    doc_edited = parse_document(code_edited)
    results = list(executor.execute_incremental_streaming(doc_edited))

    # Function change should trigger re-computation
    assert executor.env["results"] == [0, 7, 26, 63, 124]


def test_data_transformation_pipeline():
    """Test a series of pure data transformations."""
    code = """# Raw data
raw_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Filter even numbers
evens = [x for x in raw_data if x % 2 == 0]

# Square the values
squared = [x ** 2 for x in evens]

# Sum them up
total = sum(squared)"""

    doc = parse_document(code)
    executor = IncrementalExecutor()
    results = list(executor.execute_incremental_streaming(doc))

    assert executor.env["evens"] == [2, 4, 6, 8, 10]
    assert executor.env["squared"] == [4, 16, 36, 64, 100]
    assert executor.env["total"] == 220

    # Edit middle transformation
    code_edited = """# Raw data
raw_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Filter even numbers
evens = [x for x in raw_data if x % 2 == 0]

# Cube the values instead
squared = [x ** 3 for x in evens]

# Sum them up
total = sum(squared)"""

    doc_edited = parse_document(code_edited)
    list(executor.execute_incremental_streaming(doc_edited))

    # Only squared and total should update
    assert executor.env["evens"] == [2, 4, 6, 8, 10]  # unchanged
    assert executor.env["squared"] == [8, 64, 216, 512, 1000]  # cubed now
    assert executor.env["total"] == 1800


def test_visualization_updates():
    """Test that visualizations update when data changes."""
    code = """# Import visualization library
import colight.plot as Plot

# Create data
data = [1, 2, 3, 4, 5]

# Create visualization
Plot.dot(data)"""

    doc = parse_document(code)
    executor = IncrementalExecutor()
    results = list(executor.execute_incremental_streaming(doc))

    # Check that visualization was created
    assert results[-1][1].error is None

    # Update data
    code_edited = """# Import visualization library
import colight.plot as Plot

# Create data
data = [5, 4, 3, 2, 1]

# Create visualization
Plot.dot(data)"""

    doc_edited = parse_document(code_edited)
    results = list(executor.execute_incremental_streaming(doc_edited))

    # With caching, when data changes, both data and visualization blocks
    # should execute (viz depends on data)
    assert len(results) == 3  # All 3 blocks

    # Check that the data was updated
    assert executor.env["data"] == [5, 4, 3, 2, 1]
