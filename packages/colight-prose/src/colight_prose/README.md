# Colight Live

A development server that executes Python notebooks with live reloading and incremental computation.

## What it does

Colight Live watches your Python files and automatically re-executes code when you make changes. Unlike traditional approaches that re-run everything, it only executes what actually changed - making iteration fast even with expensive computations.

## How it works

Write Python files where comments become prose and code generates visualizations:

```python
# Exploring sine waves
# Let's visualize how frequency affects sine waves

import numpy as np

frequency = 2  # Try changing this value

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(frequency * x)

# The plot updates instantly when you change the frequency
{"x": x, "y": y, "title": f"sin({frequency}x)"}
```

Run the server:

```bash
colight-prose live your_file.py
```

## Key features

- **Incremental execution**: Only changed code blocks re-run
- **Dependency tracking**: Automatically determines execution order
- **Import awareness**: Updates cascade through file imports
- **Content caching**: Results cached by code content, not timestamps
- **WebSocket sync**: Browser updates without refresh

## Architecture

The system builds two dependency graphs:

1. **File graph**: Tracks imports between Python files
2. **Block graph**: Analyzes variable usage within files

When you save a file, it:

1. Identifies which blocks changed
2. Finds all blocks that depend on those changes
3. Re-executes only affected blocks in dependency order
4. Streams results to connected browsers

This design enables near-instant feedback loops while maintaining correctness.
