# Colight Site

Static site generator for Colight visualizations.

Converts `.colight.py` files into markdown/HTML documents where:

- Comments become narrative markdown
- Code blocks are executed to generate Colight visualizations
- Output is embedded as interactive `.colight` files

## Usage

```bash
# Build a single file
colight-site build src/post.colight.py --output build/post.md

# Watch for changes
colight-site watch src/ --output build/

# Initialize new project
colight-site init my-blog/
```

## File Format

`.colight.py` files mix comments (markdown) with executable Python code:

```python
# My Data Visualization
# This creates an interactive plot...

import numpy as np
x = np.linspace(0, 10, 100)

# The sine wave
np.sin(x)  # This expression generates a colight visualization
```
