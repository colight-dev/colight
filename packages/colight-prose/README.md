# Colight Site

Static site generator for Colight visualizations.

Converts `.py` files into markdown/HTML documents where:

- Comments become narrative markdown
- Code blocks are executed to generate Colight visualizations
- Output is embedded as interactive `.colight` files

## Usage

```bash
# Build a single file
colight-prose build src/post.py --output build/post.md

# Watch for changes
colight-prose watch src/ --output build/

# Initialize new project
colight-prose init my-blog/
```

## File Format

`.py` files mix comments (markdown) with executable Python code:

```python
# My Data Visualization
# This creates an interactive plot...

import numpy as np
x = np.linspace(0, 10, 100)

# The sine wave
np.sin(x)  # This expression generates a colight visualization
```

## Pragma Directives

Control the output format and visibility of content using pragma comments:

### File-level pragmas

Begin with `hide-all-`, at the beginning of the file:

```python
# | hide-all-code

# Your content here...
```

### Form-level pragmas

Place directly before a code block:

```python
# | show-code
x = np.array([1, 2, 3])
```

### Available pragmas

- `hide-code` / `show-code` - Hide or show code blocks
- `hide-statements` / `show-statements` - Hide or show Python statements (imports, assignments)
- `hide-visuals` / `show-visuals` - Hide or show results/visuals
- `hide-prose` / `show-prose` - Hide or show markdown prose (comments)
- `format-html` - Output in HTML format
- `format-markdown` - Output in Markdown format

### Pragma formats

```python
# %% hide-code          # Double percent format
# | show-visuals        # Pipe format
#| colight: hide-prose  # Legacy format (still supported)
```

### Precedence

1. Form-level pragmas (highest priority)
2. CLI options
3. File-level pragmas
4. Defaults (lowest priority)

`show-*` pragmas always override corresponding `hide-*` pragmas.
