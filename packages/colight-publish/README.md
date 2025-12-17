# Colight Publish

Static site generator for Colight visualizations.

Converts `.py` files into markdown/HTML documents where:

- Comments become narrative markdown
- Code blocks are executed to generate Colight visualizations
- Output is embedded as interactive `.colight` files

## Usage

The CLI is available as `colight publish` (recommended) or `colight-publish`:

```bash
colight publish <SOURCE> --format (md|html|site) [--watch] [--output DEST]
```

### Formats

| Format | Description                         | Default output |
| ------ | ----------------------------------- | -------------- |
| `md`   | Markdown files only                 | `build/`       |
| `html` | Standalone HTML + `.colight` assets | `build/`       |
| `site` | Full interactive explorer UI        | `site-build/`  |

### Examples

```bash
# Build markdown from a single file
colight publish src/post.py --format md --output build/

# Build a static site from a directory
colight publish docs/ --format site

# Watch with live reload (HTML)
colight publish docs/ --format html --watch

# Interactive explorer with hot reload
colight publish docs/ --format site --watch
```

### Options

- `--output PATH` / `-o PATH`: Output directory (ignored for `site --watch`)
- `--watch`: Regenerate on file changes; starts dev server for `html` and `site`
- `--include PATTERN`: File patterns to include (default: `*.py`, `*.md`)
- `--ignore PATTERN`: File patterns to ignore
- `--verbose`: Print extra progress information
- `--pragma TAGS`: Comma-separated pragma overrides
- `--host` / `--port`: Dev server binding (default: `127.0.0.1:5500`)
- `--no-open`: Don't open browser when starting dev server

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
