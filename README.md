[![PyPI version](https://badge.fury.io/py/colight.svg)](https://badge.fury.io/py/colight)

# Colight: declarative, reactive visualizations in Python

-----

`colight.plot` provides a composable way to create interactive plots using [Observable Plot](https://observablehq.com/plot/).

Key features:

- Functional, composable plot creation built on Observable Plot (with near 1:1 API correspondence between Python and JavaScript)
- Support for sliders & animations
- Works in Jupyter / Google Colab
- HTML mode which persists plots across kernel restart/shutdown, and a Widget mode which supports Python<>JavaScript interactivity
- Terse layout syntax for organizing plots into rows and columns
- Hiccup implementation for interspersing arbitrary HTML
- Export visualizations as HTML or as `.colight` files for embedding in websites

For detailed usage instructions and examples, refer to the [Colight User Guide](https://colight.dev).

## Embedding Visualizations

Colight supports exporting visualizations for embedding in websites:

### HTML Export

```python
from colight import plot
from colight.html import html_page

# Create a visualization
p = plot.plot(...)

# Save as standalone HTML
with open("my_visualization.html", "w") as f:
    f.write(html_page(p))
```

### Data Export (.colight)

For more efficient embedding with binary data support:

```python
from colight import plot
from colight.html import export_colight

# Create a visualization
p = plot.plot(...)

# Export as .colight file with an example HTML file
colight_path, example_path = export_colight(
    p,
    "my_visualization.colight",
    create_example=True,  # Creates an example HTML file (default: True)
    use_local_embed=True  # Uses local embed.js for testing (default: False)
)
print(f"Colight file: {colight_path}")
print(f"Example HTML: {example_path}")

# Or create just the .colight file without an example
colight_path = export_colight(p, "my_visualization.colight", create_example=False)

# You can also create an example HTML file separately
from notebooks.embed_examples import create_embed_example
example_path = create_embed_example(
    "my_visualization.colight",  # Path to the .colight file
    use_local_embed=True  # Uses local embed.js for testing
)

# Note: When using use_local_embed=True, a specialized viewer is created
# that works directly with the file:// protocol (no server needed)
```

To embed a `.colight` file in your website:

```html
<!-- Import the Colight embed script -->
<script type="module" src="https://cdn.jsdelivr.net/npm/@colight/core/embed.js"></script>

<!-- Option 1: Simple embedding with data-src attribute -->
<div class="colight-embed" data-src="./my_visualization.colight"></div>

<!-- Option 2: Programmatic embedding -->
<script type="module">
  import { loadVisual } from "https://cdn.jsdelivr.net/npm/@colight/core/embed.js";
  loadVisual("#my-container", "./my_visualization.colight");
</script>
<div id="my-container"></div>
```

### SPA Support

For single-page applications where content is loaded dynamically:

```javascript
import { loadVisuals } from "https://cdn.jsdelivr.net/npm/@colight/core/embed.js";

// After loading new content:
loadVisuals(); // Scan the entire document
// or
loadVisuals({root: "#my-newly-loaded-content"}); // Scan a specific container

// For complete control, you can disable the automatic MutationObserver:
import { initialize } from "https://cdn.jsdelivr.net/npm/@colight/core/embed.js";

// Disable automatic detection for SPAs
initialize({ observeMutations: false });

// Then manually call loadVisuals() when needed
```

## Development

Run `yarn watch` to compile the JavaScript bundle.

### CI Workflows

The project has several CI workflows:
- **Tests**: Runs JavaScript and Python unit tests
- **WebGPU Screenshots**: Tests 3D WebGPU rendering capabilities by capturing screenshots in headless Chrome
- **Docs**: Builds and deploys documentation
- **Pyright**: Runs type checking for Python code
- **Ruff**: Runs code formatting and linting

## Credits

- [AnyWidget](https://github.com/manzt/anywidget) provides a nice Python<>JavaScript widget API
- [pyobsplot](https://github.com/juba/pyobsplot) was the inspiration for our Python->JavaScript approach
