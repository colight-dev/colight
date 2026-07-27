[![PyPI version](https://badge.fury.io/py/colight.svg)](https://badge.fury.io/py/colight)

# Colight: declarative, reactive visuals in Python

---

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

## CLI

Use the CLI to turn a `.py` file into a document or run a live-updating view.

Install:

```bash
pip install colight
# or
uv tool install colight
```

Examples:

```bash
# Live incremental execution + live reload
colight live path/to/notebook.py

# Publish a file once
colight publish path/to/notebook.py --format html --output build/

# Publish + watch + serve
colight publish path/to/notebook.py --serve

# Render a .colight file to an image or video
colight render path/to/plot.colight --out plot.png
colight render path/to/plot.colight updates.colight --out plot.mp4
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
