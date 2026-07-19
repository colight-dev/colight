[![PyPI version](https://badge.fury.io/py/colight.svg)](https://badge.fury.io/py/colight)

# Colight: declarative, reactive visuals in Python

Colight is a Python library and toolchain for interactive, JavaScript-based visuals. It has three main parts:

- **`colight.plot`** — composable interactive plots built on [Observable Plot](https://observablehq.com/plot/), with near 1:1 API correspondence between Python and JavaScript.
- **`colight.scene3d`** — declarative 3D scenes rendered with WebGPU: point clouds, ellipsoids, cuboids, meshes, image planes, line beams/segments, camera frustums, hierarchical groups, picking, and drag interactions.
- **The `colight` CLI** — turns notebook-style `.py` files (comments are markdown, expressions render as visuals) into live-updating documents or published Markdown/HTML, and renders `.colight` files to images or video.

Key features:

- Functional, composable plot creation with support for sliders & animations
- Works in Jupyter / Google Colab; HTML mode persists plots across kernel restarts, Widget mode supports Python<>JavaScript interactivity
- Shared reactive state (`$state`) across Python and JavaScript
- Terse layout syntax for organizing visuals into rows and columns, plus a hiccup implementation for interspersing arbitrary HTML
- Export visualizations as standalone HTML or as `.colight` files for embedding in websites

Install:

```bash
pip install colight
# or
uv tool install colight   # for the CLI
```

For detailed usage instructions and examples, refer to the [Colight User Guide](https://colight.dev).

## CLI

Turn a `.py` file into a document or run a live-updating view:

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

# Open a .colight file in the browser
colight view path/to/plot.colight
```

There is also `colight eval`, an eval server used by the VS Code extension.

## Packages

Colight is a monorepo:

- **[`packages/colight`](packages/colight/)** — the `colight` Python package: Plot, Scene3D, and the document engine (`colight.runtime`, `colight.publish`, `colight.live_server`) behind the CLI.
- **[`packages/colight-scene3d`](packages/colight-scene3d/)** — publishes the Scene3D renderer as the standalone, framework-agnostic npm package [`@colight/scene3d`](https://www.npmjs.com/package/@colight/scene3d), usable without the rest of Colight (`npm install @colight/scene3d`).
- **[`packages/colight-mkdocs`](packages/colight-mkdocs/)** — MkDocs plugins for building docs sites from `.py` sources (used by the Colight docs themselves).
- **[`packages/colight-vscode`](packages/colight-vscode/)** — a VS Code extension for evaluating Python "cells" in plain `.py` files with Colight visuals.

## Development

- `yarn build` compiles the JavaScript bundles; `yarn dev` runs esbuild in watch mode.
- `yarn test` runs the Python and JS test suites.
- `yarn docs:watch` / `yarn docs:build` serve and build the docs site.

See [DEVELOPING.md](DEVELOPING.md) for more.

### CI Workflows

- **Tests**: Runs JavaScript and Python unit tests
- **WebGPU Screenshots**: Tests 3D WebGPU rendering by capturing screenshots in headless Chrome
- **Docs**: Builds and deploys documentation
- **Pyright**: Runs type checking for Python code
- **Ruff**: Runs code formatting and linting

Releases are tag-driven per package: calver tags (`v2025.x.x`) publish `colight` to PyPI, `scene3d-v*` tags publish `@colight/scene3d` to npm (see `scripts/release.py` and `.github/workflows/release.yml`).

## Credits

- [AnyWidget](https://github.com/manzt/anywidget) provides a nice Python<>JavaScript widget API
- [pyobsplot](https://github.com/juba/pyobsplot) was the inspiration for our Python->JavaScript approach
