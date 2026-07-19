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

# Dump a file's block graph (stable ids, line ranges, dependencies)
colight blocks path/to/notebook.py --json

# Headless evaluation; consecutive runs report a per-block diff
# (cached | ran:unchanged | ran:changed | new | removed | error).
# --force re-executes everything; `# | pragma: always-eval` marks a
# block as never-cached
colight run path/to/notebook.py --json
colight run path/to/notebook.py --force

# Inspect a visual's structure without rendering: components, array
# schemas, state keys, and sanity warnings (NaN/Inf, empty arrays, ...)
colight inspect path/to/plot.colight --json

# Semantic diff of two visuals (.colight or .py): components added/removed,
# per-array magnitude stats (max/mean |delta|, bounds drift), state keys,
# warning delta. Exit 0 = identical within --epsilon, 1 = differences
colight diff before.colight after.colight --json

# Deterministic screenshot: fixed viewport + device-pixel-ratio, waits for
# render completion; --check renders twice and byte-compares.
# Scene3d targets also report per-component pixel `coverage` in --json and
# accept --frame "C[:A-B]" to fit the camera on a component/instance range
colight screenshot path/to/plot.colight --out shot.png --json
colight screenshot notebook.py --block ID --out shot.png
colight screenshot scene.py --out closeup.png --frame "Ellipsoid:0-4"

# Machine-legible screenshots (composed post-capture; render untouched):
# --rulers adds big labeled coordinate rulers in the exact page-pixel
# space pick-at consumes (read a coordinate off the ruler, pass it to
# pick-at); --views composes a labeled contact sheet of camera presets
# (one 2x2 grid costs an agent fewer image tiles than four images);
# --max-edge N sizes the render so the PNG long edge is exactly N,
# avoiding a second lossy resampling for agents with a known input size
colight screenshot scene.py --out rulers.png --rulers
colight screenshot scene.py --out sheet.png --views front,top,side,iso
colight screenshot scene.py --out native.png --max-edge 1092

# Scene3d pick queries (GPU pick buffer; every query re-renders):
# pick-at = what is at point X,Y (ranked hits + dereferenced values);
# pick-where = where does a selection land (bbox/centroid/visibility;
# --out writes a highlight overlay). Exit 1 = no hit / invisible
colight pick-at scene.py 240,180 --json
colight pick-where scene.py --component Ellipsoid --instances 0-4 --out overlay.png

# Golden verification: pin each visual's artifact, structure hash and
# screenshot sha under tests/goldens/; verify reports which layer changed
# (structure vs pixels) with a semantic diff summary.
# Exit 0 = match, 1 = mismatch, 2 = error, 3 = no goldens yet
colight verify notebook.py --json
colight verify notebook.py --update   # pin/refresh goldens (reports changes)

# Daemon: keeps headless Chrome (and recently loaded scenes) warm so tight
# loops of screenshot/pick-at/pick-where/verify skip the browser launch.
# Discovery is automatic via <project_root>/.colight_cache/daemon.json:
# render-path commands use a running daemon transparently (falling back
# silently to direct mode when none is usable) — no flags needed.
# A repeated query against an unchanged target is also served from a warm
# scene cache, skipping re-evaluation and re-loading entirely.
colight daemon start        # detached; self-stops after 30 idle minutes
colight daemon status --json
colight daemon stop
colight screenshot scene.py --out shot.png --no-daemon   # opt out per call
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
