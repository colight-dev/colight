# %% [markdown]
# # CLI Tools
#
# The `colight` package installs a `colight` command-line tool for working with
# notebook-style `.py` documents and `.colight` files:
#
# ```bash
# colight live <SOURCE>       # live-updating browser view with incremental execution
# colight publish <SOURCE>    # export to Markdown/HTML
# colight render <FILES>      # render .colight files to images or video
# colight view <FILE>         # open a .colight file in the browser
# colight eval                # eval server for the VS Code extension
# ```
#
# Install with `pip install colight` or `uv tool install colight`.
#
# %% [markdown]
# ## `colight live`
#
# Runs a live server over a file or directory. Files are parsed into blocks,
# executed incrementally (only changed blocks re-run), and the rendered document
# updates in the browser as you edit.
#
# ```bash
# colight live path/to/notebook.py
# colight live docs/ --port 5500
# ```
#
# Options: `--include`/`--ignore` glob filters, `--host`, `--port` (HTTP; the
# WebSocket uses port+1), `--pragma` to set default pragma tags, `--no-open` to
# skip opening a browser, `--verbose`.
#
# %% [markdown]
# ## `colight publish`
#
# Exports a `.py` file or directory tree to Markdown and/or HTML, with visuals
# saved as `.colight` files (or inlined when small).
#
# ```bash
# colight publish notebook.py                     # Markdown, next to the source
# colight publish notebook.py -f html -o build/   # standalone HTML
# colight publish docs/ -f markdown,html -o build/
# colight publish notebook.py --watch             # rebuild on change
# colight publish notebook.py --serve             # watch + dev server + live reload
# ```
#
# - `--format`/`-f`: comma-separated output formats, `markdown` and/or `html`
#   (default: `markdown`; `--serve` always adds `html`).
# - `--output`/`-o`: output file or directory (defaults: `.` for a single file,
#   `build/` for a directory, `.colight_cache/` when serving).
# - `--watch`: rebuild on change; `--serve` implies `--watch` and serves the
#   output with live reload (`--host`, `--port`, `--no-open` apply).
# - `--include`/`--ignore`: glob filters for selecting files from a tree
#   (default include: `*.py`).
# - `--pragma`: comma-separated default pragma tags (e.g. `hide-statements`).
# - `--colight-output-path`, `--colight-embed-path`, `--inline-threshold`:
#   control where `.colight` artifacts are written, how they are referenced from
#   the HTML, and below what size they are inlined into the page.
# - `--continue-on-error`: keep building when a block fails (default: true).
# - `--verbose`/`-v`: print progress information.
#
# %% [markdown]
# ## `colight render`
#
# Renders `.colight` files to static images or video using headless Chrome.
#
# ```bash
# colight render plot.colight --out plot.png
# colight render plot.colight --out plot.pdf
# colight render plot.colight updates.colight --out animation.mp4
# ```
#
# Output format follows the `--out` extension: `.png`, `.webp`, `.pdf`, `.gif`,
# or `.mp4`. The first input provides the initial state; subsequent inputs (or
# update entries appended to the same file) provide state updates, which become
# video frames. A file whose metadata includes an animated slider (`animateBy`)
# can be rendered to video directly without explicit updates.
#
# Options: `--fps`, `--width`, `--height`, `--scale`, `--quality` (WebP),
# `--frame N` to apply updates up to index N before rendering a still, `--last`
# to apply all updates, `--ready-timeout`, `--debug`.
#
# %% [markdown]
# ## `colight view`
#
# Opens a `.colight` file in the browser by generating a self-contained HTML
# page (the viewer script and data are inlined).
#
# ```bash
# colight view plot.colight
# colight view plot.colight -o plot.html --no-open
# ```
#
# %% [markdown]
# ## `colight eval`
#
# Starts an eval server that accepts code snippets over WebSocket and returns
# execution results with visuals. It exists to power the Colight VS Code
# extension; you normally don't run it by hand. Options: `--host`, `--port`
# (default 5510; WebSocket on port+1), `--verbose`.
#
# %% [markdown]
# ## Document file format
#
# Doc sources are executable Python files where comments become Markdown prose
# and expressions emit Colight visuals. Blocks are separated by blank lines — no
# cell markers are required:
#
# ```python
# # # My Page
# # A line of markdown lives inside a Python comment.
#
# import colight.plot as Plot
#
# my_data = [[0, 1], [1, 2], [2, 3]]
# Plot.line(my_data)
# ```
#
# A file may declare its dependencies inline using PEP 723 script metadata;
# `colight publish` will execute it in a matching environment.
#
# %% [markdown]
# ## Pragma directives
#
# Pragmas control what gets rendered without editing CLI arguments. A pragma is
# a comment starting with `# |` or `# %%`:
#
# ```python
# # | hide-code
# # %% hide-visuals
# ```
#
# ### Tags
#
# Each of the four element kinds has `hide-*` and `show-*` forms:
#
# - `hide-code` / `show-code`
# - `hide-statements` / `show-statements`
# - `hide-visuals` / `show-visuals`
# - `hide-prose` / `show-prose`
#
# Placed directly before a block, a pragma applies to that block. At the top of
# a file (before any code), the `hide-all-*` variants set file-level defaults,
# e.g. `# | hide-all-statements`.
#
# `# | pragma: always-eval` marks a block to re-run on every pass, bypassing the
# incremental-execution cache.
#
# ### Precedence
#
# Block-level pragmas > CLI `--pragma` options > file-level `hide-all-*`
# defaults. An explicit `show-*` always overrides a matching `hide-*`.
