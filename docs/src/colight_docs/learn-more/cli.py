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
# When the file carries **update entries** (a replay/episode artifact — an
# initial state plus one appended entry per step), the viewer shows a
# **timeline scrubber** under the visual: a step slider (0..N), play/pause and
# a step count. Each position recomputes the state from the initial entry plus
# updates 0..k, so you can scrub any replay without rebuilding one by hand. The
# scrubber only appears when update entries exist; it is independent of any
# in-visual `Plot.Slider` autoplay (`animateBy`), which continues to drive its
# own state key within whichever step is applied.
#
# ```bash
# colight view plot.colight
# colight view replay.colight        # shows the timeline scrubber
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
#
# %% [markdown]
# ## Agent-facing commands
#
# These subcommands expose block structure, evaluation results, artifact
# diffs and reproducible screenshots as structured data (add `--json` to any
# of them for machine-readable output):
#
# ```bash
# # Dump the block graph: stable ids, line ranges, provides/requires,
# # upstream dependencies, pragma tags
# colight blocks notebook.py [--json]
#
# # Headless evaluation with a persistent per-file record; consecutive runs
# # report a per-block diff (cached | ran:unchanged | ran:changed | new |
# # removed | error). Exit code is nonzero if any block errored.
# # --force re-executes every block, ignoring cache keys (statuses then
# # compare against the stored fingerprints); blocks tagged
# # `# | pragma: always-eval` never report cached.
# colight run notebook.py [--json] [--block ID] [--force]
#
# # Structural inspection of a .colight artifact (or every visual a .py file
# # produces): components, per-array dtype/shape/min/max, state keys,
# # callbacks, plus sanity warnings (empty arrays, NaN/Inf, alphas ~0,
# # zero-extent bounds, mismatched per-instance lengths)
# colight inspect target.colight [--json]
# colight inspect notebook.py [--json]
#
# # Semantic diff of two targets (.colight or .py, evaluated headlessly;
# # visuals paired by position): components added/removed/type-changed,
# # per-array dtype/shape changes and magnitude stats (changed-element
# # count/fraction, max/mean |delta| beyond --epsilon, bounds drift),
# # scalar value changes, state keys, buffer deltas, warnings.
# # Integer ("categorical") arrays lead with the changed count/fraction and
# # demote |delta| (a label of 3 vs 1 is no closer than 3 vs 15).
# # Artifacts with UPDATE ENTRIES (replay/episode files) also get a per-step
# # diff: updates are aligned by index and each pair diffed, reporting the
# # first diverging update, how many differ, and a trailing length mismatch
# # ("first divergence at update 3; 42/80 updates differ").
# # Exit 0 = identical within epsilon, 1 = differences, 2 = error.
# colight diff a.colight b.colight [--json] [--epsilon 1e-9]
# colight diff runA.colight runB.colight  # per-step replay divergence
# colight diff old.py new.py [--json]
#
# # Deterministic screenshot via the same headless-Chrome path as
# # `colight render`: fixed viewport and device-pixel-ratio, waits for
# # render completion, renders at t=0 (no update entries applied).
# # --check renders twice in fresh tabs and byte-compares (exit 1 on
# # mismatch); JSON reports pixel size, sha256 and determinism.
# # Scene3d targets also report `coverage` in --json (fraction of canvas
# # pixels per component, read from the GPU pick buffer) and accept
# # --frame "C[:A-B]" (component index or type name, optional inclusive
# # instance ranges) to fit the camera on a selection before capture —
# # the zoom loop: coverage tells you a component is tiny, --frame gets
# # you a close-up.
# colight screenshot target.colight --out shot.png [--json] [--check]
# colight screenshot notebook.py --block ID --out shot.png
# colight screenshot scene.py --out closeup.png --frame "Ellipsoid:0-4"
#
# # Machine-legible screenshots. All composition happens post-capture in
# # Python — the render itself is untouched and stays byte-deterministic
# # (--check always compares the underlying renders).
# # --rulers expands the canvas with a labeled coordinate margin band
# # (big high-contrast labels + faint gridlines) in the exact page-pixel
# # space pick-at consumes: read a coordinate off the ruler, pass it
# # straight to pick-at instead of guessing from a downscaled view. In
# # the composed PNG, page coordinate v sits at pixel margin + v*dpr
# # (JSON reports rulers: {spacing, margin}).
# # --views renders one labeled contact sheet from camera presets fit to
# # the scene bounds (front, back, left, right, side, top, bottom, iso;
# # scene3d only; JSON reports per-view cameras). Agents pay per image
# # tile: one 2x2 grid can beat four separate images. Rulers are
# # single-view only — combining --rulers with --views is an error.
# # --frame with --views frames that selection from every preset.
# # --max-edge N scales the viewport (preserving aspect from
# # --width/--height, or the measured content aspect) so the PNG's long
# # edge is exactly N px — agents that know their harness's native input
# # size avoid a second lossy resampling.
# colight screenshot scene.py --out rulers.png --rulers [--json]
# colight screenshot scene.py --out sheet.png --views front,top,side,iso
# colight screenshot scene.py --out native.png --max-edge 1092
#
# # What is at point X,Y? Re-renders the target and queries the GPU pick
# # buffer (scene3d only; every query re-renders — nothing is persisted).
# # Coordinates are CSS pixels of the rendered page, origin top-left,
# # y down — the same space as a `colight screenshot` PNG at the same
# # --width/--height (at dpr 1). Hits within --radius (default 6px) are
# # ranked by distance then coverage; top hits include the instance's
# # dereferenced attribute values (center/color/size/... as rendered).
# # Exit 0 = hit, 1 = no hit within radius, 2 = error (incl. non-scene3d).
# # Reported alpha is decoration-aware (a decoration lowering an instance's
# # alpha is reflected in the hit's `alpha`, not the base 1.0).
# colight pick-at scene.py 240,180 [--radius 6] [--json]
#
# # --min-alpha T splits transparent occluders out of the hit list: hits
# # whose (decoration-aware) alpha is below T are skipped as `hits` and
# # reported in a separate `occluders` list, with `min_alpha` echoed. Use
# # it to pick the first *solid* surface under a translucent overlay (e.g.
# # a 35%-alpha topography draped over the geometry you actually want).
# colight pick-at scene.py 240,180 --min-alpha 0.5 [--json]
#
# # Where does a selection land on screen? Reports visible pixel count,
# # bbox and centroid (page CSS pixels) plus a visibility fraction:
# # visible pixels / the selection's unoccluded projected footprint
# # (measured by re-rendering only the selection). --out writes a PNG
# # with the selection highlighted via per-instance decorations and
# # everything else dimmed, so the selection can be seen.
# # Exit 0 = visible, 1 = entirely invisible (occluded or out of view),
# # 2 = error.
# colight pick-where scene.py --component Ellipsoid [--instances 0-4]
# colight pick-where scene.py --component 0 --out overlay.png [--json]
#
# # Golden verification. A golden pins three layers per visual-producing
# # block: the .colight artifact bytes, the canonicalized-structure hash
# # (per-run ids normalized), and the deterministic screenshot sha256 +
# # dimensions. On mismatch the report names the changed layer (structure
# # vs pixels); structure changes include a semantic diff summary
# # (max/mean |delta|, changed paths). --update pins/refreshes goldens and
# # reports what changed vs the previous set. The pixel layer is skipped
# # with a warning when Chrome is unavailable, or via --no-pixels.
# # Goldens live in <project_root>/tests/goldens/<relpath-of-target>/
# # (override with --goldens DIR).
# # Exit 0 = all match, 1 = mismatches, 2 = error, 3 = no goldens found.
# colight verify notebook.py [--json] [--no-pixels] [--goldens DIR]
# colight verify notebook.py --update
#
# # Daemon: keeps headless Chrome (and recently loaded scenes) warm so
# # tight loops of the render-path commands above skip the ~1-2s browser
# # launch per invocation. Discovery is automatic: the daemon writes
# # <project_root>/.colight_cache/daemon.json and screenshot / pick-at /
# # pick-where / verify use it transparently whenever the file points at a
# # live, version-matched daemon — otherwise they silently run direct.
# # Repeated queries against an unchanged target are served from a warm
# # scene cache keyed by content fingerprint (the same transitive cache
# # keys `colight run` uses), skipping re-evaluation and re-loading; user
# # code still runs only in the CLI process, never in the daemon.
# # The daemon shuts itself down after --idle-timeout seconds (default
# # 30 min) and owns up to --pool isolated Chrome instances (default 2)
# # so parallel queries parallelize.
# colight daemon start [--idle-timeout 1800] [--pool 2] [--foreground]
# colight daemon status [--json]   # pool, warm-scene and request stats
# colight daemon stop
# colight screenshot scene.py --out shot.png --no-daemon   # bypass per call
# ```
#
# Block ids are short hashes of each block's own source, so they survive
# edits to other blocks. `colight run` keeps its fingerprint records in
# `.colight_cache/cli-run/` under the project root; `--block ID` restricts
# detailed output to that block and its dependents. For `.py` targets,
# `colight screenshot` defaults to the last visual the file produces.
