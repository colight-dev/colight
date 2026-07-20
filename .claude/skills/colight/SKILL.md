---
name: colight
description: Work on or with colight visualizations - the edit/verify loop for notebook-style .py documents and .colight artifacts. Use when editing colight notebooks, debugging blank/wrong visuals, or verifying visualization changes.
---

# Colight agent loop

Colight documents are plain `.py` files: comments are markdown prose, blank lines separate
blocks, expressions render as visuals. Full API guide: `docs/src/colight_docs/llms.py`.

## The loop

After every meaningful edit to a colight `.py` file, run:

```bash
colight run FILE.py            # per-block: cached | ran:unchanged | ran:changed | error
```

Block statuses tell you exactly what your edit affected — `ran:changed` on blocks you did
not intend to change is a regression signal. Add `--json` for structured output; nonzero
exit means a block errored (errors include block id, line range, and user-frame traceback).
`--force` re-executes everything (statuses still compare against stored fingerprints);
`# | pragma: always-eval` on a block means it never reports `cached`.

## Pinning known-good state

```bash
colight verify FILE.py --update      # pin goldens: artifact + structure hash +
                                     # screenshot sha per visual block
colight verify FILE.py [--json]      # exit 0 match | 1 mismatch | 2 error | 3 no goldens
```

On mismatch the report names the changed layer (structure vs pixels) and includes a
semantic diff (max/mean |delta|, changed paths). `--update` reports the same diff before
overwriting, so refreshing goldens is an informed act. Goldens live in
`<project_root>/tests/goldens/` (`--goldens DIR` overrides); `--no-pixels` skips the
screenshot layer (auto-skipped without Chrome).

## Perception commands (cheapest first)

```bash
colight blocks FILE.py --json        # block graph: ids, line ranges, provides/deps
colight inspect TARGET --json        # structure without pixels: components, array
                                     # schemas, state keys + sanity warnings
colight diff A B --json              # semantic diff (.py or .colight): per-array
                                     # changed count/fraction (integer grids lead with
                                     # these), max/mean |delta|, bounds drift. Artifacts
                                     # with update entries diff per-step (aligned by
                                     # index): first diverging update + how many differ
colight screenshot TARGET --out x.png --json   # deterministic pixels; --check verifies
                                               # byte-identical double-render
```

- **Blank/empty scene?** `colight inspect` first — its warnings (NaN values, empty arrays,
  alphas ≈ 0, degenerate bounds, length mismatches, and a `camera-frustum` warning when an
  explicit camera's near/far can't contain the scene) catch most causes without rendering.
  Then `colight screenshot --json`: a scene rendering ≥99% background emits a
  `mostly-background` warning ("geometry may be outside the camera frustum or fully
  transparent") — the signal that an all-black render is genuinely wrong, not just dark
  (identical before/after hashes alone can't tell consistently-broken from consistently-right).
  For scene3d in far-from-origin coordinates (UTM etc.), pass `Scene(origin=...)` so the
  camera auto-fits in float32-safe space.
- **What do the colors mean?** Read `legends` from `inspect`/`screenshot --json`
  (`{component, label, cmap, domain, categorical}` for every `color_by`-colored
  component) instead of guessing color meanings from pixels.
- **Filters vs selections (both are per-instance masks).** A `filter_by`
  (`{values, min, max}`) changes _what is visible and pickable_ — filtered-out
  instances are collapsed and unpickable, so `pick-where`/coverage counts move
  with the threshold; read active `filters` (`{component, label?, min, max}`)
  from `inspect`/`screenshot --json` to know the view is filtered. A **selection**
  is a _shared named referent_: named per-instance masks resident in
  `$state.selections` that highlight their instances and are addressable by name
  — read `selections` from `inspect`/`screenshot --json`, `pick-at` reports each
  hit's `selections: [names]`, and `pick-where --selection NAME` /
  `screenshot --frame NAME` resolve the name. Human clicks
  (`scene3d.toggle_selection`) and agent predicates converge on the same object.
- **Section / clipping planes (`Scene(clip_planes=...)`).** A scene-level cut that
  slices the _whole_ scene with a half-space plane (`{normal, offset}`, or the
  origin-aware `{normal, point}` form) so interior structure shows — the section
  view for block models / drillholes. Unlike `filter_by`, it cuts _through_
  geometry, and clipping applies in the pick pass too: `pick-at` on the exposed
  cut face reports the interior instance behind the section, not the outer shell.
  `offset` may be a `Plot.js("$state...")` ref so a slider sweeps the section with
  no re-upload. Read active `clip_planes` (`{normal, offset}` or `{normal,
state_key}`) from `inspect`/`screenshot --json` to know the view is sectioned; a
  `section-excludes-scene` warning fires if the planes clip away everything (blank
  render). Max 8 planes; v1 does not cap/fill the cut (hollow shells show).
- **Annotation callouts (`scene3d.Annotation` + `scene3d.annotate`).** A named text
  callout anchored in data space — a marker dot + leader line + label rendered as a
  DOM overlay (captured in screenshots, like the legend). Anchor it to a world
  `position=[x,y,z]` (origin-aware, like a clip-plane `point`) or to a component
  instance (`component=C, instance=I`, resolved to that instance's center). Like
  selections, annotations live in `$state.annotations`, so they sync Python↔JS and
  persist into `.colight` artifacts, and the NAME is a shared referent. Machine-
  legible: `inspect` reports `{name, text, anchor}`; `screenshot --json` adds each
  callout's resolved `world` position and projected `screen` position (in pick-at
  pixel space) plus a `visible` flag (false when the anchor is behind the camera or
  off-screen); `pick-at` reports instance-anchored callouts as each hit's
  `annotations: [names]`. An agent reads a callout's `screen` and can `pick-at`
  right next to it.
- **Did my change do what I intended?** `colight diff old.py new.py` (or two artifacts)
  before reaching for pixels.
- **Need to see it?** `colight screenshot`, then Read the PNG. Screenshots are
  deterministic (fixed viewport/dpr, t=0), so before/after hashes are meaningful.
- Refer to blocks by the stable ids from `colight blocks` (content hashes of each block's
  own source — they survive edits to other blocks), not by line numbers.

## Scene3d pick queries (perceive cheaply → locate → zoom → dereference)

For 3D scenes, the GPU pick buffer answers "what's on screen" without eyeballing pixels.
Every query re-renders deterministically; nothing is persisted. Coordinates are CSS px of
the rendered page (origin top-left, y down) — the same space as a screenshot PNG at the
same `--width/--height` (dpr 1).

```bash
colight screenshot scene.py --out s.png \
    --rulers --json                              # 1. perceive: `coverage` = fraction of
                                                 #    canvas per component + background;
                                                 #    --rulers adds labeled coordinate
                                                 #    rulers so step 2's X,Y is READ off
                                                 #    the image, not guessed
colight pick-at scene.py X,Y [--radius 6] --json # 2. locate: ranked hits at a point,
                                                 #    with dereferenced values (center,
                                                 #    color, size — as rendered)
colight pick-where scene.py --component C \
    [--instances A-B] [--out overlay.png] --json # 3. selection → screen truth: bbox,
                                                 #    centroid, visibility (visible px /
                                                 #    unoccluded footprint); --out draws
                                                 #    the selection highlighted.
                                                 #    Or --selection NAME to address a
                                                 #    named $state.selections entry.
colight screenshot scene.py --out zoom.png \
    --frame "C[:A-B]" --json                     # 4. zoom: camera fit to the selection
                                                 #    (--frame NAME also resolves a named
                                                 #    selection); coverage fraction rises
```

- **Recommended flow**: `screenshot --rulers` → Read the PNG and take X,Y straight from
  the ruler labels → `pick-at X,Y`. Ruler labels are the exact page-pixel space pick-at
  consumes (in the composed PNG, coordinate v sits at pixel `margin + v`; JSON reports
  `rulers: {spacing, margin}`). This kills the biggest pick-at error source: guessing
  pixel coordinates from a downscaled view.
- `--views front,top,side,iso` composes one labeled contact sheet from bounds-fit camera
  presets (scene3d only; per-view cameras in JSON) — you pay per image tile, and one 2×2
  grid usually beats four separate images. Rulers are single-view only (combining errors);
  `--frame` + `--views` frames that selection from every preset.
- `--max-edge N` makes the PNG's long edge exactly N px (aspect preserved) — render at
  your harness's native input size once instead of being resampled twice.
- `--component` takes an index or type name (as reported by `coverage`).
- Exit codes: pick-at 1 = no hit; pick-where 1 = selection entirely invisible
  (`projected_bbox` says where it _would_ land if merely occluded); 2 = error, including
  non-scene3d targets — these commands are scene3d-only.
- Use the same `--width/--height` across screenshot and pick queries so coordinates line
  up; the payload's `scene.rect` maps page pixels to the canvas.

## Daemon (faster loops)

Render-path commands (screenshot / pick-at / pick-where / verify) discover a running
`colight daemon` automatically (via `.colight_cache/daemon.json`) and use its warm Chrome
pool + scene cache — a repeated pick-at on an unchanged file drops from ~1.5s to ~0.4s.
No flags needed; it just works. Run `colight daemon start` once to warm up before a tight
loop (it self-stops after 30 idle minutes); `--no-daemon` bypasses it per call.

## Conventions

- Run Python via `uv run` (e.g. `uv run colight ...` inside this repo).
- `colight run` state lives in `.colight_cache/` (gitignored) — safe to delete to reset.
- Other CLI verbs: `live` (browser live-reload), `publish` (md/html), `render`
  (png/pdf/mp4), `view` (open artifact). See `docs/src/colight_docs/learn-more/cli.py`.
