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

## Perception commands (cheapest first)

```bash
colight blocks FILE.py --json        # block graph: ids, line ranges, provides/deps
colight inspect TARGET --json        # structure without pixels: components, array
                                     # schemas, state keys + sanity warnings
colight diff A B --json              # semantic diff (.py or .colight): per-array
                                     # max/mean |delta|, changed fraction, bounds drift
colight screenshot TARGET --out x.png --json   # deterministic pixels; --check verifies
                                               # byte-identical double-render
```

- **Blank/empty scene?** `colight inspect` first — its warnings (NaN values, empty arrays,
  alphas ≈ 0, degenerate bounds, length mismatches) catch most causes without rendering.
- **Did my change do what I intended?** `colight diff old.py new.py` (or two artifacts)
  before reaching for pixels.
- **Need to see it?** `colight screenshot`, then Read the PNG. Screenshots are
  deterministic (fixed viewport/dpr, t=0), so before/after hashes are meaningful.
- Refer to blocks by the stable ids from `colight blocks` (content hashes of each block's
  own source — they survive edits to other blocks), not by line numbers.

## Conventions

- Run Python via `uv run` (e.g. `uv run colight ...` inside this repo).
- `colight run` state lives in `.colight_cache/` (gitignored) — safe to delete to reset.
- Other CLI verbs: `live` (browser live-reload), `publish` (md/html), `render`
  (png/pdf/mp4), `view` (open artifact). See `docs/src/colight_docs/learn-more/cli.py`.
