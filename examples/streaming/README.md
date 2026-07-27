# Streaming: any language writes, colight measures

A `.colight` file is an append-only stream. A producer opens one and appends a
state update per tick; colight's CLI reads, diffs, and renders the result
without caring what wrote it.

This example makes that concrete with a producer that contains **no Python at
all**. `deforming_surface.mjs` is a Node script depending only on
`@colight/format`: it generates a traveling wave over a grid mesh and appends
one update entry per tick. Everything downstream — `inspect`, `diff`, `render`
— is the ordinary colight CLI, working on bytes a JavaScript program wrote.

That is the point of the example. It shows the loop, not a shape.

## Run it

Build once so `@colight/format` is compiled to plain JavaScript:

```bash
yarn build
```

Produce an artifact — 60 ticks of a 32×32 surface:

```bash
node examples/streaming/deforming_surface.mjs --out /tmp/surface.colight --ticks 60 --grid 32
```

```
wrote /tmp/surface.colight: 1024 vertices, 60 update entries
```

### What is in it

```bash
colight inspect /tmp/surface.colight
```

```
/tmp/surface.colight
  update entries: 60
  component Column×1
  component scene3d.Scene×1
  component scene3d.Mesh×1 (1 instances)
  array ast[2][1].layers[0]/scene3d.Mesh[0].geometry.indices: uint32 [5766] min=0 max=1023
  array ast[2][1].layers[0]/scene3d.Mesh[0].centers: float32 [3] min=0 max=0
  array ast[2][1].layers[0]/scene3d.Mesh[0].color: float64 [3] min=0.29 max=0.89
  array state.positions: float32 [1024, 3] min=-0.5 max=0.5
  state keys: 2
  buffers: 3 (35364 bytes)
```

### What changed between two runs

`--phase` shifts the wave. Two runs that differ only in `--phase` are exactly
what `colight diff` is for — same geometry, same tick count, different motion:

```bash
node examples/streaming/deforming_surface.mjs --out /tmp/a.colight --ticks 20 --phase 0
node examples/streaming/deforming_surface.mjs --out /tmp/b.colight --ticks 20 --phase 0.5
colight diff /tmp/a.colight /tmp/b.colight
```

```
A: /tmp/a.colight (1 visual(s))
B: /tmp/b.colight (1 visual(s))
pair 0:
  array state.positions: max |Δ| 0.2195 mean 0.02431 changed 33.3%
  state changed: positions
updates: 20/20 differ (first at update 0)
  update 0: state.positions 33.3% changed, max |Δ| 0.2084
  update 1: state.positions 33.3% changed, max |Δ| 0.1984
  …
1 array(s) changed, max |Δ| 0.2195 in state.positions; 1 state key change(s);
first divergence at update 0; 20/20 updates differ
```

`diff` aligns update entries step by step, so it reports where two runs first
diverge, not merely that they do.

### Rendering a particular tick

`colight render --frame N` applies the first `N` update entries before
rendering, which is how you screenshot a chosen point in the stream:

```bash
colight render /tmp/surface.colight --frame 5  -o /tmp/tick5.png  --width 400
colight render /tmp/surface.colight --frame 40 -o /tmp/tick40.png --width 400
```

The two images differ: the wave has travelled. `--last` renders the final
state, and omitting both renders the initial one.

Note that `colight screenshot` does **not** take an update index — its
`--frame` option is camera framing (which component to fit the camera on), and
it always renders the initial state. Use `colight render` when you need a
specific tick.

### Reading while it is still being written

`--delay` paces the producer so you can watch a reader keep up. In one shell:

```bash
node examples/streaming/deforming_surface.mjs --out /tmp/live.colight --ticks 100 --delay 100
```

In another, while that is running:

```bash
colight inspect /tmp/live.colight     # run it repeatedly
```

The reported `update entries:` count climbs, and every read is of a complete,
valid artifact. A reader that arrives in the middle of an append sees the
entries written so far and silently ignores the partial one — that tolerance is
part of the format, so no locking or coordination is needed between the two
processes.

## Why this works

Three rules, stated in full in [format.md §4.1][streaming]:

- **Append-only.** The producer never rewrites anything it already wrote, so
  there is no window in which the file is inconsistent.
- **Alignment preserved.** Every entry is padded to a multiple of 8, so each
  appended entry — and every buffer inside it — stays 8-byte aligned, which is
  what lets a reader build zero-copy typed arrays over the bytes.
- **Readers tolerate a torn tail.** A half-written final entry is dropped, so a
  concurrent reader always sees a monotonically growing, never-corrupt prefix.

The producer holds the file open across all its appends
(`ColightFileWriter.create`), which is roughly three times faster than
reopening per entry and just as safe.

## Doing this from Python

The same API exists on the Python side, deliberately mirrored:

```python
import colight.plot as Plot
from colight.format import ColightWriter

with ColightWriter.create("surface.colight", initial_visual) as writer:
    for tick in range(1, 100):
        writer.append(Plot.State({"positions": positions(tick)}))
```

`colight.format.append_update(path, update)` is the one-shot
open-append-close form for occasional writes.

[streaming]: ../../docs/src/colight_docs/format.md#41-appending-the-streaming-contract
