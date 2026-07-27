# # Wolfpass: a porphyry copper deposit in colight
#
# This document rebuilds the *viewing* experience of a geological modeling
# package (Leapfrog-style) on colight's Scene3D, using the MIT-licensed
# **Wolfpass** sample deposit distributed in the Open Mining Format (OMF v1).
#
# Download the data file (20.9 MB) next to this document as `wolfpass.omf`:
# https://github.com/OpenGeoVis/omfvista/raw/master/assets/test_file.omf
#
# The project contains 55 drillhole collars, an 8,583-segment drillhole trace
# set with assay data (Cu, Au, Ag, Mo, ...), a topography surface, five
# geology surfaces, and a 110x160x96 block model of estimated copper grade.

import os
from pathlib import Path


import colight.plot as Plot
from colight import scene3d
from colight.omf_loader import load_omf

OMF_PATH = os.environ.get("WOLFPASS_OMF", str(Path(__file__).parent / "wolfpass.omf"))

project = load_omf(OMF_PATH)

# Coordinates in the file are UTM-scale (x ~445,000 m), which exceed float32
# GPU precision. `load_omf` keeps the geometry in those true world coordinates
# and hands back the bounds midpoint on `project.center`; passing it as
# `Scene(origin=project.center)` shifts every position into float32-safe range
# once, at the serialization boundary. `pick-at` adds the offset back, so
# reported positions come out in world coordinates.

ORIGIN = project.center

collars = project.points["collar"]
drillholes = project.line_sets["wolfpass_WP_assay"]
topo = project.surfaces["Topography"]
block_model = project.volumes["Block Model"]

# A Leapfrog-style color scheme for the geology surfaces.

GEOLOGY_COLORS = {
    "Cover": [0.62, 0.77, 0.55],
    "Dacite": [0.91, 0.60, 0.62],
    "Intermineral diorite": [0.94, 0.63, 0.33],
    "Early Diorite": [0.65, 0.46, 0.72],
    "Basement": [0.55, 0.55, 0.58],
}

# ## Overview: topography, geology and drillholes
#
# The classic project overview — semi-transparent topography draped over the
# deposit, the five modeled geology volumes as solid surfaces, drillhole
# traces fanning out beneath the collars. No hand-tuned camera: with no
# explicit camera the scene auto-fits its own bounds (near/far derived from
# the deposit's extent), so a ~3 km scene frames correctly out of the box.
#
# Note the layer order: Scene3D composites transparency in layer order
# (there is no cross-component depth sort), so the transparent topography
# must be listed *last* or everything beneath it disappears.

overview = scene3d.Scene(
    *[
        project.surfaces[name].mesh(color=color)
        for name, color in GEOLOGY_COLORS.items()
    ],
    drillholes.line_segments(color=[0.75, 0.75, 0.78], size=6.0),
    collars.point_cloud(color=[0.95, 0.26, 0.21], size=30.0),
    topo.mesh(
        color=[0.82, 0.76, 0.65],
        decorations=[scene3d.deco([0], alpha=0.35)],
    ),
    origin=ORIGIN,
)

overview

# ## Drillholes colored by copper grade
#
# Each of the 8,583 assay intervals is a line segment colored by Cu %
# through Scene3D's first-class `color_by` colormap support — the viridis
# scale is clamped to 0–2 % Cu and the scene renders a legend for it, so
# "yellow = 2 % Cu" is readable off the image (and reported by
# `colight inspect` / `screenshot --json`). The hot intervals cluster
# along the dipping mineralized zone.

scene3d.Scene(
    drillholes.line_segments(
        color_by="CU_pct", domain=(0.0, 2.0), label="Cu %", size=10.0
    ),
    collars.point_cloud(color=[0.95, 0.26, 0.21], size=25.0),
    topo.mesh(
        color=[0.82, 0.76, 0.65],
        decorations=[scene3d.deco([0], alpha=0.25)],
    ),
    origin=ORIGIN,
)

# ## Drillholes with a channel selector (ParaView-style)
#
# ParaView's color-by dropdown *is* the viewing UI: geologists truncate
# exports to a handful of attributes and re-export to change which one drives
# the colors. Scene3D's `color_channels` makes that switch client-side — every
# channel's raw values ship ONCE, and a `$state`-driven dropdown recolors the
# active channel in the browser (no re-export, no server round-trip).
#
# Here the drillholes carry a continuous **Cu %** channel and a **grade class**
# categorical channel synthesized by binning Cu (the Wolfpass assay has no
# native rock-group attribute; `binned_categorical` derives one). Categorical
# channels report their class *labels* on `pick-at` — "click an interval, read
# the data row".

GRADE_CLASS = drillholes.binned_categorical(
    "CU_pct",
    edges=[0.3, 0.7, 1.2],
    labels=["trace", "low", "moderate", "high"],
    colors=[
        [0.85, 0.85, 0.85],
        [0.55, 0.75, 0.95],
        [0.95, 0.75, 0.35],
        [0.85, 0.20, 0.20],
    ],
)

(
    scene3d.Scene(
        drillholes.line_segments_channels(
            channels={
                "Cu %": {
                    "attribute": "CU_pct",
                    "cmap": "viridis",
                    "domain": (0.0, 2.0),
                },
                "Grade class": GRADE_CLASS,
            },
            active_channel=Plot.js("$state.color_channel"),
            size=10.0,
        ),
        collars.point_cloud(color=[0.95, 0.26, 0.21], size=25.0),
        topo.mesh(
            color=[0.82, 0.76, 0.65],
            decorations=[scene3d.deco([0], alpha=0.25)],
        ),
        origin=ORIGIN,
    )
    | Plot.Slider(
        "color_channel",
        options=["Cu %", "Grade class"],
        init="Cu %",
        label="Color by",
    )
)

# Flipping the dropdown from "Cu %" to "Grade class" recolors all 8,583
# intervals in the browser — the viridis grade ramp becomes four discrete
# grade classes (with a legend of swatches), and `pick-at` on any interval now
# reports both `channels: {"Cu %": 0.83, "Grade class": "moderate"}`. Nothing
# re-uploads: only the per-instance colors buffer is rewritten.

# ## Block model above cutoff
#
# The block model holds estimated Cu grade for 1,689,600 ten-metre cells.
# Rendering all of them as cuboids is possible but heavy; a grade *cutoff*
# is also how a geologist actually reads a block model.
#
# Scene3D's per-instance `filter_by` makes this ONE layer: every cell above a
# base 0.5 % floor is uploaded once, and the slider drives `filter_by.min`
# (`$state.cutoff`) so the shell shrinks *client-side* as the cutoff rises —
# no Python round-trip, no re-upload, and filtered-out cells become unpickable
# (so `pick-where` / coverage report the visible shell honestly). This
# replaces the previous three baked `Plot.cond` shells (0.5 / 0.7 / 1.0),
# cutting the payload to a single ~6.6 MB cuboid layer.

BASE_CUTOFF = 0.5

block_layer = block_model.cuboids_filter_by(
    "CU_pct",
    min=Plot.js("$state.cutoff"),
    base_cutoff=BASE_CUTOFF,
    domain=(0.0, 2.0),
    label="Cu %",
)

grades = block_model.cell_attributes["CU_pct"]
base_count = int((grades >= BASE_CUTOFF).sum())

(
    scene3d.Scene(
        drillholes.line_segments(color=[0.55, 0.55, 0.58], size=4.0),
        block_layer,
        topo.mesh(
            color=[0.82, 0.76, 0.65],
            decorations=[scene3d.deco([0], alpha=0.2)],
        ),
        origin=ORIGIN,
    )
    | Plot.Slider(
        "cutoff",
        init=1.0,
        range=[BASE_CUTOFF, 2.0],
        step=0.05,
        label=Plot.js("`Cu cutoff: ${$state.cutoff.toFixed(2)} %`"),
    )
)

# The uploaded shell holds 274,780 blocks (grade >= 0.5 %). Sliding the cutoff
# to 1.0 % hides all but the 106,780 highest-grade cells with no server round
# trip. The high-grade core sits inside the Early Diorite volume, which is what
# the geology surfaces in the overview scene suggest.

# ## Section view: slicing the deposit
#
# A block-model shell is still a solid mass; to *read* the grade distribution a
# geologist cuts a section through it. `Scene(clip_planes=...)` slices the whole
# scene with a half-space plane — every layer (blocks, drillholes, topo) is cut
# in both the render AND pick passes, so the exposed interior is visible and
# `pick-at` reports the block the section reveals, not the outer shell.
#
# The plane offset is origin-relative (the same post-`origin` space the shader
# uses), so a `$state` slider sweeps the cut north-to-south with no re-upload —
# only the clip-plane uniform changes. We derive the sweep range from the block
# model's northing (Y) extent, origin-relative.

block_centers = block_model.cell_centers()  # (N, 3) world coords
y_world = block_centers[:, 1]
y_lo = float(y_world.min() - ORIGIN[1])
y_hi = float(y_world.max() - ORIGIN[1])
Y_MID = round((y_lo + y_hi) / 2.0, 1)

(
    scene3d.Scene(
        drillholes.line_segments(color=[0.55, 0.55, 0.58], size=4.0),
        block_layer,
        topo.mesh(
            color=[0.82, 0.76, 0.65],
            decorations=[scene3d.deco([0], alpha=0.2)],
        ),
        origin=ORIGIN,
        # Keep the half-space south of the cut (northing <= section_y); sweeping
        # section_y walks the exposed face across the deposit.
        clip_planes=[{"normal": [0, 1, 0], "offset": Plot.js("$state.section_y")}],
    )
    | Plot.Slider(
        "section_y",
        init=Y_MID,
        range=[round(y_lo, 1), round(y_hi, 1)],
        step=10.0,
        label=Plot.js("`Section (northing): ${$state.section_y.toFixed(0)} m`"),
    )
)

# Sliding `section_y` from the mid-deposit toward the north exposes deeper
# cross-sections of the grade shell — the high-grade core and its drillhole
# support read directly off the cut face. Because clipping is per-fragment, the
# cut passes THROUGH blocks (hollow shells show on the section, v1 does not
# cap-fill), and pick-at on the exposed face reports the interior block.
