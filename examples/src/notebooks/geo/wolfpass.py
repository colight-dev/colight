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

# Coordinates in the file are UTM-scale (x ~445,000 m). `load_omf` re-centers
# everything about the midpoint of the project bounds so positions fit
# comfortably in the float32 buffers the GPU uses; the offset is kept in
# `project.center` for mapping back to world coordinates.

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

# Scene3D's camera defaults (near=0.001, far=100) assume unit-scale scenes;
# a deposit is ~3 km across, so the clip planes must be set explicitly.
CAMERA = {
    "defaultCamera": {
        "position": [3600, -3400, 2800],
        "target": [0, 0, -200],
        "up": [0, 0, 1],
        "fov": 45,
        "near": 5.0,
        "far": 50000.0,
    }
}

# ## Overview: topography, geology and drillholes
#
# The classic project overview — semi-transparent topography draped over the
# deposit, the five modeled geology volumes as solid surfaces, drillhole
# traces fanning out beneath the collars.
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
    CAMERA,
)

overview

# ## Drillholes colored by copper grade
#
# Each of the 8,583 assay intervals is a line segment colored by Cu %.
# Scene3D has no built-in colormap, so grades are mapped to viridis colors
# in Python (`colight.omf_loader.colormap`). The color scale is clamped to
# 0–2 % Cu; the hot intervals cluster along the dipping mineralized zone.

scene3d.Scene(
    drillholes.line_segments(color_by="CU_pct", vmin=0.0, vmax=2.0, size=10.0),
    collars.point_cloud(color=[0.95, 0.26, 0.21], size=25.0),
    topo.mesh(
        color=[0.82, 0.76, 0.65],
        decorations=[scene3d.deco([0], alpha=0.25)],
    ),
    CAMERA,
)

# ## Block model above cutoff
#
# The block model holds estimated Cu grade for 1,689,600 ten-metre cells.
# Rendering all of them as cuboids is possible but heavy; a grade *cutoff*
# is also how a geologist actually reads a block model. There is no
# client-side attribute filter in Scene3D, so each cutoff shell is
# precomputed in Python and the slider toggles between them.

CUTOFFS = [0.5, 0.7, 1.0]

block_layers = [
    Plot.cond(
        Plot.js(f"$state.cutoff_idx === {i}"),
        block_model.cuboids("CU_pct", cutoff=c, vmin=0.0, vmax=2.0),
    )
    for i, c in enumerate(CUTOFFS)
]

counts = [int((block_model.cell_attributes["CU_pct"] >= c).sum()) for c in CUTOFFS]

(
    scene3d.Scene(
        drillholes.line_segments(color=[0.55, 0.55, 0.58], size=4.0),
        *block_layers,
        topo.mesh(
            color=[0.82, 0.76, 0.65],
            decorations=[scene3d.deco([0], alpha=0.2)],
        ),
        CAMERA,
    )
    | Plot.Slider(
        "cutoff_idx",
        init=2,
        range=[0, len(CUTOFFS) - 1],
        label=Plot.js(
            "`Cu cutoff: ${%1[$state.cutoff_idx]} % (${%2[$state.cutoff_idx].toLocaleString()} blocks)`",
            CUTOFFS,
            counts,
        ),
    )
)

# At a 0.5 % cutoff the shell holds 274,780 blocks; at 1.0 %, 106,780. The
# high-grade core sits inside the Early Diorite volume, which is what the
# geology surfaces in the overview scene suggest.
