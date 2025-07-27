# %%
# ruff: noqa: F401
import json
from typing import Any

import colight.plot_defs as plot_defs
from colight.core import Frames, Import, Slider, case, cond, katex, md
from colight.env import configure
from colight.layout import (
    Column,
    Grid,
    Hiccup,
    JSCall,
    JSCode,
    JSExpr,
    JSRef,
    LayoutItem,
    Ref,
    Row,
    State,
    js,
    onChange,
    ref,
)
from colight.plot_defs import (
    area,
    areaX,
    areaY,
    arrow,
    auto,
    autoSpec,
    axisFx,
    axisFy,
    axisX,
    axisY,
    barX,
    barY,
    bin,
    binX,
    binY,
    bollinger,
    bollingerX,
    bollingerY,
    boxX,
    boxY,
    cell,
    cellX,
    cellY,
    centroid,
    circle,
    cluster,
    column,
    contour,
    crosshair,
    crosshairX,
    crosshairY,
    delaunayLink,
    delaunayMesh,
    density,
    differenceX,
    differenceY,
    dodgeX,
    dodgeY,
    dot,
    dotX,
    dotY,
    filter,
    find,
    formatIsoDate,
    formatMonth,
    formatNumber,
    formatWeekday,
    frame,
    geo,
    geoCentroid,
    graticule,
    gridFx,
    gridFy,
    gridX,
    gridY,
    group,
    groupX,
    groupY,
    groupZ,
    hexagon,
    hexbin,
    hexgrid,
    hull,
    image,
    initializer,
    interpolatorBarycentric,
    interpolatorRandomWalk,
    legend,
    line,
    linearRegressionX,
    linearRegressionY,
    lineX,
    lineY,
    link,
    map,
    mapX,
    mapY,
    marks,
    normalize,
    normalizeX,
    normalizeY,
    numberInterval,
    pointer,
    pointerX,
    pointerY,
    raster,
    rect,
    rectX,
    rectY,
    reverse,
    ruleX,
    ruleY,
    scale,
    select,
    selectFirst,
    selectLast,
    selectMaxX,
    selectMaxY,
    selectMinX,
    selectMinY,
    shiftX,
    shiftY,
    shuffle,
    sort,
    sphere,
    spike,
    stackX,
    stackX1,
    stackX2,
    stackY,
    stackY1,
    stackY2,
    text,
    textX,
    textY,
    tickX,
    tickY,
    timeInterval,
    tip,
    transform,
    tree,
    treeLink,
    treeNode,
    utcInterval,
    valueof,
    vector,
    vectorX,
    vectorY,
    voronoi,
    voronoiMesh,
    waffleX,
    waffleY,
    window,
    windowX,
    windowY,
)
from colight.plot_extras import (
    bitmap,
    canvas_mark,
    ellipse,
    events,
    histogram,
    img,
    pixels,
    renderChildEvents,
)
from colight.plot_spec import MarkSpec, PlotSpec, new
from colight.protocols import Collector

# This module provides a composable way to create interactive plots using Observable Plot
# and AnyWidget, built on the work of pyobsplot.
#
# See:
# - https://observablehq.com/plot/
# - https://github.com/manzt/anywidget
# - https://github.com/juba/pyobsplot
#
#
# Key features:
# - Create plot specifications declaratively by combining marks, options and transformations
# - Compose plot specs using + operator to layer marks and merge options
# - Render specs to interactive plot widgets, with lazy evaluation and caching
# - Easily create grids to compare small multiples
# - Includes shortcuts for common options like grid lines, color legends, margins

new = new
"""Creates a new plot, given layers."""

html = Hiccup
"""Wraps a Hiccup-style list to be rendered as an interactive widget in the JavaScript runtime."""

repeat = JSRef("repeat")
"""For passing columnar data to Observable.Plot which should repeat/cycle.
eg. for a set of 'xs' that are to be repeated for each set of `ys`."""


def plot(options: dict[str, Any]) -> PlotSpec:
    """Create a new plot from options and marks."""
    plot_options = options.copy()
    plot_marks = plot_options.pop("marks", [])
    return new(plot_options, *plot_marks)


def constantly(x: Any) -> JSCode:
    """
    Returns a javascript function which always returns `x`.

    Typically used to specify a constant property for all values passed to a mark,
    eg. `plot.dot(values, fill=plot.constantly('My Label'))`. In this example, the
    fill color will be assigned (from a color scale) and show up in the color legend.
    """
    x = json.dumps(x)
    return js(f"()=>{x}")


def identity():
    """Returns a JavaScript identity function.

    This function creates a JavaScript snippet that represents an identity function,
    which returns its input unchanged.

    Returns:
        A JavaScript function that returns its first argument unchanged.
    """
    return js("(x) => x")


identity.for_json = lambda: identity()  # allow bare Plot.identity


def index():
    """Returns a JavaScript function that returns the index of each data point.

    In Observable Plot, this function is useful for creating channels based on
    the position of data points in the dataset, rather than their values.

    Returns:
        A JavaScript function that takes two arguments (data, index) and returns the index.
    """
    return js("(data, index) => index")


index.for_json = lambda: index()


def grid(x=None, y=None):
    """Sets grid lines for x and/or y axes."""
    # no arguments
    if x is None and y is None:
        return {"grid": True}
    return {
        "x": {"grid": x if x is not None else False},
        "y": {"grid": y if y is not None else False},
    }


def hideAxis(x=None, y=None):
    """Sets `{"axis": None}` for specified axes."""
    # no arguments
    if x is None and y is None:
        return {"axis": None}
    return {"x": {"axis": None if x else True}, "y": {"axis": None if y else True}}


def colorLegend():
    """Sets `{"color": {"legend": True}}`."""
    return {"color": {"legend": True}}


color_legend = colorLegend  # backwards compat


def clip() -> dict:
    """Sets `{"clip": True}`."""
    return {"clip": True}


def title(title: Any) -> dict:
    """Sets `{"title": title}`."""
    return {"title": title}


def subtitle(subtitle: Any) -> dict:
    """Sets `{"subtitle": subtitle}`."""
    return {"subtitle": subtitle}


def caption(caption: Any) -> dict:
    """Sets `{"caption": caption}`."""
    return {"caption": caption}


def width(width: Any) -> dict:
    """Sets `{"width": width}`."""
    return {"width": width}


def height(height: Any) -> dict:
    """Sets `{"height": height}`."""
    return {"height": height}


def size(size: Any, height: Any = None) -> dict:
    """Sets width and height, using size for both if height not specified."""
    return {"width": size, "height": height or size}


def aspectRatio(r: Any) -> dict:
    """Sets `{"aspectRatio": r}`."""
    return {"aspectRatio": r}


aspect_ratio = aspectRatio  # backwards compat


def inset(i: Any) -> dict:
    """Sets `{"inset": i}`."""
    return {"inset": i}


def colorScheme(name: Any) -> dict:
    """Sets `{"color": {"scheme": <name>}}`."""
    # See https://observablehq.com/plot/features/scales#color-scales
    return {"color": {"scheme": name}}


def domainX(d: Any) -> dict:
    """Sets `{"x": {"domain": d}}`."""
    return {"x": {"domain": d}}


def domainY(d: Any) -> dict:
    """Sets `{"y": {"domain": d}}`."""
    return {"y": {"domain": d}}


def domain(x: Any, y: Any = None) -> dict:
    """Sets domain for x and optionally y scales."""
    return {"x": {"domain": x}, "y": {"domain": y or x}}


def colorMap(mappings: Any) -> dict:
    """
    Adds colors to the plot's color_map. More than one colorMap can be specified
    and colors will be merged. This is a way of dynamically building up a color scale,
    keeping color definitions colocated with their use. The name used for a color
    will show up in the color legend, if displayed.

    Colors defined in this way must be used with `Plot.constantly(<name>)`.

    Example:

    ```
    plot = (
        Plot.dot(data, fill=Plot.constantly("car"))
        + Plot.colorMap({"car": "blue"})
        + Plot.colorLegend()
    )
    ```

    In JavaScript, colors provided via `colorMap` are merged into a
    `{color: {domain: [...], range: [...]}}` object.
    """
    return {"color_map": mappings}


color_map = colorMap  # backwards compat


def margin(*args: Any) -> dict:
    """
    Set margin values for a plot using CSS-style margin shorthand.

    Supported arities:
        margin(all)
        margin(vertical, horizontal)
        margin(top, horizontal, bottom)
        margin(top, right, bottom, left)

    Args:
        *args: Margin values as integers or floats, following CSS margin shorthand rules

    Returns:
        A dictionary mapping margin properties to their values
    """
    if len(args) == 1:
        return {"margin": args[0]}
    elif len(args) == 2:
        return {
            "marginTop": args[0],
            "marginBottom": args[0],
            "marginLeft": args[1],
            "marginRight": args[1],
        }
    elif len(args) == 3:
        return {
            "marginTop": args[0],
            "marginLeft": args[1],
            "marginRight": args[1],
            "marginBottom": args[2],
        }
    elif len(args) == 4:
        return {
            "marginTop": args[0],
            "marginRight": args[1],
            "marginBottom": args[2],
            "marginLeft": args[3],
        }
    else:
        raise ValueError(f"Invalid number of arguments: {len(args)}")


state = initialState = State

# Add this near the top of the file, after the imports
__all__ = [
    # ## Plot: Mark utilities
    # Useful for constructing arguments to pass to Mark functions.
    "constantly",
    "identity",
    "index",
    # ## Plot: Marks
    # The following are the original JavaScript docs for the built-in Observable Plot marks.
    # Usage is slightly different from Python.
    "area",
    "areaX",
    "areaY",
    "arrow",
    "auto",
    "barX",
    "barY",
    "boxX",
    "boxY",
    "cell",
    "cellX",
    "cellY",
    "circle",
    "dot",
    "dotX",
    "dotY",
    "image",
    "line",
    "lineX",
    "lineY",
    "link",
    "rect",
    "rectX",
    "rectY",
    "ruleX",
    "ruleY",
    "spike",
    "text",
    "textX",
    "textY",
    "vector",
    "vectorX",
    "vectorY",
    "waffleX",
    "waffleY",
    # ## Plot: Transforms
    "bin",
    "binX",
    "binY",
    "bollinger",
    "bollingerX",
    "bollingerY",
    "centroid",
    "cluster",
    "density",
    "differenceX",
    "differenceY",
    "dodgeX",
    "dodgeY",
    "filter",
    "find",
    "group",
    "groupX",
    "groupY",
    "groupZ",
    "hexbin",
    "hull",
    "map",
    "mapX",
    "mapY",
    "normalize",
    "normalizeX",
    "normalizeY",
    "reverse",
    "select",
    "selectFirst",
    "selectLast",
    "selectMaxX",
    "selectMaxY",
    "selectMinX",
    "selectMinY",
    "shiftX",
    "shiftY",
    "shuffle",
    "sort",
    "stackX",
    "stackX1",
    "stackX2",
    "stackY",
    "stackY1",
    "stackY2",
    "transform",
    "window",
    "windowX",
    "windowY",
    # ## Plot: Axes and grids
    "axisFx",
    "axisFy",
    "axisX",
    "axisY",
    "gridFx",
    "gridFy",
    "gridX",
    "gridY",
    "tickX",
    "tickY",
    # ## Plot: Geo features
    "geo",
    "geoCentroid",
    "graticule",
    "sphere",
    # ## Plot: Delaunay/Voronoi
    "delaunayLink",
    "delaunayMesh",
    "voronoi",
    "voronoiMesh",
    # ## Plot: Trees and networks
    "tree",
    "treeLink",
    "treeNode",
    # ## Plot: Interactivity
    "crosshair",
    "crosshairX",
    "crosshairY",
    "pointer",
    "pointerX",
    "pointerY",
    "tip",
    # ## Plot: Formatting and interpolation
    "formatIsoDate",
    "formatMonth",
    "formatNumber",
    "formatWeekday",
    "interpolatorBarycentric",
    "interpolatorRandomWalk",
    "numberInterval",
    "timeInterval",
    "utcInterval",
    # ## Plot: Other utilities
    "new",
    "frame",
    "hexagon",
    "hexgrid",
    "legend",
    "linearRegressionX",
    "linearRegressionY",
    "raster",
    "scale",
    "valueof",
    # ## Plot: Options Helpers
    "aspectRatio",
    "caption",
    "clip",
    "colorLegend",
    "colorMap",
    "colorScheme",
    "domain",
    "domainX",
    "domainY",
    "grid",
    "height",
    "hideAxis",
    "inset",
    "margin",
    "repeat",
    "size",
    "subtitle",
    "title",
    "width",
    # # Colight Core
    # ## Interactivity
    "State",
    "onChange",
    "Frames",
    "Slider",
    # ## Layout
    "Column",
    "Grid",
    "Row",
    # ## Flow Control
    "cond",
    "case",
    # ## JavaScript Interop
    "Import",
    "js",
    # ## Formatting
    "html",
    "md",
    "katex",
    # ## Re-Exports
    "LayoutItem",
    "state",
    "State",
    "initialState",
]
