from typing import Any

import colight.plot_defs as defs
from colight.components.bitmap import bitmap
from colight.layout import JSCode, JSRef, js
from colight.plot_spec import MarkSpec, PlotSpec, new


def histogram(
    values,
    thresholds=None,
    interval=None,
    domain=None,
    cumulative=False,
    layout={"width": 200, "height": 200, "inset": 0},
    **plot_opts,
) -> PlotSpec:
    """
    Create a histogram plot from the given values.

    Args:

    values (list or array-like): The data values to be binned and plotted.
    mark (str): 'rectY' or 'dot'.
    thresholds (str, int, list, or callable, optional): The thresholds option may be specified as a named method or a variety of other ways:

    - `auto` (default): Scott’s rule, capped at 200.
    - `freedman-diaconis`: The Freedman–Diaconis rule.
    - `scott`: Scott’s normal reference rule.
    - `sturges`: Sturges’ formula.
    - A count (int) representing the desired number of bins.
    - An array of n threshold values for n - 1 bins.
    - An interval or time interval (for temporal binning).
    - A function that returns an array, count, or time interval.

     Returns:
      PlotSpec: A plot specification for a histogram with the y-axis representing the count of values in each bin.
    """
    bin_options = {"x": {}, "tip": True, **plot_opts}
    for option, value in [
        ("thresholds", thresholds),
        ("interval", interval),
        ("domain", domain),
    ]:
        if value is not None:
            bin_options["x"][option] = value
    if cumulative:
        bin_options["y"] = {"cumulative": True}
    return (
        defs.rectY(values, defs.binX({"y": "count"}, bin_options))
        + defs.ruleY([0])
        + layout
    )


def renderChildEvents(options: dict[str, Any] = {}, **kwargs) -> JSRef:
    """
    Creates a render function that adds drag-and-drop and click functionality to child elements of a plot.
    Must be passed as the 'render' option to a mark, e.g.:

        Plot.dot(data, render=Plot.renderChildEvents(
            onDrag=update_position,
            onClick=handle_click
        ))

    This function enhances the rendering of plot elements by adding interactive behaviors such as dragging, clicking, and tracking position changes. It's designed to work with Observable Plot's rendering pipeline.

    Args:
        options (dict): Configuration options for the child events
        **kwargs: Event handlers passed as keyword arguments:
            - `onDragStart` (callable): Callback function called when dragging starts
            - `onDrag` (callable): Callback function called during dragging
            - `onDragEnd` (callable): Callback function called when dragging ends
            - `onClick` (callable): Callback function called when a child element is clicked

    Returns:
        A render function to be used in the Observable Plot rendering pipeline.
    """
    return JSRef("renderChildEvents")({**options, **kwargs})


def canvas_mark(user_canvas_fn):
    """
    Create a custom Plot mark that renders using HTML5 Canvas within an SVG context.

    This function enables high-performance custom visualization by allowing direct
    canvas drawing operations while maintaining compatibility with Observable Plot's
    coordinate system and layout. The canvas is automatically scaled for high-DPI
    displays and embedded as a foreignObject in the SVG.

    Args:
        user_canvas_fn (callable): A function that performs the canvas drawing operations.
            Called with arguments:
            - ctx: Canvas 2D rendering context (pre-scaled for device pixel ratio)
            - scales: Observable Plot scale functions for converting data to pixel coordinates
            - dim: Object with width/height properties in CSS pixels
            - mark_context: Plot mark context extended with 'mark_canvas' property

    Returns:
        PlotSpec: A plot specification that can be included in plot compositions.

    Example:
        ```python
        def draw_circles(ctx, scales, dim, mark_context):
            ctx.fillStyle = 'red'
            ctx.beginPath()
            ctx.arc(dim.width/2, dim.height/2, 50, 0, 2*np.pi)
            ctx.fill()

        Plot.canvas_mark(draw_circles) + Plot.domain([0, 1])
        ```
    """
    return new(
        js(
            """(_indexes, scales, _values, dim, mark_context) => {
    const devicePixelRatio = window.devicePixelRatio || 1;

    /* ---- build the canvas-in-SVG wrapper -------------------------------- */
    const svgNS = "http://www.w3.org/2000/svg";
    const fo    = document.createElementNS(svgNS, "foreignObject");
    fo.setAttribute("width",  dim.width);
    fo.setAttribute("height", dim.height);

    const canvas = document.createElement("canvas");
    
    // Set actual canvas size (high-res)
    canvas.width  = dim.width * devicePixelRatio;
    canvas.height = dim.height * devicePixelRatio;
    
    // Set CSS size (display size)
    canvas.style.width = dim.width + 'px';
    canvas.style.height = dim.height + 'px';
    fo.appendChild(canvas);

    const ctx = canvas.getContext("2d");
    ctx.scale(devicePixelRatio, devicePixelRatio);
                   
    user_canvas_fn(ctx, scales, dim, {...mark_context, mark_canvas: canvas})
    
    return fo;
}""",
            user_canvas_fn=user_canvas_fn,
        )
    )


def ellipse(values: Any, options: dict[str, Any] = {}, **kwargs) -> PlotSpec:
    """
    Returns a new ellipse mark for the given *values* and *options*.

    If neither **x** nor **y** are specified, *values* is assumed to be an array of
    pairs [[*x₀*, *y₀*], [*x₁*, *y₁*], [*x₂*, *y₂*, …] such that **x** = [*x₀*,
    *x₁*, *x₂*, …] and **y** = [*y₀*, *y₁*, *y₂*, …].

    The **rx** and **ry** options specify the x and y radii respectively. If only
    **r** is specified, it is used for both radii. The optional **rotate** option
    specifies rotation in degrees.

    Additional styling options such as **fill**, **stroke**, and **strokeWidth**
    can be specified to customize the appearance of the ellipses.

    Args:
        values: The data for the ellipses.
        options: Additional options for customizing the ellipses.
        **kwargs: Additional keyword arguments to be merged with options.

    Returns:
        A PlotSpec object representing the ellipse mark.
    """
    return PlotSpec(MarkSpec("ellipse", values, {**options, **kwargs}))


def pixels(
    pixelData: list[float] | JSCode | Any,  # Raw pixel data or JSCode
    *,
    imageWidth: int | JSCode | None = None,
    imageHeight: int | JSCode | None = None,
    x: float | JSCode = 0,
    y: float | JSCode = 0,
    width: float | JSCode | None = None,
    height: float | JSCode | None = None,
) -> PlotSpec:
    """
    A custom mark for efficiently rendering a single image from raw RGB(A) pixel data.
    Unlike most Observable Plot marks which render multiple elements from data arrays,
    this mark renders a single image from a flat array of pixel values.

    Args:
        pixelData: Raw pixel data as a flat array in either RGB format [r,g,b,r,g,b,...]
                  or RGBA format [r,g,b,a,r,g,b,a,...]. Each value should be 0-255.
        imageWidth: Width of the source image in pixels
        imageHeight: Height of the source image in pixels
        x: X coordinate of top-left corner in plot coordinates (default: 0)
        y: Y coordinate of top-left corner in plot coordinates (default: 0)
        width: Displayed width in plot coordinates (defaults to imageWidth)
        height: Displayed height in plot coordinates (defaults to imageHeight)

    Returns:
        A PlotSpec object representing the pixel image mark
    """
    options = {
        "imageWidth": imageWidth,
        "imageHeight": imageHeight,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }
    return PlotSpec(MarkSpec("pixels", pixelData, options))


def events(options: dict[str, Any] = {}, **kwargs) -> PlotSpec:
    """
    Captures events on a plot.

    Args:
        options: Callback functions. Supported: `onClick`, `onMouseMove`, `onMouseDown`, `onDrawStart`, `onDraw`, `onDrawEnd`.
        **kwargs: Additional keyword arguments to be merged with options.

    Each callback receives an event object with:

    - `type`, the event name
    - `x`, the x coordinate
    - `y`, the y coordinate
    - for draw events, `startTime`

    Returns:
        A PlotSpec object representing the events mark.
    """
    return PlotSpec(MarkSpec("events", [], {**options, **kwargs}))


def img(values, options: dict[str, Any] = {}, **kwargs) -> PlotSpec:
    """
    The image mark renders images on the plot. The **src** option specifies the
    image source, while **x**, **y**, **width**, and **height** define the image's
    position and size in the x/y scales. This differs from the built-in Observable Plot
    image mark, which specifies width/height in pixels.

    Args:
        values: The data for the images.
        options: Options for customizing the images.
        **kwargs: Additional keyword arguments to be merged with options.

    Returns:
        A PlotSpec object representing the image mark.

    The following options are supported:
    - `src`: The source path of the image.
    - `x`: The x-coordinate of the top-left corner.
    - `y`: The y-coordinate of the top-left corner.
    - `width`: The width of the image.
    - `height`: The height of the image.
    """
    return PlotSpec(MarkSpec("img", values, {**options, **kwargs}))


htl = JSRef("htl")
"""Hypertext Literal library (https://observablehq.com/@observablehq/htl)"""


__all__ = [
    # ## Custom marks
    "bitmap",
    "canvas_mark",
    "ellipse",
    "histogram",
    "img",
    "pixels",
    # ## Interactivity Utils
    "events",
    "renderChildEvents",
    # ## Utils
    "htl",
]
