"""Post-capture composition that makes screenshots machine-legible.

Agents consume renders resampled to model-native tiles: small text dies and
pixel coordinates get guessed from downscaled views. This module composes
captured PNG bytes — the render path stays untouched and deterministic —
into images an agent can actually read:

* :func:`compose_rulers` expands the canvas with a top/left margin band of
  labeled coordinate rulers (big, high-contrast labels) plus faint
  gridlines, all in the page-CSS-pixel space ``pick-at`` consumes.
* :func:`compose_grid` lays labeled per-view captures out as one contact
  sheet (one image of N views costs an agent fewer tiles than N images).

Composition is pure: same input bytes -> same output bytes (Pillow's text
rasterization and PNG encoder are deterministic for a fixed version, and
the daemon/client version match is already enforced by discovery).
"""

import io
import math
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

# Candidate tick spacings in CSS pixels ("nice" numbers).
_SPACINGS = [10, 20, 25, 50, 100, 200, 250, 500, 1000]
# Aim for at most this many gridlines along the longest edge.
_MAX_TICKS = 10

_BAND_COLOR = (255, 255, 255)
_TEXT_COLOR = (0, 0, 0)
_TICK_COLOR = (0, 0, 0)
_EDGE_COLOR = (120, 120, 120)
_GRID_RGBA = (90, 90, 90, 56)  # faint gridlines over scene content
_TICK_LEN = 8
_PAD = 6


def ruler_spacing(size_css: float) -> int:
    """Tick spacing (CSS px) for an image whose long edge is ``size_css``.

    Picks the smallest "nice" spacing that keeps the long edge at or under
    ``_MAX_TICKS`` gridlines, so labels stay big and uncrowded at any size.
    """
    for spacing in _SPACINGS:
        if size_css / spacing <= _MAX_TICKS:
            return spacing
    return _SPACINGS[-1]


_AnyFont = ImageFont.ImageFont | ImageFont.FreeTypeFont


def _font(size: int) -> _AnyFont:
    """Pillow's bundled scalable font (deterministic for a fixed version)."""
    return ImageFont.load_default(size=size)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: _AnyFont) -> Tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return int(right - left), int(bottom - top)


def compose_rulers(png: bytes, dpr: float = 1.0) -> Tuple[bytes, Dict[str, int]]:
    """Add labeled coordinate rulers (top + left) around a captured PNG.

    The canvas is expanded by a white margin band — scene pixels are never
    covered by the rulers themselves; only faint gridlines cross the
    content. Tick labels are page CSS-pixel coordinates: exactly the values
    ``colight pick-at X,Y`` accepts for the same --width/--height/--dpr
    render. A label ``v`` sits at composed-image pixel ``margin + v*dpr``
    on its axis; equivalently, composed pixel ``p`` maps to page coordinate
    ``(p - margin) / dpr``.

    Args:
        png: The captured PNG bytes (untouched render output).
        dpr: Device pixel ratio the PNG was rendered at.

    Returns:
        Tuple of (composed PNG bytes, ``{"spacing", "margin"}``) — spacing
        in CSS px between ticks, margin in composed-image pixels.
    """
    base = Image.open(io.BytesIO(png)).convert("RGB")
    width_px, height_px = base.size
    width_css = width_px / dpr
    height_css = height_px / dpr
    spacing = ruler_spacing(max(width_css, height_css))

    # Big labels: readable after model-native downsampling. Scale with dpr
    # AND with image size — a 1600px render gets resampled ~2x harder than
    # an 800px one, so its labels must start proportionally bigger.
    scale = max(1.0, dpr, max(width_px, height_px) / 800.0)
    font_size = max(16, int(round(20 * scale)))
    font = _font(font_size)

    probe = ImageDraw.Draw(base)
    max_y_label = int(math.floor(height_css / spacing)) * spacing
    label_w, _ = _text_size(probe, str(max(max_y_label, 0)), font)
    _, label_h = _text_size(probe, "0123456789", font)
    margin = max(label_w, label_h) + _TICK_LEN + 2 * _PAD

    out = Image.new("RGB", (width_px + margin, height_px + margin), _BAND_COLOR)
    out.paste(base, (margin, margin))

    # Faint gridlines over the scene content (RGBA overlay, alpha-blended).
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    grid = ImageDraw.Draw(overlay)
    for tick in range(0, int(width_css) + 1, spacing):
        x = margin + int(round(tick * dpr))
        if x >= out.width:
            break
        grid.line([(x, margin), (x, out.height - 1)], fill=_GRID_RGBA, width=1)
    for tick in range(0, int(height_css) + 1, spacing):
        y = margin + int(round(tick * dpr))
        if y >= out.height:
            break
        grid.line([(margin, y), (out.width - 1, y)], fill=_GRID_RGBA, width=1)
    out = Image.alpha_composite(out.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(out)
    # Band/content separators.
    draw.line([(margin - 1, margin - 1), (out.width - 1, margin - 1)], fill=_EDGE_COLOR)
    draw.line(
        [(margin - 1, margin - 1), (margin - 1, out.height - 1)], fill=_EDGE_COLOR
    )

    # Top ruler: x coordinates.
    for tick in range(0, int(width_css) + 1, spacing):
        x = margin + int(round(tick * dpr))
        if x >= out.width:
            break
        draw.line([(x, margin - _TICK_LEN), (x, margin - 1)], fill=_TICK_COLOR)
        text = str(tick)
        text_w, _ = _text_size(draw, text, font)
        text_x = min(max(x - text_w // 2, margin - _TICK_LEN), out.width - text_w - 1)
        draw.text(
            (text_x, _PAD // 2),
            text,
            fill=_TEXT_COLOR,
            font=font,
        )

    # Left ruler: y coordinates.
    for tick in range(0, int(height_css) + 1, spacing):
        y = margin + int(round(tick * dpr))
        if y >= out.height:
            break
        draw.line([(margin - _TICK_LEN, y), (margin - 1, y)], fill=_TICK_COLOR)
        text = str(tick)
        text_w, text_h = _text_size(draw, text, font)
        text_y = min(max(y - text_h // 2, margin - _TICK_LEN), out.height - text_h - 1)
        draw.text(
            (margin - _TICK_LEN - _PAD - text_w, text_y),
            text,
            fill=_TEXT_COLOR,
            font=font,
        )

    return _encode(out), {"spacing": spacing, "margin": margin}


def compose_grid(cells: Sequence[Tuple[str, bytes]]) -> bytes:
    """Compose labeled per-view captures into one contact-sheet PNG.

    Cells are laid out in a near-square grid (2x2 for four views); each
    cell gets a white label band above it carrying the view name in big
    text — labels never overlay scene pixels. Cell content is pasted at
    native resolution (no resampling).

    Args:
        cells: (view name, PNG bytes) per view, in display order.

    Returns:
        The contact-sheet PNG bytes.
    """
    if not cells:
        raise ValueError("contact sheet needs at least one view")
    images = [Image.open(io.BytesIO(png)).convert("RGB") for _name, png in cells]
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    columns = int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / columns))

    font_size = max(16, cell_w // 20)
    font = _font(font_size)
    probe = ImageDraw.Draw(images[0])
    _, label_h = _text_size(probe, "ABCXYZgy", font)
    band = label_h + 2 * _PAD

    gutter = 2  # thin separator between cells
    total_w = columns * cell_w + (columns + 1) * gutter
    total_h = rows * (cell_h + band) + (rows + 1) * gutter
    out = Image.new("RGB", (total_w, total_h), _EDGE_COLOR)
    draw = ImageDraw.Draw(out)

    for index, ((name, _png), image) in enumerate(zip(cells, images)):
        col = index % columns
        row = index // columns
        x0 = gutter + col * (cell_w + gutter)
        y0 = gutter + row * (cell_h + band + gutter)
        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + band - 1], fill=_BAND_COLOR)
        draw.text((x0 + _PAD, y0 + _PAD), name, fill=_TEXT_COLOR, font=font)
        out.paste(image, (x0, y0 + band))
        # Empty trailing slots stay the background color.

    return _encode(out)


def png_size(png: bytes) -> Tuple[int, int]:
    """(width, height) of PNG bytes."""
    with Image.open(io.BytesIO(png)) as image:
        return image.width, image.height


def _encode(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def grid_cell_count(names: List[str]) -> Tuple[int, int]:
    """(columns, rows) the contact sheet will use for ``names``."""
    columns = int(math.ceil(math.sqrt(len(names))))
    rows = int(math.ceil(len(names) / columns))
    return columns, rows


__all__ = [
    "compose_grid",
    "compose_rulers",
    "grid_cell_count",
    "png_size",
    "ruler_spacing",
]
