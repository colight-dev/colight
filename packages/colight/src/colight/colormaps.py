"""First-class colormaps: scalar/categorical values -> RGB, plus legend specs.

Pure numpy — no matplotlib dependency. Continuous maps are piecewise-linear
ramps through anchor colors sampled from the standard matplotlib definitions
(17 evenly spaced anchors each, accurate to ~1e-3 against the 256-entry
originals); categorical palettes are fixed swatch lists.

Two halves, one source of truth:

- :func:`apply_colormap` turns values into per-instance float32 RGB colors
  (the arrays the GPU renders).
- :func:`colormap_metadata` builds the JSON-safe spec ({cmap, domain, label,
  stops/colors, ...}) that travels with a component so legends can be drawn
  and machine consumers (``colight inspect`` / ``screenshot --json``) can
  report what the colors encode.

``scene3d`` primitives accept a ``color_by={"values", "cmap", "domain",
"label"}`` prop that routes through both.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

DEFAULT_NAN_COLOR: Tuple[float, float, float] = (0.5, 0.5, 0.5)

# Continuous ramps: 17 evenly spaced RGB anchors over [0, 1], sampled from
# the matplotlib colormaps of the same name (endpoints exact).
CONTINUOUS_CMAPS: Dict[str, np.ndarray] = {
    "viridis": np.array(
        [
            [0.267004, 0.004874, 0.329415],
            [0.282327, 0.094955, 0.417331],
            [0.278826, 0.175490, 0.483397],
            [0.258965, 0.251537, 0.524736],
            [0.229739, 0.322361, 0.545706],
            [0.199430, 0.387607, 0.554642],
            [0.172719, 0.448791, 0.557885],
            [0.149039, 0.508051, 0.557250],
            [0.127568, 0.566949, 0.550556],
            [0.120638, 0.625828, 0.533488],
            [0.157851, 0.683765, 0.501686],
            [0.246070, 0.738910, 0.452024],
            [0.369214, 0.788888, 0.382914],
            [0.515992, 0.831158, 0.294279],
            [0.678489, 0.863742, 0.189503],
            [0.845561, 0.887322, 0.099702],
            [0.993248, 0.906157, 0.143936],
        ]
    ),
    "magma": np.array(
        [
            [0.001462, 0.000466, 0.013866],
            [0.039608, 0.031090, 0.133515],
            [0.113094, 0.065492, 0.276784],
            [0.211718, 0.061992, 0.418647],
            [0.316654, 0.071690, 0.485380],
            [0.414709, 0.110431, 0.504662],
            [0.512831, 0.148179, 0.507648],
            [0.613617, 0.181811, 0.498536],
            [0.716387, 0.214982, 0.475290],
            [0.816914, 0.255895, 0.436461],
            [0.904281, 0.319610, 0.388137],
            [0.960949, 0.418323, 0.359630],
            [0.986700, 0.535582, 0.382210],
            [0.996096, 0.653659, 0.446213],
            [0.996898, 0.769591, 0.534892],
            [0.992440, 0.884330, 0.640099],
            [0.987053, 0.991438, 0.749504],
        ]
    ),
    "plasma": np.array(
        [
            [0.050383, 0.029803, 0.527975],
            [0.193374, 0.018354, 0.590330],
            [0.299855, 0.009561, 0.631624],
            [0.399411, 0.000859, 0.656133],
            [0.494877, 0.011990, 0.657865],
            [0.584391, 0.068579, 0.632812],
            [0.665129, 0.138566, 0.585582],
            [0.736019, 0.209439, 0.527908],
            [0.798216, 0.280197, 0.469538],
            [0.853319, 0.351553, 0.413734],
            [0.901807, 0.425087, 0.359688],
            [0.942598, 0.502639, 0.305816],
            [0.973416, 0.585761, 0.251540],
            [0.991365, 0.675355, 0.198453],
            [0.993033, 0.771720, 0.154808],
            [0.974443, 0.874622, 0.144061],
            [0.940015, 0.975158, 0.131326],
        ]
    ),
    # Diverging blue -> white -> red.
    "coolwarm": np.array(
        [
            [0.229806, 0.298718, 0.753683],
            [0.304174, 0.406945, 0.845263],
            [0.383662, 0.510183, 0.917831],
            [0.467678, 0.605591, 0.968546],
            [0.554312, 0.690097, 0.995516],
            [0.640828, 0.760752, 0.997846],
            [0.724041, 0.814910, 0.975651],
            [0.800601, 0.850358, 0.930008],
            [0.867428, 0.864377, 0.862602],
            [0.925563, 0.825517, 0.771136],
            [0.959518, 0.766973, 0.674145],
            [0.969683, 0.690484, 0.575138],
            [0.956653, 0.598034, 0.477302],
            [0.921406, 0.491420, 0.383408],
            [0.865391, 0.371128, 0.295769],
            [0.790562, 0.231397, 0.216242],
            [0.705673, 0.015556, 0.150233],
        ]
    ),
    # White -> black (matplotlib "Greys").
    "greys": np.array(
        [
            [1.000000, 1.000000, 1.000000],
            [0.970473, 0.970473, 0.970473],
            [0.940823, 0.940823, 0.940823],
            [0.895548, 0.895548, 0.895548],
            [0.850119, 0.850119, 0.850119],
            [0.795002, 0.795002, 0.795002],
            [0.739377, 0.739377, 0.739377],
            [0.662607, 0.662607, 0.662607],
            [0.586082, 0.586082, 0.586082],
            [0.517186, 0.517186, 0.517186],
            [0.448443, 0.448443, 0.448443],
            [0.383483, 0.383483, 0.383483],
            [0.317416, 0.317416, 0.317416],
            [0.228835, 0.228835, 0.228835],
            [0.141115, 0.141115, 0.141115],
            [0.068281, 0.068281, 0.068281],
            [0.000000, 0.000000, 0.000000],
        ]
    ),
}

# Categorical palettes: value i maps to swatch i (mod palette length).
CATEGORICAL_CMAPS: Dict[str, np.ndarray] = {
    # The d3/matplotlib tab10 palette.
    "tab10": np.array(
        [
            [0.121569, 0.466667, 0.705882],  # blue
            [1.000000, 0.498039, 0.054902],  # orange
            [0.172549, 0.627451, 0.172549],  # green
            [0.839216, 0.152941, 0.156863],  # red
            [0.580392, 0.403922, 0.741176],  # purple
            [0.549020, 0.337255, 0.294118],  # brown
            [0.890196, 0.466667, 0.760784],  # pink
            [0.498039, 0.498039, 0.498039],  # grey
            [0.737255, 0.741176, 0.133333],  # olive
            [0.090196, 0.745098, 0.811765],  # cyan
        ]
    ),
    # Okabe-Ito colorblind-safe palette (without black).
    "okabe_ito": np.array(
        [
            [0.901961, 0.623529, 0.000000],  # orange
            [0.337255, 0.705882, 0.913725],  # sky blue
            [0.000000, 0.619608, 0.450980],  # bluish green
            [0.941176, 0.894118, 0.258824],  # yellow
            [0.000000, 0.447059, 0.698039],  # blue
            [0.835294, 0.368627, 0.000000],  # vermillion
            [0.800000, 0.474510, 0.654902],  # reddish purple
        ]
    ),
}


def is_categorical(cmap: str) -> bool:
    """Whether ``cmap`` names a categorical palette (vs a continuous ramp).

    Raises:
        ValueError: Unknown colormap name.
    """
    name = cmap.lower()
    if name in CATEGORICAL_CMAPS:
        return True
    if name in CONTINUOUS_CMAPS:
        return False
    known = ", ".join(sorted([*CONTINUOUS_CMAPS, *CATEGORICAL_CMAPS]))
    raise ValueError(f"unknown colormap {cmap!r} (available: {known})")


def colormap_stops(cmap: str) -> np.ndarray:
    """The anchor colors of a colormap as a (K, 3) float64 array in [0, 1]."""
    name = cmap.lower()
    if is_categorical(name):
        return CATEGORICAL_CMAPS[name]
    return CONTINUOUS_CMAPS[name]


def resolve_domain(
    values: np.ndarray, domain: Optional[Sequence[float]] = None
) -> Tuple[float, float]:
    """Resolve the value range a continuous colormap spans.

    Args:
        values: Scalar values (NaNs ignored).
        domain: Explicit (min, max); None derives it from finite values.

    Returns:
        (lo, hi) floats. Falls back to (0, 1) when no finite values exist.

    Raises:
        ValueError: When an explicit domain is malformed (hi <= lo).
    """
    if domain is not None:
        lo, hi = float(domain[0]), float(domain[1])
        if not (hi > lo):
            raise ValueError(f"colormap domain must have max > min, got {domain}")
        return lo, hi
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def apply_colormap(
    values: Any,
    cmap: str = "viridis",
    domain: Optional[Sequence[float]] = None,
    nan_color: Sequence[float] = DEFAULT_NAN_COLOR,
) -> np.ndarray:
    """Map scalar or category values to RGB colors.

    Continuous maps clamp values to ``domain`` and interpolate linearly
    through the ramp's anchor colors. Categorical palettes index swatches by
    integer value (modulo palette length).

    Args:
        values: 1D array of values. Continuous: floats, NaNs allowed.
            Categorical: non-negative integer codes (floats are floored;
            NaN/negative codes get ``nan_color``).
        cmap: Colormap name (see ``CONTINUOUS_CMAPS`` / ``CATEGORICAL_CMAPS``).
        domain: Continuous only — (min, max) the ramp spans; None derives it
            from the finite values.
        nan_color: RGB assigned to NaN (and invalid categorical) values.

    Returns:
        (N, 3) float32 array of RGB colors in [0, 1].

    Raises:
        ValueError: Unknown colormap, malformed domain, or non-1D values.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"colormap values must be 1D, got shape {array.shape}")
    nan_rgb = np.asarray(nan_color, dtype=np.float64)

    if is_categorical(cmap):
        palette = CATEGORICAL_CMAPS[cmap.lower()]
        invalid = ~np.isfinite(array) | (array < 0)
        codes = np.where(invalid, 0, array).astype(np.int64) % len(palette)
        rgb = palette[codes]
        rgb = np.where(invalid[:, None], nan_rgb, rgb)
        return rgb.astype(np.float32)

    ramp = CONTINUOUS_CMAPS[cmap.lower()]
    lo, hi = resolve_domain(array, domain)
    nan_mask = ~np.isfinite(array)
    t = np.clip((np.where(nan_mask, lo, array) - lo) / (hi - lo), 0.0, 1.0)
    xs = np.linspace(0.0, 1.0, len(ramp))
    rgb = np.stack([np.interp(t, xs, ramp[:, c]) for c in range(3)], axis=-1)
    rgb[nan_mask] = nan_rgb
    return rgb.astype(np.float32)


def colormap_metadata(
    cmap: str,
    domain: Optional[Sequence[float]] = None,
    label: Optional[str] = None,
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the JSON-safe colormap spec that travels with a component.

    This single payload drives both the rendered legend (JS reads ``stops``
    or ``colors`` to draw the gradient/swatches) and machine-readable
    reporting (``colight inspect`` / ``screenshot --json``).

    Args:
        cmap: Colormap name.
        domain: Continuous only — resolved (min, max) the colors span.
        label: What the colors encode (e.g. an attribute name).
        categories: Categorical only — display names for codes 0..K-1.

    Returns:
        Dict with ``cmap``, ``categorical``, and either ``domain`` +
        ``stops`` (continuous) or ``colors`` (categorical, one RGB per
        category — palette order, cycled to cover ``categories`` when
        given); ``label``/``categories`` included when provided.

    Raises:
        ValueError: Unknown colormap, or ``categories``/``domain`` given for
            the wrong kind of colormap.
    """
    name = cmap.lower()
    categorical = is_categorical(name)
    meta: Dict[str, Any] = {"cmap": name, "categorical": categorical}
    if label is not None:
        meta["label"] = str(label)
    if categorical:
        if domain is not None:
            raise ValueError("domain applies to continuous colormaps only")
        palette = CATEGORICAL_CMAPS[name]
        count = len(categories) if categories else len(palette)
        indices = np.arange(count) % len(palette)
        meta["colors"] = [[round(float(c), 6) for c in palette[i]] for i in indices]
        if categories is not None:
            meta["categories"] = [str(c) for c in categories]
    else:
        if categories is not None:
            raise ValueError("categories apply to categorical colormaps only")
        if domain is not None:
            lo, hi = float(domain[0]), float(domain[1])
            meta["domain"] = [lo, hi]
        meta["stops"] = [
            [round(float(c), 6) for c in row] for row in CONTINUOUS_CMAPS[name]
        ]
    return meta


ColorByInput = Union[Dict[str, Any], "Any"]


def resolve_color_by(color_by: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Resolve a ``color_by`` spec into (colors, legend metadata).

    Args:
        color_by: Dict with ``values`` (required, 1D scalars or category
            codes) and optional ``cmap`` (default "viridis"), ``domain``,
            ``label``, ``categories``, ``nan_color``.

    Returns:
        Tuple of ((N, 3) float32 RGB colors, metadata dict from
        :func:`colormap_metadata` with the resolved domain filled in).

    Raises:
        ValueError: Missing ``values`` or unknown keys/colormap.
    """
    spec = dict(color_by)
    if "values" not in spec:
        raise ValueError("color_by requires a 'values' array")
    values = np.asarray(spec.pop("values"), dtype=np.float64)
    cmap = str(spec.pop("cmap", "viridis"))
    domain = spec.pop("domain", None)
    label = spec.pop("label", None)
    categories = spec.pop("categories", None)
    nan_color = spec.pop("nan_color", DEFAULT_NAN_COLOR)
    if spec:
        raise ValueError(
            f"unknown color_by keys: {sorted(spec)} "
            "(accepted: values, cmap, domain, label, categories, nan_color)"
        )
    colors = apply_colormap(values, cmap=cmap, domain=domain, nan_color=nan_color)
    if not is_categorical(cmap):
        domain = resolve_domain(values, domain)
    meta = colormap_metadata(cmap, domain=domain, label=label, categories=categories)
    return colors, meta


__all__ = [
    "CATEGORICAL_CMAPS",
    "CONTINUOUS_CMAPS",
    "DEFAULT_NAN_COLOR",
    "apply_colormap",
    "colormap_metadata",
    "colormap_stops",
    "is_categorical",
    "resolve_color_by",
    "resolve_domain",
]
