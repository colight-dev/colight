import pathlib
import re
from typing import Any, Optional, Sequence, Union

import numpy as np

from colight.components.slider import Slider
from colight.layout import (
    Column,
    Grid,
    Hiccup,
    JSCall,
    JSCode,
    JSRef,
    LayoutItem,
    Ref,
    Row,
    State,
    js,
    onChange,
    ref,
)
from colight.plot_spec import new
from colight.protocols import Collector

html = Hiccup
new = new


def cond(*pairs: Union[JSCode, str, list, Any]) -> JSCall:
    """Render content based on conditions, like Clojure's cond.

    Takes pairs of test/expression arguments, evaluating each test in order.
    When a test is truthy, returns its corresponding expression.
    An optional final argument serves as the "else" expression.

    Args:
        *args: Alternating test/expression pairs, with optional final else expression

    Example:
        Plot.cond(
            Plot.js("$state.detail == 'a'"), ["div", "Details for A"],
            Plot.js("$state.detail == 'b'"), ["div", "Details for B"],
            "No details selected"  # else case
        )

        # Without else:
        Plot.cond(
            Plot.js("$state.detail"), ["div", Plot.js("$state.detail")]
        )
    """
    return JSCall("COND", pairs)


def case(value: Union[JSCode, str, Any], *pairs: Union[str, list, Any]) -> JSCall:
    """Render content based on matching a value against cases, like a switch statement.

    Takes a value to match against, followed by pairs of case/expression arguments.
    When a case matches the value, returns its corresponding expression.
    An optional final argument serves as the default expression.

    Args:
        value: The value to match against cases
        *args: Alternating case/expression pairs, with optional final default expression

    Example:
        Plot.case(Plot.js("$state.selected"),
            "a", ["div", "Selected A"],
            "b", ["div", "Selected B"],
            ["div", "Nothing selected"]  # default case
        )
    """
    return JSCall("CASE", [value, *pairs])


class Import(LayoutItem, Collector):
    """Import JavaScript code into the Colight environment.

    Args:
        source: JavaScript source code. Can be:
            - Inline JavaScript code
            - URL starting with http(s):// for remote modules
            - Local file path starting with path: prefix
        alias: Namespace alias for the entire module
        default: Name for the default export
        refer: Set of names to import directly, or True to import all
        refer_all: Alternative to refer=True
        rename: Dict of original->new names for referred imports
        exclude: Set of names to exclude when using refer_all
        format: Module format ('esm' or 'commonjs')

    Imported JavaScript code can access:
    - `colight.imports`: Previous imports in the current plot (only for CommonJS imports)
    - `React`, `d3`, `html` (for hiccup) and `colight.api` are defined globally

    Examples:
    ```python
    # CDN import with namespace alias
    Plot.Import(
        source="https://cdn.skypack.dev/lodash-es",
        alias="_",
        refer=["flattenDeep", "partition"],
        rename={"flattenDeep": "deepFlatten"}
    )

    # Local file import
    Plot.Import(
        source="path:src/app/utils.js",  # relative to working directory
        refer=["formatDate"]
    )

    # Inline source with refer_all
    Plot.Import(
        source='''
        export const add = (a, b) => a + b;
        export const subtract = (a, b) => a - b;
        ''',
        refer_all=True,
        exclude=["subtract"]
    )

    # Default export handling
    Plot.Import(
        source="https://cdn.skypack.dev/d3-scale",
        default="createScale"
    )
    ```
    """

    def __init__(
        self,
        source: str,
        alias: Optional[str] = None,
        default: Optional[str] = None,
        refer: Optional[list[str]] = None,
        refer_all: bool = False,
        rename: Optional[dict[str, str]] = None,
        exclude: Optional[list[str]] = None,
        format: str = "esm",
    ):
        super().__init__()

        # Create spec for the import
        spec: dict[str, Union[str, list[str], bool, dict[str, str]]] = {
            "format": format
        }

        # Handle source based on prefix
        if source.startswith("path:"):
            path = source[5:]  # Remove 'path:' prefix
            try:
                resolved_path = pathlib.Path.cwd() / path
                with open(resolved_path) as f:
                    spec["source"] = f.read()
            except Exception as e:
                raise ValueError(f"Failed to load file at {path}: {e}")
        else:
            spec["source"] = source

        if alias:
            spec["alias"] = alias
        if default:
            spec["default"] = str(default)
        if refer:
            spec["refer"] = refer
        if refer_all:
            spec["refer_all"] = True
        if rename:
            spec["rename"] = rename
        if exclude:
            spec["exclude"] = exclude

        # Store as a list of specs instead of dict
        self._state_imports = [spec]

    def for_json(self):
        return None

    def collect(self, collector):
        """Collect imports and disappear from output."""
        for spec in self._state_imports:
            collector.add_import(spec)
        return None


_Frames = JSRef("Frames")


def Frames(
    frames: list[Any] | Ref,
    key: str | None = None,
    slider: bool = True,
    tail: bool = False,
    **opts: Any,
) -> LayoutItem:
    """
    Create an animated plot that cycles through a list of frames.

    Args:
        frames (list): A list of plot specifications or renderable objects to animate.
        key (str | None): The state key to use for the frame index. If None, uses "frame".
        slider (bool): Whether to show the slider control. Defaults to True.
        tail (bool): Whether animation should stop at the end. Defaults to False.
        **opts: Additional options for the animation, such as fps (frames per second).

    Returns:
        A Hiccup-style representation of the animated plot.
    """
    frames = ref(frames)
    if key is None:
        key = "frame"
        return Hiccup([_Frames, {"state_key": key, "frames": frames}]) | Slider(
            key,
            rangeFrom=frames,
            tail=tail,
            visible=slider,
            **opts,
        )
    else:
        return Hiccup([_Frames, {"state_key": key, "frames": frames}])


initial_state = initialState = state = State


def md(text: str, **kwargs: Any) -> JSCall:
    """Render a string as Markdown, in a LayoutItem."""
    return JSRef("md")(kwargs, text)


katex = JSRef("katex")
"""Render a TeX string, in a LayoutItem."""


CHANNEL_RULES = ("nearest", "step", "linear", "qlerp")
"""The resampling rules a `channel` may declare."""

_PARAMETER_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def channel(
    parameter: str,
    values: Any,
    at: Optional[Sequence[float]] = None,
    rule: str = "linear",
) -> JSCall:
    """Declare a prop whose value is resampled client-side from shipped samples.

    The `values` rows travel once, as one array. A `$state` scalar named
    `parameter` indexes them: whenever it changes, the browser resamples row
    `at[i]`..`at[i+1]` under `rule` and hands the result to the prop, with no
    Python round trip. The declaration is legible to `colight inspect`, which
    reports the parameter, its domain, the rule and the prop it drives.

    A channel carries no notion of time: `parameter="t"` is a clock,
    `parameter="grade"` a cutoff, `parameter="blend"` a morph.

    Args:
        parameter: The `$state` key this channel is indexed by.
        values: Array-like of shape (N, ...) — row `i` is the value at `at[i]`.
            Scalars per row (N,), vectors (N, 3)/(N, 4), and wide rows (N, M)
            (e.g. a table of flattened vertex positions) are all accepted.
        at: (N,) strictly increasing sample coordinates. Defaults to
            `arange(N)`.
        rule: How values between samples are produced — `"nearest"`, `"step"`
            (hold the lower sample), `"linear"` (elementwise lerp) or
            `"qlerp"` (normalized quaternion lerp with antipodal correction,
            for (N, 4) xyzw rows).

    Returns:
        A `JSCall` to the client-side resampler, usable anywhere a prop value
        is accepted.

    Raises:
        ValueError: On an unknown rule, a malformed parameter name, an empty
            or mismatched `at`, non-increasing `at`, or `qlerp` on values that
            are not (N, 4).
    """
    if not isinstance(parameter, str) or not _PARAMETER_NAME_RE.match(parameter):
        raise ValueError(
            f"channel parameter must be an identifier-like $state key, got {parameter!r}"
        )
    if rule not in CHANNEL_RULES:
        raise ValueError(
            f"channel rule must be one of {list(CHANNEL_RULES)}, got {rule!r}"
        )

    values_array = np.asarray(values)
    if values_array.ndim == 0:
        raise ValueError("channel values must have shape (N, ...), got a scalar")
    if values_array.dtype.kind not in "fiu":
        raise ValueError(
            f"channel values must be numeric, got dtype {values_array.dtype}"
        )
    if values_array.dtype.kind != "f":
        values_array = values_array.astype(np.float32)
    n = int(values_array.shape[0])
    if n < 1:
        raise ValueError("channel values must have at least one row")

    if at is None:
        at_array = np.arange(n, dtype=np.float64)
    else:
        at_array = np.asarray(at, dtype=np.float64)
        if at_array.ndim != 1:
            raise ValueError(f"channel `at` must be 1-D, got shape {at_array.shape}")
        if at_array.shape[0] != n:
            raise ValueError(
                f"channel `at` has {at_array.shape[0]} coordinates but values has "
                f"{n} rows"
            )
        if n > 1 and not np.all(np.diff(at_array) > 0):
            raise ValueError("channel `at` must be strictly increasing")

    if rule == "qlerp" and (values_array.ndim != 2 or values_array.shape[1] != 4):
        raise ValueError(
            f"channel rule 'qlerp' requires values of shape (N, 4) xyzw quaternions, "
            f"got {tuple(values_array.shape)}"
        )

    return JSCall(
        "colight.resampleChannel",
        [
            {
                "parameter": parameter,
                "value": js(f'$state["{parameter}"]'),
                "at": at_array,
                "values": values_array,
                "rule": rule,
            }
        ],
    )


__all__ = [
    # ## Interactivity
    "State",
    "onChange",
    "Frames",
    "Slider",
    "channel",
    # ## Layout
    # Useful for layouts and custom views.
    # Note that syntax sugar exists for `Column` (`|`) and `Row` (`&`) using operator overloading.
    # ```
    # (A & B) | C # A & B on one row, with C below.
    # ```
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
]
