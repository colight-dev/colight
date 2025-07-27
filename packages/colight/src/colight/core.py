import pathlib
from typing import Any, Optional, Union

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


bylight = Bylight = JSRef("Bylight")
"""Creates a highlighted code block using the [Bylight library](https://mhuebert.github.io/bylight/).

Args:
    source (str): The source text/code to highlight
    patterns (list): A list of patterns to highlight. Each pattern can be either:
        - A string to match literally
        - A dict with 'match' (required) and 'color' (optional) keys
    props (dict, optional): Additional properties to pass to the pre element. Defaults to {}.

Example:
    ```python
    Plot.bylight('''
        def hello():
            print("Hello World!")
    ''', ["def", "print"])
    ```

Returns:
    A Bylight component that renders the highlighted code block.
"""

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


__all__ = [
    # ## Interactivity
    "State",
    "onChange",
    "Frames",
    "Slider",
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
