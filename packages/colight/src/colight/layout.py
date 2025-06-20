import os
import uuid
from pathlib import Path
from typing import Any, List, Optional, Self, Tuple, Union, cast

import colight.format as format
import colight.screenshots as screenshots
from colight.env import CONFIG
from colight.html import html_page, html_snippet
from colight.widget import Widget, WidgetState, to_json_with_initialState


def create_parent_dir(path: str) -> None:
    """Create parent directory if it doesn't exist."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


class HTML:
    def __init__(self, ast):
        self.ast = ast
        self.id = f"colight-widget-{uuid.uuid4().hex}"

    def set_ast(self, ast):
        self.ast = ast

    def _repr_mimebundle_(self, **kwargs):
        html_content = html_snippet(self.ast, self.id)
        return {"text/html": html_content}, {}


class LayoutItem:
    def __init__(self):
        self._html: HTML | None = None
        self._widget: Widget | None = None
        self._display_as: str | None = None

    def display_as(self, display_as) -> Self:
        if display_as not in ["html", "widget"]:
            raise ValueError("display_pref must be either 'html' or 'widget'")
        self._display_as = display_as
        return self

    def for_json(self) -> dict[str, Any] | None:
        raise NotImplementedError("Subclasses must implement for_json method")

    def __and__(self, other: Any) -> "Row":
        if isinstance(self, Row):
            return Row(*self.items, other)
        if isinstance(other, Row):
            return Row(self, *other.items)
        return Row(self, other)

    def __rand__(self, other: Any) -> "Row":
        if isinstance(self, Row):
            return Row(other, *self.items)
        return Row(other, self)

    def __or__(self, other: Any) -> "Column":
        if isinstance(self, Column):
            return Column(*self.items, other)
        if isinstance(other, Column):
            return Column(self, *other.items)
        return Column(self, other)

    def __ror__(self, other: Any) -> "Column":
        if isinstance(self, Column):
            return Column(other, *self.items)
        return Column(other, self)

    def _repr_mimebundle_(self, **kwargs: Any) -> Any:
        return self.repr()._repr_mimebundle_(**kwargs)

    def _repr_html_(self, **kwargs: Any) -> str | None:
        bundle = self.repr()._repr_mimebundle_(**kwargs)
        if (
            isinstance(bundle, tuple)
            and len(bundle) > 0
            and isinstance(bundle[0], dict)
        ):
            return bundle[0].get("text/html")
        return None

    def html(self) -> HTML:
        """
        Lazily generate & cache the HTML for this LayoutItem.
        """
        if self._html is None:
            self._html = HTML(self.for_json())
        return self._html

    def widget(self) -> Widget:
        """
        Lazily generate & cache the widget for this LayoutItem.
        """
        if self._widget is None:
            self._widget = Widget(self)
        return cast(Widget, self._widget)

    def repr(self) -> Widget | HTML:
        display_as = self._display_as or CONFIG["display_as"]
        if display_as == "widget":
            return self.widget()
        else:
            return self.html()

    def save_html(self, path: str, dist_url=None) -> str:
        create_parent_dir(path)
        with open(path, "w") as f:
            f.write(html_page(self.for_json(), dist_url=dist_url))
        print(f"HTML saved to {path}")
        return str(path)

    def save_file(self, path: str) -> str:
        data, buffers = to_json_with_initialState(self, buffers=[])
        return format.create_file(data, buffers, path)

    def save_image(
        self, path, width=500, height=None, scale: float = 1.0, quality=90, debug=False
    ):
        """Save the plot as an image using headless browser.

        Args:
            path: Path to save the image to. Format is inferred from file extension (.png or .webp)
            width: Width of the image in pixels (default: 500)
            height: Optional height of the image in pixels
            scale: Scale factor for rendering (default: 1.0)
            quality: Image quality for WebP format (0-100, ignored for PNG, default: 90)
            debug: Whether to print debug information
        """
        screenshots.save_image(
            self,
            path,
            width=width,
            height=height,
            scale=scale,
            quality=quality,
            debug=debug,
        )
        print(f"Image saved to {path}")

    def save_pdf(self, path, width=500, height=None, scale: float = 1.0, debug=False):
        """Save the plot as a PDF using headless Chrome."""

        create_parent_dir(path)
        screenshots.save_pdf(
            self, path, width=width, height=height, scale=scale, debug=debug
        )
        print(f"PDF saved to {path}")

    def save_images(
        self,
        state_updates,
        output_dir: Union[str, Path] = "./scratch/screenshots",
        filenames=None,
        filename_base="screenshot",
        width=500,
        height=None,
        quality=90,
        debug=False,
    ):
        """Save a sequence of images for different states of the plot.

        Args:
            state_updates: List of state updates to apply before each screenshot
            output_dir: Directory to save screenshots (default: "./scratch/screenshots")
            filenames: Optional list of filenames for each screenshot. Must match length of state_updates
            filename_base: Base name for auto-generating filenames if filenames not provided
            width: Width of the images in pixels (default: 500)
            height: Optional height of the images in pixels
            quality: Image quality for WebP format (0-100, ignored for PNG, default: 90)
            debug: Whether to print debug information

        Returns:
            List of paths to saved screenshots
        """

        return screenshots.save_images(
            self,
            state_updates,
            output_dir=output_dir,
            filenames=filenames,
            filename_base=filename_base,
            width=width,
            height=height,
            quality=quality,
            debug=debug,
        )

    def save_video(
        self,
        state_updates,
        filename,
        fps=24,
        width=500,
        height=None,
        scale=2.0,
        debug=False,
    ):
        """Save a sequence of states as a video.

        Args:
            state_updates: List of state updates to apply sequentially
            filename: Path where the resulting video will be saved. Use .gif extension to save as GIF, otherwise saves as MP4
            fps: Frame rate (frames per second) for the video (default: 24)
            width: Width of the video in pixels (default: 500)
            height: Optional height of the video in pixels
            scale: Scale factor for rendering (default: 2.0)
            debug: Whether to print debug information

        Returns:
            Path to the saved video file (.mp4 or .gif)
        """
        return screenshots.save_video(
            self,
            state_updates,
            filename,
            fps=fps,
            width=width,
            height=height,
            scale=scale,
            debug=debug,
        )

    def reset(self, other: "LayoutItem") -> None:
        """
        Render a new LayoutItem to this LayoutItem's widget.

        Args:
            new_item: A LayoutItem to reset to.
        """
        ensure_widget(self).set_ast(other.for_json())

    @property
    def state(self) -> WidgetState:
        """
        Get the widget state. Raises ValueError if widget is not initialized.
        """
        return ensure_widget(self).state


def ensure_widget(self: LayoutItem) -> Widget:
    if self._html is not None:
        raise ValueError(
            "Cannot reset an HTML widget. Use display_as='widget' or foo.widget() to create a resettable widget."
        )
    return self.widget()


class JSCall(LayoutItem):
    """Represents a JavaScript function call."""

    def __init__(self, path: str, args: Union[List[Any], Tuple[Any, ...]] = []):
        super().__init__()
        self.path = path
        self.args = args

    def for_json(self) -> dict:
        return {
            "__type__": "function",
            "path": self.path,
            "args": self.args,
        }


class JSRef(LayoutItem):
    """Refers to a JavaScript module or name. When called, returns a function call representation."""

    def __init__(
        self,
        path: str,
        label: Optional[str] = None,
        doc: Optional[str] = None,
    ):
        super().__init__()
        self.path = path
        self.__name__ = label or path.split(".")[-1]
        self.__doc__ = doc

    def __call__(self, *args: Any) -> Any:
        """Invokes the wrapped JavaScript function in the runtime with the provided arguments."""
        return JSCall(self.path, args)

    def __getattr__(self, name: str) -> "JSRef":
        """Returns a reference to a nested property or method of the JavaScript object."""
        if name.startswith("_state_"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute {name}"
            )
        return JSRef(f"{self.path}.{name}")

    def for_json(self) -> dict:
        return {"__type__": "js_ref", "path": self.path}


def js_ref(path: str) -> "JSRef":
    """Represents a reference to a JavaScript module or name."""
    return JSRef(path=path)


class JSCode(LayoutItem):
    """Represents raw JavaScript code to be evaluated."""

    def __init__(self, code: str, *params: Any, expression: bool):
        super().__init__()
        self.code = code
        self.params = params
        self.expression = expression

    def for_json(self) -> dict:
        return {
            "__type__": "js_source",
            "value": self.code,
            "params": self.params,
            "expression": self.expression,
        }


JSExpr = Union[JSCall, JSRef, JSCode]
"""A type alias representing JavaScript expressions that can be evaluated in the runtime."""


def is_js_expr(obj: Any) -> bool:
    """Check if an object is a JavaScript expression (JSCall, JSRef, or JSCode).

    Args:
        obj: Object to check

    Returns:
        bool: True if object is a JSExpr type, False otherwise
    """
    return isinstance(obj, (JSCall, JSRef, JSCode))


def js(txt: str, *params: Any, expression=True) -> JSCode:
    """Represents raw JavaScript code to be evaluated as a LayoutItem.

    The code will be evaluated in a scope that includes:
    - $state: Current plot state
    - html: render HTML using a JavaScript hiccup syntax
    - d3: D3.js library
    - colight.api: roughly, the api exposed via the colight.plot module

    Args:
        txt (str): JavaScript code with optional %1, %2, etc. placeholders
        *params: Values to substitute for %1, %2, etc. placeholders
        expression (bool): Whether to evaluate as expression or statement
    """
    return JSCode(txt, *params, expression=expression)


class Hiccup(LayoutItem):
    """Wraps a Hiccup-style list to be rendered as an interactive widget in the JavaScript runtime."""

    def __init__(self, *hiccup_elements) -> None:
        LayoutItem.__init__(self)
        self.hiccup_element = (
            hiccup_elements[0]
            if len(hiccup_elements) == 1
            else ["<>", *hiccup_elements]
        )

    def for_json(self) -> Any:
        return self.hiccup_element


_Row = JSRef("Row")


class Row(LayoutItem):
    """Render children in a row.

    Args:
        *items: Items to render in the row
        **kwargs: Additional options including:
            widths: List of flex sizes for each child. Can be:
                - Numbers for flex ratios (e.g. [1, 2] means second item is twice as wide)
                - Strings with fractions (e.g. ["1/2", "1/2"] for equal halves)
                - Strings with explicit sizes (e.g. ["100px", "200px"])
            gap: Gap size between items (default: 1)
            className: Additional CSS classes
    """

    def __init__(self, *items: Any, **kwargs):
        super().__init__()
        self.items, self.options = items, kwargs

    def for_json(self) -> Any:
        return Hiccup([_Row, self.options, *self.items])


_Column = JSRef("Column")


class Column(LayoutItem):
    """Render children in a column.

    Args:
        *items: Items to render in the column
        **kwargs: Additional options including:
            heights: List of flex sizes for each child. Can be:
                - Numbers for flex ratios (e.g. [1, 2] means second item is twice as tall)
                - Strings with fractions (e.g. ["1/2", "1/2"] for equal halves)
                - Strings with explicit sizes (e.g. ["100px", "200px"])
            gap: Gap size between items (default: 1)
            className: Additional CSS classes
    """

    def __init__(self, *items: Any, **kwargs):
        super().__init__()
        self.items, self.options = items, kwargs

    def for_json(self) -> Any:
        return Hiccup([_Column, self.options, *self.items])


def unwrap_for_json(x):
    while hasattr(x, "for_json"):
        x = x.for_json()
    return x


class Listener(LayoutItem):
    def __init__(self, listeners: dict):
        self._state_listeners = listeners

    def for_json(self):
        return None


def onChange(callbacks):
    """
    Adds callbacks to be invoked when state changes.

    Args:
        callbacks (dict): A dictionary mapping state keys to callbacks, which are called with (widget, event) when the corresponding state changes.

    Returns:
        Listener: A Listener object that will be rendered to set up the event handlers.

    Example:
        >>> Plot.onChange({
        ...     "x": lambda w, e: print(f"x changed to {e}"),
        ...     "y": lambda w, e: print(f"y changed to {e}")
        ... })
    """
    return Listener(callbacks)


class Ref(LayoutItem):
    def __init__(self, value, state_key=None, sync=False):
        self._state_key = str(uuid.uuid1()) if state_key is None else state_key
        self._state_sync = sync
        self.value = value

    def for_json(self):
        return unwrap_for_json(self.value)

    def _repr_mimebundle_(self, **kwargs: Any) -> Any:
        if hasattr(self.value, "_repr_mimebundle_"):
            return self.value._repr_mimebundle_(**kwargs)
        return super()._repr_mimebundle_(**kwargs)


def ref(value: Any, state_key=None, sync=False) -> Ref:
    """
    Wraps a value in a `Ref`, which allows for (1) deduplication of re-used values
    during serialization, and (2) updating the value of refs in live widgets.

    Args:
        value (Any): Initial value for the reference. If this is already a Ref and no id is provided, returns it unchanged.
        id (str, optional): Unique identifier for the reference. If not provided, a UUID will be generated.
    Returns:
        Ref: A reference object containing the initial value and id.
    """
    if state_key is None and isinstance(value, Ref):
        return value
    return Ref(value, state_key=state_key, sync=sync)


def unwrap_ref(maybeRef: Any) -> Any:
    """
    Unwraps a Ref if the input is one.

    Args:
        obj (Any): The object to unwrap.

    Returns:
        Any: The unwrapped object if input was a Ref, otherwise the input object.
    """
    if isinstance(maybeRef, Ref):
        return maybeRef.value
    return maybeRef


def Grid(*children, **kwargs):
    """
    Creates a responsive grid layout that automatically arranges child elements in a grid pattern.

    The grid adjusts the number of columns based on the available width and minimum width per item.
    Each item maintains consistent spacing controlled by gap parameters.

    Args:
        *children: Child elements to arrange in the grid.
        **opts: Grid options including:
            - minWidth (int): Minimum width for each grid item in pixels. Default is 165.
            - gap (int): Gap size for both row and column gaps. Default is 1.
            - rowGap (int): Vertical gap between rows. Overrides gap if specified.
            - colGap (int): Horizontal gap between columns. Overrides gap if specified.
            - cols (int): Fixed number of columns. If not set, columns are calculated based on minWidth.
            - minCols (int): Minimum number of columns. Default is 1.
            - maxCols (int): Maximum number of columns.
            - widths (List[Union[int, str]]): Array of column widths. Can be numbers (for fractions) or strings.
            - heights (List[Union[int, str]]): Array of row heights. Can be numbers (for fractions) or strings.
            - height (str): Container height.
            - style (dict): Additional CSS styles to apply to grid container.
            - className (str): Additional CSS classes to apply.

    Returns:
        A grid layout component that will be rendered in the JavaScript runtime.
    """
    return Hiccup(
        [JSRef("Grid"), kwargs, *children],
    )


Grid.for_json = lambda: JSRef("Grid")  # allow Grid to be used in hiccup
