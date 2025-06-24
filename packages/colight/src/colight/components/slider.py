from colight.layout import JSExpr, JSRef, Ref, js, Hiccup
from typing import Any, Optional, Dict, Union, List


_Slider = JSRef("Slider")


def Slider(
    key: Union[str, JSExpr],
    init: Any = None,
    range: Optional[Union[int, float, List[Union[int, float]], JSExpr]] = None,
    rangeFrom: Any = None,
    fps: Optional[Union[int, str, JSExpr]] = None,
    step: Union[int, float, JSExpr] = 1,
    tail: Union[bool, JSExpr] = False,
    loop: Union[bool, JSExpr] = True,
    label: Optional[Union[str, JSExpr]] = None,
    showValue: Union[bool, JSExpr] = False,
    controls: Optional[Union[List[str], JSExpr, bool]] = None,
    className: Optional[str] = None,
    style: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
):
    """
    Creates a slider with reactive functionality, allowing for dynamic interaction and animation.

    Args:
        key (str): The key for the reactive variable in the state.
        init (Any, optional): Initial value for the variable.
        range (Union[int, List[int]], optional):  A list of two values, `[from, until]` (inclusive), to be traversed by `step`. Or a single value `n` which becomes `[from, n-1]`, aligned with python's range(n).
        rangeFrom (Any, optional): Derive the range from the length of this (ref) argument.
        fps (int, optional): Frames per second for animation through the range. If > 0, enables animation.
        step (int, optional): Step size for the range. Defaults to 1.
        tail (bool, optional): If True, animation stops at the end of the range. Defaults to False.
        loop (bool, optional): If True, animation loops back to start when reaching the end. Defaults to True.
        label (str, optional): Label for the slider.
        showValue (bool, optional): If True, shows the current value immediately after the label.
        controls (list, optional): List of controls to display, such as ["slider", "play", "fps"]. Defaults to ["slider"] if fps is not set, otherwise ["slider", "play"].
        **kwargs: Additional keyword arguments.

    Returns:
        Slider: A Slider component with the specified options.

    Example:
    ```python
    Plot.Slider("frame", init=0, range=100, fps=30, label="Frame")
    ```
    """

    if range is None and rangeFrom is None:
        raise ValueError("'range', or 'rangeFrom' must be defined")
    if tail and rangeFrom is None:
        raise ValueError("Slider: 'tail' can only be used when 'rangeFrom' is provided")

    if init is None:
        init = js(f"$state.{key}")
    else:
        init = Ref(init, state_key=key)

    slider_options = kwargs | {
        "state_key": key,
        "init": init,
        "range": range,
        "rangeFrom": rangeFrom,
        "fps": fps,
        "step": step,
        "tail": tail,
        "loop": loop,
        "label": label,
        "showValue": showValue,
        "controls": controls,
        "className": className,
        "style": style,
        "kind": "Slider",
    }

    hiccup = Hiccup([_Slider, slider_options])

    # Store metadata when animatable (string key + fps)
    if isinstance(key, str) and fps is not None:
        hiccup._state_animate_by = {  # type: ignore
            "key": key,
            "range": range,
            "rangeFrom": rangeFrom,
            "fps": fps,
            "step": step,
        }

    return hiccup
