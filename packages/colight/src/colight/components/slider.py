from colight.layout import JSExpr, JSRef, Ref, js, Hiccup, LayoutItem
from colight.protocols import Collector
from typing import Any, Optional, Dict, Union, List


_Slider = JSRef("Slider")


class Slider(LayoutItem, Collector):
    def __init__(
        self,
        key: Union[str, JSExpr],
        init: Any = None,
        range: Optional[Union[int, float, List[Union[int, float]], JSExpr]] = None,
        rangeFrom: Any = None,
        fps: Optional[Union[int, str, JSExpr]] = None,
        autoplay: Optional[Union[bool, JSExpr]] = None,
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
            autoplay (bool, optional): If True, animation starts automatically. Defaults to True when fps is provided.
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
        Plot.Slider("frame", init=0, range=100, fps=30, autoplay=False, label="Frame")
        ```
        """
        super().__init__()

        if range is None and rangeFrom is None:
            raise ValueError("'range', or 'rangeFrom' must be defined")
        if tail and rangeFrom is None:
            raise ValueError(
                "Slider: 'tail' can only be used when 'rangeFrom' is provided"
            )

        self.key = key
        self.fps = fps
        self.range = range
        self.rangeFrom = rangeFrom
        self.step = step

        if init is None:
            init = js(f"$state.{key}")
        else:
            init = Ref(init, state_key=key)

        self.slider_options = kwargs | {
            "state_key": key,
            "init": init,
            "range": range,
            "rangeFrom": rangeFrom,
            "fps": fps,
            "autoplay": autoplay,
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

    def for_json(self):
        return Hiccup([_Slider, self.slider_options]).for_json()

    def collect(self, collector):
        """Collect animation metadata if animatable and return serialized slider."""
        # Store metadata when animatable (string key + fps)
        if isinstance(self.key, str) and self.fps is not None:
            collector.animateBy.append(
                {
                    "key": self.key,
                    "range": self.range,
                    "rangeFrom": self.rangeFrom,
                    "fps": self.fps,
                    "step": self.step,
                }
            )

        # Import here to avoid circular import
        from colight.widget import to_json

        # Serialize the hiccup structure completely
        hiccup = Hiccup([_Slider, self.slider_options])
        return to_json(hiccup, collected_state=collector)
