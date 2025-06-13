import pathlib
import importlib.util
import os
from typing import TypedDict, Literal, Any, Union, cast


class Config(TypedDict):
    display_as: Literal["widget", "html"]
    dev: bool
    defaults: dict[Any, Any]


def configure(options: dict[str, Any] = {}, **kwargs: Any) -> None:
    CONFIG.update(cast(Config, {**options, **kwargs}))


def get_config(k: str) -> Union[str, None]:
    return CONFIG.get(k)


try:
    # First try the importlib.util approach
    util_spec = importlib.util.find_spec("colight.util")
    if util_spec and util_spec.origin:
        COLIGHT_PATH = pathlib.Path(util_spec.origin).parent
    else:
        # Fallback: Get the directory of the current file
        COLIGHT_PATH = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
except Exception:
    # Another fallback approach
    COLIGHT_PATH = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))

DIST_PATH = COLIGHT_PATH.parent / "js-dist"

CONFIG: Config = {"display_as": "widget", "dev": False, "defaults": {}}

# CDN URLs for published assets - set during package build
CDN_SCRIPT_URL = None

# Local development paths
WIDGET_URL = CDN_SCRIPT_URL or (DIST_PATH / "widget.mjs")

ANYWIDGET_URL = str(WIDGET_URL).replace("widget.mjs", "anywidget.mjs")
