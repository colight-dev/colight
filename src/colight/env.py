import pathlib
import importlib
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
    PARENT_PATH = pathlib.Path(importlib.util.find_spec("colight.util").origin).parent  # type: ignore
except AttributeError:
    raise ImportError("Cannot find the colight.util module")

CONFIG: Config = {"display_as": "widget", "dev": False, "defaults": {}}

# CDN URLs for published assets - set during package build
CDN_SCRIPT_URL = None
CDN_CSS_URL = None

# Local development paths
WIDGET_URL = CDN_SCRIPT_URL or (PARENT_PATH / "js/widget.dist.js")
CSS_URL = CDN_CSS_URL or (PARENT_PATH / "widget.css")
