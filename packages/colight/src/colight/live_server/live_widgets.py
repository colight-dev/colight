import asyncio
import base64
import logging
from typing import Any, Callable, Dict, List, Optional

from colight_serde import replace_buffers
from colight.widget import SubscriptableNamespace, WidgetState

logger = logging.getLogger(__name__)


class LiveWidget:
    def __init__(
        self,
        widget_id: str,
        file_path: str,
        send_message: Callable[[str, Dict[str, Any]], Any],
    ):
        self.id = widget_id
        self.file_path = file_path
        self.callback_registry: Dict[str, Any] = {}
        self.state = WidgetState(self)
        self._send_message = send_message

    def update_file(self, file_path: str) -> None:
        self.file_path = file_path

    def send(self, message: Dict[str, Any], buffers: Optional[List[bytes]] = None):
        payload = dict(message)
        payload["widgetId"] = self.id
        if buffers:
            payload["buffers"] = [
                base64.b64encode(buffer).decode("ascii") for buffer in buffers
            ]

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._send_message(self.file_path, payload))
            return

        loop.create_task(self._send_message(self.file_path, payload))


class LiveWidgetManager:
    def __init__(self, send_message: Callable[[str, Dict[str, Any]], Any]):
        self._send_message = send_message
        self._widgets: Dict[str, LiveWidget] = {}

    def get_widget(self, widget_id: str, file_path: str) -> LiveWidget:
        widget = self._widgets.get(widget_id)
        if widget is None:
            widget = LiveWidget(widget_id, file_path, self._send_message)
            self._widgets[widget_id] = widget
            print(f"[LiveWidgetManager] Created widget: {widget_id}", flush=True)
        else:
            widget.update_file(file_path)
        return widget

    def remove_widget(self, widget_id: str) -> bool:
        """Remove a widget from the manager. Returns True if widget existed."""
        if widget_id in self._widgets:
            del self._widgets[widget_id]
            print(f"[LiveWidgetManager] Removed widget: {widget_id}", flush=True)
            return True
        return False

    def handle_command(
        self,
        widget_id: str,
        command: str,
        params: Dict[str, Any],
        buffers: Optional[List[bytes]] = None,
    ) -> tuple[bool, Optional[str]]:
        """Handle a widget command from the client.

        Returns:
            (success, error_message): success=True if handled, error_message if failed
        """
        print(f"[LiveWidgetManager] handle_command: widget_id={widget_id}, command={command}", flush=True)
        print(f"[LiveWidgetManager] known widgets: {list(self._widgets.keys())}", flush=True)
        widget = self._widgets.get(widget_id)
        if widget is None:
            error = f"Unknown widget id: {widget_id}"
            print(f"[LiveWidgetManager] {error}", flush=True)
            logger.warning(error)
            return (False, error)

        try:
            if command == "handle_updates":
                updates = params.get("updates")
                if updates is None:
                    return (False, "Missing 'updates' parameter")
                if buffers:
                    updates = replace_buffers(updates, buffers)
                widget.state.accept_js_updates(updates)
                return (True, None)

            if command == "handle_callback":
                callback_id = params.get("id")
                if callback_id not in widget.callback_registry:
                    error = f"Unknown callback id: {callback_id}"
                    logger.warning(error)
                    return (False, error)
                event = params.get("event", {})
                if buffers:
                    event = replace_buffers(event, buffers)
                if isinstance(event, dict):
                    event = SubscriptableNamespace(**event)
                widget.callback_registry[callback_id](widget, event)
                return (True, None)
        except Exception as exc:
            error = f"Widget command failed: {command}: {exc}"
            logger.exception(error)
            return (False, error)

        error = f"Unknown widget command: {command}"
        logger.warning(error)
        return (False, error)
