from colight.live_server.live_widgets import LiveWidgetManager
from colight.live_server.server import LiveServer


def test_live_widget_manager_updates_and_callbacks():
    async def send_message(_file_path, _message):
        return None

    manager = LiveWidgetManager(send_message)
    widget = manager.get_widget("widget-1", "/tmp/example.py")

    updates = [["clicks", "reset", 2], ["items", "append", "a"]]
    success, error = manager.handle_command(
        "widget-1", "handle_updates", {"updates": updates}
    )
    assert success
    assert error is None
    assert widget.state._state["clicks"] == 2
    assert widget.state._state["items"] == ["a"]

    called = {}

    def on_click(_widget, event):
        called["value"] = event["value"]

    widget.callback_registry["cb-1"] = on_click
    success, error = manager.handle_command(
        "widget-1", "handle_callback", {"id": "cb-1", "event": {"value": 7}}
    )
    assert success
    assert error is None
    assert called["value"] == 7


def test_live_widget_manager_error_handling():
    async def send_message(_file_path, _message):
        return None

    manager = LiveWidgetManager(send_message)
    manager.get_widget("widget-1", "/tmp/example.py")

    # Test unknown widget
    success, error = manager.handle_command("unknown-id", "handle_updates", {})
    assert not success
    assert "Unknown widget id" in error

    # Test missing parameter
    success, error = manager.handle_command("widget-1", "handle_updates", {})
    assert not success
    assert "Missing 'updates' parameter" in error

    # Test unknown callback
    success, error = manager.handle_command(
        "widget-1", "handle_callback", {"id": "unknown-cb", "event": {}}
    )
    assert not success
    assert "Unknown callback id" in error

    # Test unknown command
    success, error = manager.handle_command("widget-1", "unknown-command", {})
    assert not success
    assert "Unknown widget command" in error


def test_resolve_client_path_single_file(tmp_path):
    file_path = tmp_path / "example.py"
    file_path.write_text("print('ok')\n")

    server = LiveServer(
        file_path,
        include=["*.py"],
        ignore=None,
        open_url=False,
    )

    assert server._resolve_client_path("") == file_path
    assert server._resolve_client_path("example.py") == file_path
    assert server._resolve_client_path("example") == file_path
