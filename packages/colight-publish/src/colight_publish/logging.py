"""Shared Rich logging utilities."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Optional


def _import_rich():
    try:
        from rich.console import Console
        from rich.live import Live
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        from rich.table import Table

        return {
            "Console": Console,
            "Live": Live,
            "Layout": Layout,
            "Panel": Panel,
            "Table": Table,
            "Progress": Progress,
            "SpinnerColumn": SpinnerColumn,
            "TextColumn": TextColumn,
            "BarColumn": BarColumn,
            "TimeElapsedColumn": TimeElapsedColumn,
        }
    except ModuleNotFoundError:
        return None


def build_rich_progress() -> Optional[Any]:
    rich = _import_rich()
    if not rich:
        return None

    Progress = rich["Progress"]
    SpinnerColumn = rich["SpinnerColumn"]
    TextColumn = rich["TextColumn"]
    BarColumn = rich["BarColumn"]
    TimeElapsedColumn = rich["TimeElapsedColumn"]

    return Progress(
        SpinnerColumn(),
        TextColumn("{task.completed}/{task.total}", justify="right"),
        BarColumn(bar_width=None),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
    )


def rich_console():
    rich = _import_rich()
    if not rich:
        return None
    return rich["Console"]()


def format_duration_ms(ms: float) -> str:
    return f"{ms:.1f} ms"


class LiveDashboard:
    """Optional Rich dashboard showing watched files and logs."""

    def __init__(self, client_registry):
        rich = _import_rich()
        if not rich:
            self.live = None
            return

        Console = rich["Console"]
        Layout = rich["Layout"]
        Panel = rich["Panel"]
        Table = rich["Table"]
        Live = rich["Live"]

        self.client_registry = client_registry
        self.console = Console()
        self.layout = Layout()
        self.layout.split(Layout(name="watchers", size=10), Layout(name="log"))

        self.Panel = Panel
        self.Table = Table
        self.live = Live(
            self.layout, console=self.console, refresh_per_second=4, transient=False
        )
        self.log_lines: Deque[str] = deque(maxlen=200)

    def __bool__(self):
        return self.live is not None

    def start(self):
        if self.live:
            self.live.__enter__()
            self._refresh()

    def stop(self):
        if self.live:
            self.live.__exit__(None, None, None)
            self.live = None

    def log(self, message: str, style: str = "white") -> bool:
        if not self.live:
            return False
        self.log_lines.append(f"[{style}]{message}[/{style}]")
        self._refresh_logs()
        return True

    def refresh_watchers(self):
        if not self.live:
            return
        table = self.Table(show_header=True, header_style="bold magenta")
        table.add_column("Client", style="cyan", no_wrap=True)
        table.add_column("Watching", style="green")

        rows_added = False
        for client_id, path in sorted(self.client_registry.client_files.items()):
            table.add_row(client_id, path or "—")
            rows_added = True

        if not rows_added:
            table.add_row("—", "No active watchers")

        self.layout["watchers"].update(self.Panel(table, title="Watched Files"))

    def _refresh_logs(self):
        if not self.live:
            return
        log_text = "\n".join(self.log_lines) if self.log_lines else "No events yet"
        self.layout["log"].update(
            self.Panel(log_text, title="Events", border_style="cyan")
        )

    def _refresh(self):
        self.refresh_watchers()
        self._refresh_logs()
