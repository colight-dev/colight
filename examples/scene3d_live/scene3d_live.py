from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, Optional

import websockets

from colight_wire_protocol import pack_message, unpack_message


class Event(SimpleNamespace):
    """Event payload with attribute and dict-style access."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass(frozen=True)
class EventToken:
    """Marker used in scene payloads to bind client-side events."""

    name: str


def _normalize_payload(value: Any) -> Any:
    if isinstance(value, EventToken):
        return {"__event__": value.name}
    if isinstance(value, dict):
        return {k: _normalize_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_payload(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_normalize_payload(v) for v in value)
    return value


class WireSocket:
    def __init__(self, websocket):
        self._ws = websocket

    async def send(self, payload: Any) -> None:
        normalized = _normalize_payload(payload)
        envelope, buffers = pack_message(normalized)
        await self._ws.send(json.dumps(envelope))
        for buffer in buffers:
            await self._ws.send(buffer)

    async def recv(self) -> Any:
        raw = await self._ws.recv()
        if isinstance(raw, bytes):
            raise ValueError("Expected JSON envelope before binary buffers")
        envelope = json.loads(raw)
        buffers = []
        for _ in range(envelope.get("buffer_count", 0)):
            buffer = await self._ws.recv()
            if isinstance(buffer, str):
                raise ValueError("Expected binary buffer frame")
            buffers.append(buffer)
        return unpack_message(envelope, buffers)


class Scene3DSession:
    def __init__(
        self,
        session_id: str,
        wire: WireSocket,
        state: Dict[str, Any],
    ) -> None:
        self.id = session_id
        self.state = state
        self._wire = wire

    async def send(self, payload: Any) -> None:
        await self._wire.send(payload)

    async def send_scene(self, scene: Dict[str, Any]) -> None:
        await self.send({"type": "scene", "scene": scene})

    def update(self, updates: Dict[str, Any]) -> None:
        self.state.update(updates)


StateFactory = Callable[[], Dict[str, Any]]
SceneBuilder = Callable[[Dict[str, Any], Scene3DSession], Dict[str, Any]]
EventHandler = Callable[[Scene3DSession, Event], Any | Awaitable[Any]]


class Scene3DApp:
    """Minimal scene3d live server for per-session state updates."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8001) -> None:
        self.host = host
        self.port = port
        self._state_factory: Optional[StateFactory] = None
        self._scene_builder: Optional[SceneBuilder] = None
        self._handlers: Dict[str, EventHandler] = {}

    def state(self, func: StateFactory) -> StateFactory:
        self._state_factory = func
        return func

    def scene(self, func: SceneBuilder) -> SceneBuilder:
        self._scene_builder = func
        return func

    def on(self, name: str) -> Callable[[EventHandler], EventHandler]:
        def decorator(func: EventHandler) -> EventHandler:
            self._handlers[name] = func
            return func

        return decorator

    def event(self, name: str) -> EventToken:
        return EventToken(name)

    async def _handle_event(self, session: Scene3DSession, message: dict) -> None:
        event_name = message.get("event")
        if not event_name:
            return
        handler = self._handlers.get(event_name)
        if not handler:
            return

        payload = message.get("data") or {}
        event = Event(**payload) if isinstance(payload, dict) else payload
        result = handler(session, event)
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, dict):
            session.update(result)

        if not self._scene_builder:
            return
        scene = self._scene_builder(session.state, session)
        await session.send_scene(scene)

    async def _handle_client(self, websocket):
        if not self._state_factory or not self._scene_builder:
            raise RuntimeError("Scene3DApp requires state() and scene() handlers")

        wire = WireSocket(websocket)
        session = Scene3DSession(uuid.uuid4().hex, wire, self._state_factory())

        await session.send({"type": "init", "scene": self._scene_builder(session.state, session)})

        try:
            while True:
                message = await wire.recv()
                if isinstance(message, dict) and message.get("type") == "event":
                    await self._handle_event(session, message)
        except websockets.exceptions.ConnectionClosed:
            return

    async def _serve(self) -> None:
        async with websockets.serve(
            self._handle_client, self.host, self.port, reuse_address=True
        ):
            await asyncio.Future()

    def run(self) -> None:
        asyncio.run(self._serve())
