from __future__ import annotations

import asyncio
import os
import pathlib
import sys

import numpy as np
from watchfiles import awatch

from scene3d_live import Scene3DApp

WATCH_DIR = pathlib.Path(__file__).parent


def create_app():
    """Create and configure the Scene3D app."""
    app = Scene3DApp(host="127.0.0.1", port=8001)

    @app.state
    def initial_state():
        return {"centers": np.array([0.0, 0.0, 0.0], dtype=np.float32)}

    @app.scene
    def build_scene(state, _session):
        centers = np.asarray(state["centers"], dtype=np.float32).reshape(-1)
        return {
            "components": [
                {
                    "type": "Cuboid",
                    "centers": centers,
                    "half_size": 0.3,
                    "color": [0.2, 0.7, 1.0],
                    "alpha": 0.9,
                    "drag": app.event("drag"),
                }
            ],
            "defaultCamera": {
                "position": [2.2, 2.2, 1.8],
                "target": [0, 0, 0],
                "up": [0, 0, 1],
                "fov": 45,
                "near": 0.01,
                "far": 100,
            },
            "controls": ["fps"],
        }

    @app.on("drag")
    def handle_drag(session, event):
        idx = int(event.index)
        position = np.array(event.position, dtype=np.float32)
        centers = np.asarray(session.state["centers"], dtype=np.float32).reshape(-1)
        base = idx * 3
        centers[base : base + 3] = position
        session.state["centers"] = centers

    return app


async def watch_and_restart():
    """Watch for Python file changes and restart."""
    async for changes in awatch(str(WATCH_DIR)):
        for _change_type, path in changes:
            if pathlib.Path(path).suffix == ".py":
                print(f"\n{pathlib.Path(path).name} changed, restarting...")
                os.execv(sys.executable, [sys.executable] + sys.argv)


async def main_async():
    app = create_app()
    print("\nScene3D Python server:")
    print(f"  ws://127.0.0.1:{app.port}")
    print("\nRun 'yarn dev' in this directory to start the frontend")
    print("Watching Python files for changes...")

    await asyncio.gather(
        app._serve(),
        watch_and_restart(),
    )


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
