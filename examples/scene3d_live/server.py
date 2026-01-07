from __future__ import annotations

import pathlib

import numpy as np

from colight.http_server import ColightHTTPServer
from scene3d_live import Scene3DApp


ROOT = pathlib.Path(__file__).resolve().parents[2]
WEB_ROOT = pathlib.Path(__file__).parent / "web"
SCENE3D_PATH = ROOT / "packages/colight-scene3d/dist/scene3d.mjs"
SCENE3D_MAP_PATH = ROOT / "packages/colight-scene3d/dist/scene3d.mjs.map"

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


def main() -> None:
    http = ColightHTTPServer(
        host="127.0.0.1",
        port=8000,
        static_dir=str(WEB_ROOT),
    )
    http.add_served_file("scene3d.mjs", SCENE3D_PATH.read_bytes())
    if SCENE3D_MAP_PATH.exists():
        http.add_served_file("scene3d.mjs.map", SCENE3D_MAP_PATH.read_bytes())
    http.start()
    print("Scene3D live demo:")
    print(f"- http://127.0.0.1:{http.actual_port}/")
    print(f"- ws://127.0.0.1:{app.port}")
    app.run()


if __name__ == "__main__":
    main()
