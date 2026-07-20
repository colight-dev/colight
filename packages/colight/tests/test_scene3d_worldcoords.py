"""Tests for the world-coordinate / usability cluster (Wolfpass friction log).

Covers the data-layer (non-GPU) parts of five fixes:

1. ``Scene(origin=...)`` translates position-typed attributes by -origin at
   serialization and carries the offset as metadata; ``pick-at`` adds it back.
2. Camera auto-fit is the default (JS-side; smoke here that no explicit camera
   is required to serialize).
3. Far-plane / lost-scene warnings: ``inspect`` (data-side camera-frustum) and
   the coverage ``mostly-background`` warning (pixel-side).
4. Mesh ergonomics: default ``center``; no false ``length-mismatch`` for a
   single-instance mesh's per-vertex arrays.
5. ``pick-at --min-alpha`` splits transparent occluders out of ``hits``.

The GPU-dependent halves (actual render correctness, decoration-aware alpha in
``describeInstance``) are covered by the JS/Chrome tests.
"""

import numpy as np

from colight import scene3d
from colight.cli_tools import scene_pick
from colight.cli_tools.inspect_tools import (
    _implied_instance_count,
    inspect_visual_data,
)
from colight.cli_tools.structure import ArrayRecord, collect_structure
from colight.widget import to_json_with_state


def _serialize(scene: scene3d.Scene):
    """Serialize a Scene to the (data envelope, buffers) the CLI inspects."""
    return to_json_with_state(scene)


def _arrays_by_key(scene: scene3d.Scene):
    data, buffers = _serialize(scene)
    state = collect_structure(data, buffers)
    return {r.key: r for r in state.arrays if r.values is not None}


# ============================ Fix 1: origin ============================

UTM = 445000.0


def test_origin_shifts_centers():
    """Positions are translated by -origin at serialization."""
    scene = scene3d.Scene(
        scene3d.PointCloud(centers=[[UTM, UTM, 100.0], [UTM + 10, UTM + 5, 120.0]]),
        origin=[UTM, UTM, 100.0],
    )
    centers = _arrays_by_key(scene)["centers"].values
    # Fails without the fix: centers would still be ~445000 (float32-lossy).
    np.testing.assert_allclose(centers, [0, 0, 0, 10, 5, 20], atol=1e-3)


def test_origin_travels_as_metadata():
    scene = scene3d.Scene(
        scene3d.PointCloud(center=[UTM, UTM, 0.0]), origin=[UTM, UTM, 0.0]
    )
    # for_json puts the origin on the Scene props so JS/CLI can add it back.
    _ref, props = scene.for_json()
    assert props["origin"] == [UTM, UTM, 0.0]


def test_origin_shifts_line_and_mesh_positions():
    """starts/ends shift by -origin; a mesh re-centers so the *composite*
    (center + local vertex) lands in the shifted frame with small buffers."""
    scene = scene3d.Scene(
        scene3d.LineSegments(starts=[[UTM, 0, 0]], ends=[[UTM + 4, 0, 0]]),
        scene3d.Mesh(
            positions=[[UTM, 0, 0], [UTM + 1, 1, 0], [UTM, 1, 0]],
            indices=[0, 1, 2],
        ),
        origin=[UTM, 0, 0],
    )
    arrays = _arrays_by_key(scene)
    np.testing.assert_allclose(arrays["starts"].values, [0, 0, 0], atol=1e-3)
    np.testing.assert_allclose(arrays["ends"].values, [4, 0, 0], atol=1e-3)
    # Mesh geometry is folded to its centroid: local coords are small (no
    # ~445000-magnitude vertices left in the float32 GPU buffer).
    local = np.asarray(arrays["positions"].values).reshape(-1, 3)
    assert np.abs(local).max() < 5.0
    # The composite center + local vertex reproduces the origin-shifted world
    # vertices [0,0,0], [1,1,0], [0,1,0] exactly (identity of the fold).
    center = np.asarray(arrays["centers"].values).reshape(3)
    np.testing.assert_allclose(
        center + local, [[0, 0, 0], [1, 1, 0], [0, 1, 0]], atol=1e-2
    )


def test_no_origin_leaves_positions_untouched():
    scene = scene3d.Scene(scene3d.PointCloud(center=[UTM, UTM, 0.0]))
    centers = _arrays_by_key(scene)["centers"].values
    np.testing.assert_allclose(centers, [UTM, UTM, 0.0], rtol=1e-4)


def test_world_mesh_not_double_shifted():
    """Regression for the black-scene bug: a world-space mesh (center defaults
    to [0,0,0]) must be shifted by -origin exactly ONCE. The old code shifted
    geometry.positions AND centers, landing the composite at ~-origin (445 km
    off screen); centroid-folding + a single center shift keeps it near 0."""
    rng = np.random.default_rng(1)
    verts = (rng.random((2000, 3)) * 3000.0 + [UTM, UTM, 2900.0]).astype("float32")
    origin = verts.mean(axis=0)
    scene = scene3d.Scene(
        scene3d.Mesh(positions=verts, indices=np.arange(1998)), origin=origin
    )
    arrays = _arrays_by_key(scene)
    center = np.asarray(arrays["centers"].values).reshape(3)
    local = np.asarray(arrays["positions"].values).reshape(-1, 3)
    # Local geometry is small (float32-safe), not ~445000-magnitude.
    assert np.abs(local).max() < 2000.0
    # Composite of every vertex lands within the shifted extent (~[-1500,1500]),
    # NOT ~-445000. This is the single-shift invariant.
    composite = center + local
    assert np.abs(composite).max() < 2000.0


def test_recentering_preserves_composite_with_transform():
    """Centroid-fold accounts for per-instance scale + rotation so the
    composite world position is unchanged (identity)."""
    # 90deg rotation about Z as [w,x,y,z]; scale 2.
    q = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    positions = np.array([[10.0, 0, 0], [12, 0, 0], [10, 4, 0]], dtype="float32")
    center = [100.0, 200.0, 0.0]
    mesh = scene3d.Mesh(
        positions=positions, indices=[0, 1, 2], center=center, quaternion=q, scale=2.0
    )
    scene = scene3d.Scene(mesh, origin=[0, 0, 0])
    arrays = _arrays_by_key(scene)
    local = np.asarray(arrays["positions"].values).reshape(-1, 3)
    new_center = np.asarray(arrays["centers"].values).reshape(3)
    # Reproduce the GPU composite: center + R*(S*local), R from [w,x,y,z].
    from colight.scene3d import _quat_rotate

    composite = new_center + _quat_rotate(np.asarray(q), 2.0 * local)
    expected = np.asarray(center) + _quat_rotate(np.asarray(q), 2.0 * positions)
    np.testing.assert_allclose(composite, expected, atol=1e-2)


# ============================ Fix 4: mesh ============================


def test_mesh_defaults_center():
    """A plain world-space mesh needs no center=[0,0,0]."""
    # Fails without the fix: Mesh raised ValueError for a missing center.
    mesh = scene3d.Mesh(positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], indices=[0, 1, 2])
    assert mesh.type == "Mesh"
    np.testing.assert_allclose(mesh.props["centers"], [0, 0, 0])


def test_single_instance_mesh_no_length_mismatch():
    """Per-vertex geometry arrays don't trip the per-instance length check."""
    verts = np.random.default_rng(0).random((5000, 3)).astype("float32")
    faces = np.arange(4998).reshape(-1, 3)
    scene = scene3d.Scene(scene3d.Mesh(positions=verts, indices=faces))
    data, buffers = _serialize(scene)
    _payload, warnings = inspect_visual_data(data, buffers)
    codes = [w["code"] for w in warnings]
    # Fails without the fix: "length-mismatch" (centers=1 vs positions=5000).
    assert "length-mismatch" not in codes


def test_geometry_positions_excluded_from_instance_count():
    geom = ArrayRecord(
        path="x/scene3d.Mesh[0].geometry.positions",
        key="positions",
        values=None,
        dtype="f",
        shape=[300],
    )
    top = ArrayRecord(
        path="x/scene3d.Mesh[0].centers",
        key="centers",
        values=None,
        dtype="f",
        shape=[3],
    )
    assert _implied_instance_count(geom) is None
    assert _implied_instance_count(top) == 1


# ==================== Fix 3: warnings (data + pixel) ====================


def test_camera_frustum_warning_when_scene_beyond_far():
    """Unit-scale far on a kilometre-scale scene warns (would render black)."""
    scene = scene3d.Scene(
        scene3d.PointCloud(centers=[[0, 0, 0], [3000, 3000, 300]])
    ) + {
        "defaultCamera": {
            "position": [3600, -3400, 2800],
            "target": [0, 0, 0],
            "up": [0, 0, 1],
            "fov": 45,
            "near": 0.001,
            "far": 100.0,
        }
    }
    data, buffers = _serialize(scene)
    _payload, warnings = inspect_visual_data(data, buffers)
    # Fails without the fix: nothing warned the scene was outside the frustum.
    assert any(w["code"] == "camera-frustum" for w in warnings)


def test_camera_frustum_ok_with_fitting_far():
    scene = scene3d.Scene(
        scene3d.PointCloud(centers=[[0, 0, 0], [3000, 3000, 300]])
    ) + {
        "defaultCamera": {
            "position": [3600, -3400, 2800],
            "target": [0, 0, 0],
            "up": [0, 0, 1],
            "fov": 45,
            "near": 5.0,
            "far": 50000.0,
        }
    }
    data, buffers = _serialize(scene)
    _payload, warnings = inspect_visual_data(data, buffers)
    assert not any(w["code"] == "camera-frustum" for w in warnings)


def test_coverage_warning_mostly_background():
    coverage = {"background": {"fraction": 0.995, "pixels": 1}, "components": []}
    warnings = scene_pick.coverage_warnings(coverage)
    # Fails without the fix: no warning was emitted for a near-empty frame.
    assert len(warnings) == 1
    assert warnings[0]["code"] == "mostly-background"


def test_coverage_warning_absent_when_populated():
    coverage = {"background": {"fraction": 0.4, "pixels": 1}, "components": []}
    assert scene_pick.coverage_warnings(coverage) == []


# ==================== Fix 5b: min-alpha occluder split ====================


class _FakeStudio:
    """Minimal studio stub for pick_at_source without a browser."""

    def __init__(self, alpha_by_instance):
        self._alpha = alpha_by_instance


def test_min_alpha_splits_occluders(monkeypatch):
    """Transparent hits move from ``hits`` into ``occluders``."""
    # Two hits: instance 0 is a 25%-alpha occluder, instance 1 is opaque.
    fake_hits = [
        {
            "component": 0,
            "type": "Mesh",
            "instance": 0,
            "distance": 0.0,
            "pixels": 4,
            "share": 0.5,
        },
        {
            "component": 1,
            "type": "Cuboid",
            "instance": 5,
            "distance": 1.0,
            "pixels": 4,
            "share": 0.5,
        },
    ]
    alpha_by = {(0, 0): 0.25, (1, 5): 1.0}

    class _Snapshot:
        rect = {"left": 0.0, "top": 0.0, "width": 10, "height": 10}
        width = 10
        height = 10
        dpr = 1.0
        scenes = 1
        ids = np.zeros((10, 10), dtype="uint32")

    class _Scene:
        studio = object()
        block_id = None

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def scene(self):
            return self

    class _Source:
        block_id = None

        def scene(self):
            return _Scene()

    monkeypatch.setattr(scene_pick, "require_scene", lambda studio: 1)
    monkeypatch.setattr(scene_pick, "take_snapshot", lambda studio: _Snapshot())
    monkeypatch.setattr(scene_pick, "hits_at", lambda snap, x, y, r: list(fake_hits))
    monkeypatch.setattr(
        scene_pick,
        "instance_values",
        lambda studio, comp, inst: {"alpha": alpha_by[(comp, inst)]},
    )

    payload = scene_pick.pick_at_source(
        _Source(), "target", 5.0, 5.0, radius=6.0, min_alpha=0.5
    )
    # Fails without the fix: the occluder stays in hits and shadows the pick.
    assert [h["instance"] for h in payload["hits"]] == [5]
    assert [o["instance"] for o in payload["occluders"]] == [0]
    assert payload["min_alpha"] == 0.5


def test_pick_at_without_min_alpha_has_no_occluders(monkeypatch):
    class _Snapshot:
        rect = {"left": 0.0, "top": 0.0, "width": 10, "height": 10}
        width = 10
        height = 10
        dpr = 1.0
        scenes = 1
        ids = np.zeros((10, 10), dtype="uint32")

    class _Scene:
        studio = object()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Source:
        block_id = None

        def scene(self):
            return _Scene()

    monkeypatch.setattr(scene_pick, "require_scene", lambda studio: 1)
    monkeypatch.setattr(scene_pick, "take_snapshot", lambda studio: _Snapshot())
    monkeypatch.setattr(scene_pick, "hits_at", lambda snap, x, y, r: [])

    payload = scene_pick.pick_at_source(_Source(), "target", 5.0, 5.0)
    assert "occluders" not in payload
    assert "min_alpha" not in payload
