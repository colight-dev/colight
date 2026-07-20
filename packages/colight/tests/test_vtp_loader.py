"""Tests for colight.vtp_loader against small in-test VTP fixtures.

Fixtures are generated with pyvista (which is also the reader backend), so the
whole module skips cleanly when pyvista is unavailable — mirroring how
test_omf_loader gates on ``omf``.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from colight.vtp_loader import load_vtp  # noqa: E402


def _write(mesh, tmp_path, name: str) -> str:
    path = tmp_path / name
    mesh.save(str(path))
    return str(path)


@pytest.fixture
def quad_mesh_path(tmp_path) -> str:
    """A 2-quad surface with a per-cell scalar and a per-point scalar."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    # Two quads (4-vertex polys) -> must triangulate to 4 triangles.
    faces = np.hstack([[4, 0, 1, 2, 3], [4, 1, 4, 5, 2]])
    mesh = pv.PolyData(points, faces)
    mesh.cell_data["region"] = np.array([10.0, 20.0])
    mesh.point_data["elevation"] = np.arange(6, dtype=float)
    return _write(mesh, tmp_path, "quads.vtp")


@pytest.fixture
def polyline_path(tmp_path) -> str:
    """A polyline drillhole set: two holes with per-cell grade."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -2.0],  # hole A: 3 pts -> 2 segments
            [1.0, 0.0, 0.0],
            [1.0, 0.0, -1.0],  # hole B: 2 pts -> 1 segment
        ]
    )
    # VTK polyline cell format: [nPts, i0, i1, ...] per line.
    lines = np.hstack([[3, 0, 1, 2], [2, 3, 4]])
    poly = pv.PolyData()
    poly.points = points
    poly.lines = lines
    poly.cell_data["grade"] = np.array([0.3, 0.9])  # one value per hole
    return _write(poly, tmp_path, "holes.vtp")


@pytest.fixture
def point_path(tmp_path) -> str:
    """A pure point set with a per-point scalar."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 2.0]])
    poly = pv.PolyData(points)
    poly.point_data["au"] = np.array([1.0, 2.0, 3.0])
    return _write(poly, tmp_path, "points.vtp")


# ---------------------------------------------------------------------------
# Mesh with cell scalars (triangulation)
# ---------------------------------------------------------------------------


def test_mesh_triangulates_quads(quad_mesh_path: str) -> None:
    data = load_vtp(quad_mesh_path)
    assert data.points.shape == (6, 3)
    # Two quad cells survive as 4-vertex polys before triangulation.
    assert len(data.polys) == 2
    assert all(len(c) == 4 for c in data.polys)
    assert "region" in data.cell_data
    np.testing.assert_allclose(data.cell_data["region"], [10.0, 20.0])
    assert "elevation" in data.point_data

    mesh = data.mesh()
    assert mesh.type == "Mesh"
    # Each quad fan-triangulates to 2 triangles -> 4 triangles -> 12 indices.
    assert mesh.props["geometry"]["indices"].size == 12


def test_mesh_triangulates_triangle_strip(tmp_path) -> None:
    # A single triangle strip over 4 points -> 2 triangles. Strips must use
    # strip semantics (not a naive fan), handled via pyvista .triangulate().
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    poly = pv.PolyData()
    poly.points = points
    poly.strips = np.hstack([[4, 0, 1, 2, 3]])
    path = str(tmp_path / "strip.vtp")
    poly.save(path)

    data = load_vtp(path)
    mesh = data.mesh()
    idx = np.asarray(mesh.props["geometry"]["indices"]).reshape(-1, 3)
    # 4-point strip -> 2 triangles.
    assert idx.shape == (2, 3)
    # Every index references a real vertex.
    assert idx.max() < 4


def test_mesh_color_by_point_scalar(quad_mesh_path: str) -> None:
    data = load_vtp(quad_mesh_path)
    mesh = data.mesh(color_by="elevation")
    # Point-data scalars map to per-vertex colors (Mesh colors per vertex,
    # stored on the inline geometry).
    vc = np.asarray(mesh.props["geometry"]["colors"]).reshape(-1, 3)
    assert vc.shape == (6, 3)


# ---------------------------------------------------------------------------
# Polyline set -> line_segments expansion (the drillhole case)
# ---------------------------------------------------------------------------


def test_polylines_split_into_segments(polyline_path: str) -> None:
    data = load_vtp(polyline_path)
    assert len(data.lines) == 2
    seg = data.line_segments()
    assert seg.type == "LineSegments"
    starts = np.asarray(seg.props["starts"]).reshape(-1, 3)
    ends = np.asarray(seg.props["ends"]).reshape(-1, 3)
    # 3-pt hole -> 2 segments, 2-pt hole -> 1 segment => 3 segments total.
    assert starts.shape == (3, 3)
    assert ends.shape == (3, 3)
    # Exact endpoint assertions for hole A's two segments and hole B's one.
    np.testing.assert_allclose(starts[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(ends[0], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(starts[1], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(ends[1], [0.0, 0.0, -2.0])
    np.testing.assert_allclose(starts[2], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(ends[2], [1.0, 0.0, -1.0])


def test_cell_data_repeats_per_segment(polyline_path: str) -> None:
    data = load_vtp(polyline_path)
    # Hole A (grade 0.3) emits 2 segments; hole B (grade 0.9) emits 1.
    expanded = data.segment_cell_values("grade")
    np.testing.assert_allclose(expanded, [0.3, 0.3, 0.9])

    # color_by on a cell attribute expands identically (one color per segment).
    seg = data.line_segments(color_by="grade")
    colors = np.asarray(seg.props["colors"]).reshape(-1, 3)
    assert colors.shape == (3, 3)
    # Same grade -> same color: the two hole-A segments match, hole-B differs.
    np.testing.assert_allclose(colors[0], colors[1])
    assert not np.allclose(colors[0], colors[2])


def test_cell_data_as_per_segment_sizes(polyline_path: str) -> None:
    # The structural-intensity-thickness path: cell data -> per-segment radii.
    data = load_vtp(polyline_path)
    radii = data.segment_cell_values("grade")
    seg = data.line_segments(sizes=radii)
    sizes = np.asarray(seg.props["sizes"]).reshape(-1)
    assert sizes.shape == (3,)
    np.testing.assert_allclose(sizes, [0.3, 0.3, 0.9])


# ---------------------------------------------------------------------------
# Point cloud with point scalars
# ---------------------------------------------------------------------------


def test_point_cloud_point_scalars(point_path: str) -> None:
    data = load_vtp(point_path)
    assert data.points.shape == (3, 3)
    assert "au" in data.point_data
    cloud = data.point_cloud(color_by="au")
    assert cloud.type == "PointCloud"
    assert np.asarray(cloud.props["centers"]).size == 9
    assert cloud.props["color_by"]["label"] == "au"
