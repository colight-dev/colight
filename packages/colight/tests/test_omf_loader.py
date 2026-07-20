"""Tests for colight.omf_loader against a small synthetic OMF project."""

import numpy as np
import pytest

omf = pytest.importorskip("omf")

from colight.omf_loader import load_omf  # noqa: E402


@pytest.fixture(scope="module")
def omf_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Write a synthetic OMF v1 project covering all four element kinds."""
    project = omf.Project(name="synthetic", description="")
    points = omf.PointSetElement(
        name="collars",
        geometry=omf.PointSetGeometry(
            vertices=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 4.0]]
        ),
        data=[omf.ScalarData(name="id", location="vertices", array=[0.0, 1.0, 2.0])],
    )
    lines = omf.LineSetElement(
        name="holes",
        geometry=omf.LineSetGeometry(
            vertices=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, -2.0],
                [2.0, 0.0, -4.0],
                [3.0, 0.0, -6.0],
            ],
            segments=[[0, 1], [1, 2], [2, 3]],
        ),
        data=[omf.ScalarData(name="grade", location="segments", array=[0.1, 0.5, 0.9])],
    )
    surface = omf.SurfaceElement(
        name="topo",
        geometry=omf.SurfaceGeometry(
            vertices=[
                [0.0, 0.0, 5.0],
                [10.0, 0.0, 5.0],
                [10.0, 10.0, 5.0],
                [0.0, 10.0, 5.0],
            ],
            triangles=[[0, 1, 2], [0, 2, 3]],
        ),
    )
    volume = omf.VolumeElement(
        name="blocks",
        geometry=omf.VolumeGridGeometry(
            tensor_u=[2.0, 2.0],
            tensor_v=[2.0, 2.0],
            tensor_w=[2.0, 2.0],
            origin=[100.0, 200.0, 300.0],
        ),
        data=[
            omf.ScalarData(
                name="grade",
                location="cells",
                array=np.arange(8, dtype=float).tolist(),
            )
        ],
    )
    project.elements = [points, lines, surface, volume]
    assert project.validate()
    path = tmp_path_factory.mktemp("omf") / "synthetic.omf"
    omf.OMFWriter(project, str(path))
    return str(path)


def test_elements_grouped_by_kind(omf_path: str) -> None:
    project = load_omf(omf_path)
    assert list(project.points) == ["collars"]
    assert list(project.line_sets) == ["holes"]
    assert list(project.surfaces) == ["topo"]
    assert list(project.volumes) == ["blocks"]


def test_world_coordinates_and_center(omf_path: str) -> None:
    project = load_omf(omf_path)
    lo, hi = project.bounds()
    # Geometry stays in true world coordinates (no loader re-centering); the
    # collar vertices are exactly the input coordinates.
    np.testing.assert_allclose(
        project.points["collars"].vertices,
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 4.0]],
    )
    # ``center`` is the suggested Scene origin: the midpoint of the full
    # project bounds (all elements, including the block-model volume, so it
    # differs from bounds() which only spans point/line/surface vertices).
    assert np.all(np.isfinite(project.center))
    assert project.center.shape == (3,)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))


def test_scene_origin_roundtrip(omf_path: str) -> None:
    """Scene(origin=project.center) shifts serialized positions to ~0; adding
    the origin back reproduces the original world coordinates (what pick-at
    reports)."""
    from colight import scene3d
    from colight.cli_tools.structure import collect_structure
    from colight.widget import to_json_with_state

    project = load_omf(omf_path)
    collars = project.points["collars"]
    scene = scene3d.Scene(collars.point_cloud(), origin=project.center)
    data, buffers = to_json_with_state(scene)
    arrays = {
        r.key: r
        for r in collect_structure(data, buffers).arrays
        if r.values is not None
    }
    shifted = np.asarray(arrays["centers"].values).reshape(-1, 3)
    world = shifted + project.center
    np.testing.assert_allclose(world, collars.vertices, atol=1e-2)


def test_line_starts_ends(omf_path: str) -> None:
    lines = load_omf(omf_path).line_sets["holes"]
    assert lines.starts.shape == (3, 3)
    assert lines.ends.shape == (3, 3)
    np.testing.assert_allclose(lines.ends[0], lines.starts[1])
    np.testing.assert_allclose(lines.segment_attributes["grade"], [0.1, 0.5, 0.9])


def test_grid_ordering_and_centers(omf_path: str) -> None:
    volume = load_omf(omf_path).volumes["blocks"]
    assert volume.shape == (2, 2, 2)
    grid = volume.grid("grade")
    # OMF v1 cell data is C-ordered over (nu, nv, nw): w varies fastest.
    assert grid[0, 0, 1] == 1.0
    assert grid[0, 1, 0] == 2.0
    assert grid[1, 0, 0] == 4.0
    centers = volume.cell_centers()
    np.testing.assert_allclose(centers[0], [101.0, 201.0, 301.0])
    np.testing.assert_allclose(centers[1], [101.0, 201.0, 303.0])  # +w
    np.testing.assert_allclose(centers[2], [101.0, 203.0, 301.0])  # +v


def test_filtered_cells_cutoff_and_stride(omf_path: str) -> None:
    volume = load_omf(omf_path).volumes["blocks"]
    centers, values = volume.filtered_cells("grade", cutoff=5.0)
    assert values.tolist() == [5.0, 6.0, 7.0]
    assert centers.shape == (3, 3)
    centers, values = volume.filtered_cells("grade", stride=2)
    assert values.tolist() == [0.0]


def test_component_builders(omf_path: str) -> None:
    project = load_omf(omf_path)
    cloud = project.points["collars"].point_cloud(size=2.0)
    assert cloud.type == "PointCloud"
    assert cloud.props["centers"].size == 9

    segments = project.line_sets["holes"].line_segments(color_by="grade")
    assert segments.type == "LineSegments"
    assert segments.props["colors"].size == 9
    # The colormap spec travels with the component (legend + reporting).
    meta = segments.props["color_by"]
    assert meta["cmap"] == "viridis"
    assert meta["label"] == "grade"
    assert meta["domain"] == [0.1, 0.9]

    mesh = project.surfaces["topo"].mesh(color=[1, 0, 0])
    assert mesh.type == "Mesh"
    assert mesh.props["geometry"]["indices"].size == 6

    cuboids = project.volumes["blocks"].cuboids("grade", cutoff=5.0)
    assert cuboids.type == "Cuboid"
    assert cuboids.props["centers"].size == 9
    assert cuboids.props["half_size"] == [1.0, 1.0, 1.0]
    # Domain anchors at the cutoff so legends read like the cutoff slider.
    assert cuboids.props["color_by"]["domain"] == [5.0, 7.0]


def test_line_segments_channels(omf_path: str) -> None:
    import colight.plot as Plot

    lines = project = load_omf(omf_path).line_sets["holes"]
    grade_class = lines.binned_categorical(
        "grade",
        edges=[0.3, 0.7],
        labels=["low", "mid", "high"],
        colors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    # grade = [0.1, 0.5, 0.9] -> codes 0 (low), 1 (mid), 2 (high).
    np.testing.assert_allclose(grade_class["values"], [0, 1, 2])
    assert [c["label"] for c in grade_class["categories"]] == ["low", "mid", "high"]

    comp = lines.line_segments_channels(
        channels={
            "Grade": {"attribute": "grade", "cmap": "viridis", "domain": (0, 1)},
            "Grade class": grade_class,
        },
        active_channel=Plot.js("$state.color_channel"),
        size=5.0,
    )
    props = comp.props
    assert set(props["color_channels"]) == {"Grade", "Grade class"}
    # Per-segment values expand 1:1 with segments (same path as color_by).
    assert props["color_channels"]["Grade"]["count"] == 3
    assert props["color_channels"]["Grade class"]["colorizer"]["kind"] == "categorical"

    with pytest.raises(ValueError, match="edges"):
        lines.binned_categorical("grade", edges=[0.5], labels=["a", "b", "c"])
