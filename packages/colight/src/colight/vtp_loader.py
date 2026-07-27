"""Load VTP (VTK XML PolyData) files into colight scene3d components.

Requires an optional VTK reader: ``pyvista`` (preferred) or ``vtk``
(``pip install colight[geo]``). VTP is the ParaView-native PolyData format, so
this reader lets users with existing ``.vtp`` exports (drillhole traces, block
surfaces, sampled point sets) view them in colight without a conversion step.

The loader normalizes a PolyData into a backend-independent
:class:`VTPPolyData`:

- ``points``      -> ``(N, 3)`` float32 vertex positions
- ``polys``       -> list of per-cell vertex-index arrays (polygons)
- ``lines``       -> list of per-cell vertex-index arrays (polylines)
- ``point_data``  -> dict of named ``(N, ...)`` numpy arrays (per vertex)
- ``cell_data``   -> dict of named ``(C, ...)`` numpy arrays (per source cell)

and exposes conveniences that mirror :mod:`colight.omf_loader`:

- :meth:`VTPPolyData.mesh`          -> ``scene3d.Mesh`` (triangulated polys)
- :meth:`VTPPolyData.line_segments` -> ``scene3d.LineSegments`` (polylines split
  into endpoint pairs; per-cell data repeated per segment — the drillhole case)
- :meth:`VTPPolyData.point_cloud`   -> ``scene3d.PointCloud``

Like the OMF conveniences, ``mesh``/``line_segments``/``point_cloud`` accept a
``color_by`` (a point- or cell-data attribute name) resolved through scene3d's
first-class colormap support, so scalars get proper colormaps AND legends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from colight import colormaps, scene3d

PathLike = Union[str, "Any"]

_INSTALL_HINT = (
    "Loading VTP files requires 'pyvista' (preferred) or 'vtk'. "
    "Install with: pip install colight[geo]"
)


def _color_by_spec(
    values: np.ndarray,
    cmap: str,
    domain: Optional[Sequence[float]],
    label: Optional[str],
) -> scene3d.ColorBy:
    spec: scene3d.ColorBy = {"values": values, "cmap": cmap}
    if domain is not None:
        spec["domain"] = domain
    if label is not None:
        spec["label"] = label
    return spec


@dataclass
class VTPPolyData:
    """A VTK PolyData normalized to numpy, with scene3d conveniences.

    Polygon cells (``polys``) and polyline cells (``lines``) are kept as lists
    of variable-length vertex-index arrays so triangle strips, quads, and
    multi-vertex polylines all survive as-is until a convenience triangulates
    or splits them.
    """

    points: np.ndarray  # (N, 3) float32
    polys: List[np.ndarray] = field(default_factory=list)  # polygon cells
    lines: List[np.ndarray] = field(default_factory=list)  # polyline cells
    point_data: Dict[str, np.ndarray] = field(default_factory=dict)
    cell_data: Dict[str, np.ndarray] = field(default_factory=dict)
    # Optional backend-precomputed (M, 3) triangle indices. When a PolyData
    # contains triangle strips (whose fan-triangulation would be wrong), the
    # pyvista backend triangulates with correct strip semantics and stores the
    # result here; ``mesh()`` prefers it over fan-triangulating ``polys``.
    triangles: Optional[np.ndarray] = None

    # -- mesh --------------------------------------------------------------
    def mesh(
        self,
        color_by: Optional[str] = None,
        cmap: str = "viridis",
        domain: Optional[Sequence[float]] = None,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> scene3d.SceneComponent:
        """Build a ``scene3d.Mesh`` from triangulated polygon cells.

        Non-triangle polys (quads, strips, n-gons) are fan-triangulated. A
        ``scene3d.Mesh`` colors *per vertex*, so ``color_by`` must name a
        point-data attribute; its scalars are mapped through the colormap and
        passed as ``vertex_colors``.

        Args:
            color_by: Point-data attribute name to color vertices by.
            cmap: Colormap name (see :mod:`colight.colormaps`).
            domain: Colormap (min, max); None derives it from the values.
            label: Unused for meshes (kept for a uniform signature).
            **kwargs: Forwarded to ``scene3d.Mesh``. ``cull_mode`` defaults to
                "none" since surfaces are viewed from both sides.
        """
        triangles = (
            np.asarray(self.triangles)
            if self.triangles is not None
            else _triangulate(self.polys)
        )
        kwargs.setdefault("cull_mode", "none")
        if color_by is not None:
            if color_by not in self.point_data:
                raise KeyError(
                    f"mesh(color_by={color_by!r}) needs a point-data attribute; "
                    f"have {sorted(self.point_data)}"
                )
            values = np.asarray(self.point_data[color_by]).reshape(-1)
            kwargs.setdefault(
                "vertex_colors",
                colormaps.apply_colormap(values, cmap=cmap, domain=domain),
            )
        return scene3d.Mesh(
            positions=self.points.astype(np.float32),
            indices=triangles.astype(np.uint32),
            **kwargs,
        )

    # -- line segments -----------------------------------------------------
    def line_segments(
        self,
        color_by: Optional[str] = None,
        cmap: str = "viridis",
        domain: Optional[Sequence[float]] = None,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> scene3d.SceneComponent:
        """Build a ``scene3d.LineSegments`` from polyline cells.

        Each polyline with ``k`` points is split into ``k - 1`` endpoint-pair
        segments. Cell data is **repeated per segment**: a polyline emitting
        ``k - 1`` segments repeats its single cell value ``k - 1`` times, so a
        length-C cell attribute expands to a length-(sum of segments) array
        aligned one-to-one with the emitted segments (the drillhole-interval
        case). ``color_by`` may name either a cell-data attribute (expanded per
        segment) or a point-data attribute (sampled at each segment start).

        Args:
            color_by: Cell- or point-data attribute name to color segments by.
            cmap: Colormap name (see :mod:`colight.colormaps`).
            domain: Colormap (min, max); None derives it from the values.
            label: Legend label; defaults to the attribute name.
            **kwargs: Forwarded to ``scene3d.LineSegments`` (size, sizes, ...).
        """
        starts, ends, seg_cell_index, seg_start_point = _split_polylines(self.lines)
        if color_by is not None:
            values = self._segment_values(color_by, seg_cell_index, seg_start_point)
            kwargs.setdefault(
                "color_by",
                _color_by_spec(values, cmap, domain, label or color_by),
            )
        return scene3d.LineSegments(
            starts=self.points[starts].astype(np.float32),
            ends=self.points[ends].astype(np.float32),
            **kwargs,
        )

    def segment_cell_values(self, key: str) -> np.ndarray:
        """Expand a cell attribute to one value per emitted line segment.

        Repeats each polyline cell's value across the ``k - 1`` segments it
        emits, in the same order :meth:`line_segments` produces them. Useful for
        building a per-segment ``sizes`` (thickness) array from cell data.
        """
        _s, _e, seg_cell_index, _sp = _split_polylines(self.lines)
        return np.asarray(self.cell_data[key])[seg_cell_index]

    def _segment_values(
        self,
        key: str,
        seg_cell_index: np.ndarray,
        seg_start_point: np.ndarray,
    ) -> np.ndarray:
        if key in self.cell_data:
            return np.asarray(self.cell_data[key]).reshape(-1)[seg_cell_index]
        if key in self.point_data:
            return np.asarray(self.point_data[key]).reshape(-1)[seg_start_point]
        raise KeyError(
            f"line_segments(color_by={key!r}) not found in cell_data "
            f"{sorted(self.cell_data)} or point_data {sorted(self.point_data)}"
        )

    # -- point cloud -------------------------------------------------------
    def point_cloud(
        self,
        color_by: Optional[str] = None,
        cmap: str = "viridis",
        domain: Optional[Sequence[float]] = None,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> scene3d.SceneComponent:
        """Build a ``scene3d.PointCloud`` from all vertices.

        Args:
            color_by: Point-data attribute name to color points by.
            cmap: Colormap name (see :mod:`colight.colormaps`).
            domain: Colormap (min, max); None derives it from the values.
            label: Legend label; defaults to the attribute name.
            **kwargs: Forwarded to ``scene3d.PointCloud`` (size, ...).
        """
        if color_by is not None:
            if color_by not in self.point_data:
                raise KeyError(
                    f"point_cloud(color_by={color_by!r}) needs a point-data "
                    f"attribute; have {sorted(self.point_data)}"
                )
            values = np.asarray(self.point_data[color_by]).reshape(-1)
            kwargs.setdefault(
                "color_by",
                _color_by_spec(values, cmap, domain, label or color_by),
            )
        return scene3d.PointCloud(centers=self.points.astype(np.float32), **kwargs)


def _triangulate(polys: List[np.ndarray]) -> np.ndarray:
    """Fan-triangulate a list of polygon cells into an ``(M, 3)`` index array."""
    tris: List[List[int]] = []
    for cell in polys:
        c = np.asarray(cell).reshape(-1)
        for i in range(1, len(c) - 1):
            tris.append([int(c[0]), int(c[i]), int(c[i + 1])])
    if not tris:
        return np.zeros((0, 3), dtype=np.uint32)
    return np.asarray(tris, dtype=np.uint32)


def _split_polylines(lines: List[np.ndarray]):
    """Split polyline cells into segment endpoint index arrays.

    Returns ``(starts, ends, seg_cell_index, seg_start_point)`` where ``starts``
    and ``ends`` are point-index arrays (one per emitted segment),
    ``seg_cell_index`` maps each segment back to its source cell (for repeating
    cell data), and ``seg_start_point`` is the segment's start vertex index (for
    sampling point data).
    """
    starts: List[int] = []
    ends: List[int] = []
    seg_cell_index: List[int] = []
    for cell_idx, cell in enumerate(lines):
        c = np.asarray(cell).reshape(-1)
        for i in range(len(c) - 1):
            starts.append(int(c[i]))
            ends.append(int(c[i + 1]))
            seg_cell_index.append(cell_idx)
    starts_arr = np.asarray(starts, dtype=np.int64)
    return (
        starts_arr,
        np.asarray(ends, dtype=np.int64),
        np.asarray(seg_cell_index, dtype=np.int64),
        starts_arr,  # seg_start_point == segment start vertex
    )


# =============================================================================
# Backend readers
# =============================================================================


def _cells_from_connectivity(conn: np.ndarray) -> List[np.ndarray]:
    """Split a VTK legacy connectivity array ``[n, i0..i(n-1), n, ...]``."""
    cells: List[np.ndarray] = []
    i = 0
    conn = np.asarray(conn).reshape(-1)
    while i < len(conn):
        n = int(conn[i])
        cells.append(conn[i + 1 : i + 1 + n].astype(np.int64))
        i += 1 + n
    return cells


def _load_with_pyvista(path: str) -> VTPPolyData:
    import pyvista as pv
    from typing import cast

    # A .vtp always reads back as PolyData; cast so the PolyData-only API
    # (.faces/.lines/.strips/.triangulate) type-checks.
    poly = cast("pv.PolyData", pv.read(str(path)))
    points = np.asarray(poly.points, dtype=np.float32).reshape(-1, 3)

    faces = np.asarray(poly.faces).reshape(-1)
    polys = _cells_from_connectivity(faces) if faces.size else []
    line_conn = np.asarray(poly.lines).reshape(-1)
    lines = _cells_from_connectivity(line_conn) if line_conn.size else []

    # Triangle strips need real strip-triangulation (a fan is wrong). Let
    # pyvista's .triangulate() produce correct triangles over the same points;
    # store them so mesh() uses them directly. (Only when strips are present —
    # plain polygons fan-triangulate fine and we avoid the copy.)
    triangles = None
    strips = np.asarray(getattr(poly, "strips", np.empty(0))).reshape(-1)
    if strips.size:
        tri = poly.triangulate()
        tri_faces = np.asarray(tri.faces).reshape(-1)
        triangles = np.asarray(
            [c for c in _cells_from_connectivity(tri_faces)], dtype=np.uint32
        )
        # Keep raw polygon cells too (strips are represented via `triangles`).

    point_data = {k: np.asarray(poly.point_data[k]) for k in poly.point_data.keys()}
    cell_data = {k: np.asarray(poly.cell_data[k]) for k in poly.cell_data.keys()}

    return VTPPolyData(
        points=points,
        polys=polys,
        lines=lines,
        point_data=point_data,
        cell_data=cell_data,
        triangles=triangles,
    )


def _load_with_vtk(path: str) -> VTPPolyData:
    import vtk  # type: ignore[import-untyped]
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore[import-untyped]

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()

    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float32).reshape(-1, 3)

    def _conn(cell_array) -> List[np.ndarray]:
        if cell_array is None or cell_array.GetNumberOfCells() == 0:
            return []
        return _cells_from_connectivity(vtk_to_numpy(cell_array.GetData()))

    polys = _conn(poly.GetPolys()) + _conn(poly.GetStrips())
    lines = _conn(poly.GetLines())

    def _attrs(data) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for i in range(data.GetNumberOfArrays()):
            arr = data.GetArray(i)
            if arr is None:
                continue
            name = data.GetArrayName(i) or f"array_{i}"
            out[name] = vtk_to_numpy(arr)
        return out

    return VTPPolyData(
        points=points,
        polys=polys,
        lines=lines,
        point_data=_attrs(poly.GetPointData()),
        cell_data=_attrs(poly.GetCellData()),
    )


def load_vtp(path: PathLike) -> VTPPolyData:
    """Load a VTP (VTK XML PolyData) file into a numpy-backed record.

    Tries ``pyvista`` first, then falls back to ``vtk``; raises a clear
    :class:`ImportError` naming the extra to install if neither is present.

    Args:
        path: Path to a ``.vtp`` file.

    Returns:
        A :class:`VTPPolyData` with points, poly/line connectivity, and
        point/cell data, plus ``.mesh()/.line_segments()/.point_cloud()``
        conveniences.

    Raises:
        ImportError: If neither ``pyvista`` nor ``vtk`` is installed.
    """
    try:
        import pyvista  # noqa: F401
    except ImportError:
        pyvista = None  # type: ignore[assignment]

    if pyvista is not None:
        return _load_with_pyvista(str(path))

    try:
        import vtk  # noqa: F401
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e

    return _load_with_vtk(str(path))
