"""Load OMF (Open Mining Format) v1 projects into colight scene3d components.

Requires the optional ``omf`` dependency (``pip install colight[geo]``).

OMF projects use real-world (often UTM) coordinates in the hundreds of
kilometres, which exceed float32 precision on the GPU. ``load_omf`` keeps
geometry in those true world coordinates and returns the bounds midpoint on
``OMFProject.center``; pass it as ``scene3d.Scene(origin=project.center)`` and
the shift into float32-safe range happens once, at the Scene serialization
boundary (``pick-at`` adds it back, so reported positions stay in world
coordinates).

Element mapping:

- ``PointSetElement``  -> :class:`OMFPointSet`  -> ``scene3d.PointCloud``
- ``LineSetElement``   -> :class:`OMFLineSet`   -> ``scene3d.LineSegments``
- ``SurfaceElement``   -> :class:`OMFSurface`   -> ``scene3d.Mesh``
- ``VolumeElement`` (regular grid) -> :class:`OMFGridVolume` -> ``scene3d.Cuboid``

Scalar attributes are colored through scene3d's first-class ``color_by``
prop (see :mod:`colight.colormaps`), so drillhole traces and block models
get proper colormaps AND legends for free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from colight import scene3d

PathLike = Union[str, "Any"]


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
class OMFPointSet:
    """An OMF point set (e.g. drillhole collars) in world coordinates."""

    name: str
    vertices: np.ndarray  # (N, 3) float64, world coordinates
    attributes: Dict[str, np.ndarray] = field(default_factory=dict)

    def point_cloud(self, **kwargs: Any) -> scene3d.SceneComponent:
        """Build a scene3d PointCloud from the vertices.

        Args:
            **kwargs: Forwarded to ``scene3d.PointCloud`` (color, size, ...).
        """
        return scene3d.PointCloud(centers=self.vertices.astype(np.float32), **kwargs)


@dataclass
class OMFLineSet:
    """An OMF line set (drillhole traces with assay data) in world coordinates."""

    name: str
    vertices: np.ndarray  # (N, 3) float64, world coordinates
    segments: np.ndarray  # (M, 2) int vertex indices
    vertex_attributes: Dict[str, np.ndarray] = field(default_factory=dict)
    segment_attributes: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def starts(self) -> np.ndarray:
        """(M, 3) start position of each segment."""
        return self.vertices[self.segments[:, 0]]

    @property
    def ends(self) -> np.ndarray:
        """(M, 3) end position of each segment."""
        return self.vertices[self.segments[:, 1]]

    def line_segments(
        self,
        color_by: Optional[str] = None,
        cmap: str = "viridis",
        domain: Optional[Sequence[float]] = None,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> scene3d.SceneComponent:
        """Build a scene3d LineSegments component.

        Args:
            color_by: Name of a segment attribute to color by (mapped
                through ``scene3d``'s colormap support, with a legend).
            cmap: Colormap name (see :mod:`colight.colormaps`).
            domain: Colormap (min, max); None derives it from the values.
            label: Legend label; defaults to the attribute name.
            **kwargs: Forwarded to ``scene3d.LineSegments`` (size, color, ...).
        """
        if color_by is not None:
            values = self.segment_attributes[color_by]
            kwargs.setdefault(
                "color_by",
                _color_by_spec(values, cmap, domain, label or color_by),
            )
        return scene3d.LineSegments(
            starts=self.starts.astype(np.float32),
            ends=self.ends.astype(np.float32),
            **kwargs,
        )


@dataclass
class OMFSurface:
    """An OMF triangulated surface (topography, geology) in world coordinates."""

    name: str
    vertices: np.ndarray  # (N, 3) float64, world coordinates
    triangles: np.ndarray  # (M, 3) int vertex indices
    vertex_attributes: Dict[str, np.ndarray] = field(default_factory=dict)

    def mesh(self, **kwargs: Any) -> scene3d.SceneComponent:
        """Build a scene3d Mesh from the triangulation.

        Args:
            **kwargs: Forwarded to ``scene3d.Mesh`` (color, alpha via
                decorations, cull_mode, ...). ``cull_mode`` defaults to
                "none" since geological surfaces are viewed from both sides.
        """
        kwargs.setdefault("cull_mode", "none")
        # Geometry is already in scene coordinates; scene3d.Mesh defaults to a
        # single instance at the origin, so no center is needed.
        return scene3d.Mesh(
            positions=self.vertices.astype(np.float32),
            indices=self.triangles.astype(np.uint32),
            **kwargs,
        )


@dataclass
class OMFGridVolume:
    """An OMF regular-grid block model in world coordinates.

    Cell attribute arrays are stored flat in OMF order: C-contiguous over
    ``(nu, nv, nw)``, i.e. the w (z) index varies fastest.
    """

    name: str
    corner: np.ndarray  # (3,) world-coordinate min corner of the grid
    axes: np.ndarray  # (3, 3) rows = axis_u, axis_v, axis_w unit vectors
    tensors: Tuple[np.ndarray, np.ndarray, np.ndarray]  # cell widths along u, v, w
    cell_attributes: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Number of cells along (u, v, w)."""
        return (len(self.tensors[0]), len(self.tensors[1]), len(self.tensors[2]))

    def grid(self, key: str) -> np.ndarray:
        """Return a cell attribute reshaped to ``(nu, nv, nw)``."""
        return self.cell_attributes[key].reshape(self.shape)

    def cell_centers(self) -> np.ndarray:
        """Compute all cell centers as an ``(nu*nv*nw, 3)`` array in OMF order."""
        offsets = [np.cumsum(t) - t / 2.0 for t in self.tensors]
        u, v, w = np.meshgrid(*offsets, indexing="ij")
        local = np.stack([u, v, w], axis=-1).reshape(-1, 3)
        return self.corner + local @ self.axes

    def cuboids(
        self,
        color_by: str,
        cutoff: Optional[float] = None,
        stride: int = 1,
        cmap: str = "viridis",
        domain: Optional[Sequence[float]] = None,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> scene3d.SceneComponent:
        """Build a scene3d Cuboid instance layer for cells passing a cutoff.

        Args:
            color_by: Cell attribute to color by (mapped through
                ``scene3d``'s colormap support, with a legend; also used
                for the cutoff test).
            cutoff: Keep only cells with ``value >= cutoff``. None keeps all.
            stride: Subsample the grid, keeping every ``stride``-th cell
                along each axis (before the cutoff filter).
            cmap: Colormap name (see :mod:`colight.colormaps`).
            domain: Colormap (min, max); None spans ``cutoff`` (when given)
                to the max surviving value.
            label: Legend label; defaults to the attribute name.
            **kwargs: Forwarded to ``scene3d.Cuboid``.

        Returns:
            A Cuboid component with one instance per surviving cell.
        """
        centers, values = self.filtered_cells(color_by, cutoff=cutoff, stride=stride)
        half = np.array([t[0] for t in self.tensors], dtype=np.float32) / 2.0
        if domain is None and cutoff is not None and values.size:
            domain = (float(cutoff), float(np.nanmax(values)))
        kwargs.setdefault(
            "color_by",
            _color_by_spec(values, cmap, domain, label or color_by),
        )
        kwargs.setdefault("half_size", half.tolist())
        return scene3d.Cuboid(centers=centers.astype(np.float32), **kwargs)

    def cuboids_filter_by(
        self,
        color_by: str,
        min: Any,
        base_cutoff: Optional[float] = None,
        stride: int = 1,
        cmap: str = "viridis",
        domain: Optional[Sequence[float]] = None,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> scene3d.SceneComponent:
        """Build ONE Cuboid layer whose visible cells are chosen at render time.

        Unlike :meth:`cuboids` (which bakes a fixed cutoff into the geometry),
        this uploads every surviving cell once and attaches a per-instance
        ``filter_by`` so a ``$state`` slider raises/lowers the grade cutoff
        client-side with no Python round-trip and no re-upload.

        Args:
            color_by: Cell attribute to color by and filter on.
            min: The ``filter_by`` lower threshold — typically a
                ``Plot.js("$state.cutoff")`` reference (or a literal).
            base_cutoff: Keep only cells with ``value >= base_cutoff`` in the
                uploaded set (bounds the payload). None uploads all cells.
            stride: Subsample the grid (every ``stride``-th cell per axis).
            cmap: Colormap name.
            domain: Colormap (min, max); defaults to (base_cutoff, max value).
            label: Legend / filter label; defaults to the attribute name.
            **kwargs: Forwarded to ``scene3d.Cuboid``.

        Returns:
            A single Cuboid component carrying all cells + a ``filter_by``.
        """
        centers, values = self.filtered_cells(
            color_by, cutoff=base_cutoff, stride=stride
        )
        half = np.array([t[0] for t in self.tensors], dtype=np.float32) / 2.0
        if domain is None and values.size:
            lo = (
                float(base_cutoff)
                if base_cutoff is not None
                else float(np.nanmin(values))
            )
            domain = (lo, float(np.nanmax(values)))
        kwargs.setdefault(
            "color_by",
            _color_by_spec(values, cmap, domain, label or color_by),
        )
        kwargs.setdefault("half_size", half.tolist())
        return scene3d.Cuboid(
            centers=centers.astype(np.float32),
            filter_by={"values": values, "min": min, "label": label or color_by},
            **kwargs,
        )

    def filtered_cells(
        self,
        key: str,
        cutoff: Optional[float] = None,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select cell centers and values by stride and cutoff.

        Args:
            key: Cell attribute to filter on.
            cutoff: Keep only cells with ``value >= cutoff``. None keeps all.
            stride: Keep every ``stride``-th cell along each axis.

        Returns:
            Tuple of ``(centers, values)`` where centers is (N, 3) and values
            is (N,) for the surviving cells.
        """
        values = self.grid(key)
        centers = self.cell_centers().reshape(self.shape + (3,))
        if stride > 1:
            values = values[::stride, ::stride, ::stride]
            centers = centers[::stride, ::stride, ::stride]
        values = values.reshape(-1)
        centers = centers.reshape(-1, 3)
        if cutoff is not None:
            mask = values >= cutoff
            values = values[mask]
            centers = centers[mask]
        return centers, values


@dataclass
class OMFProject:
    """An OMF project in true world coordinates.

    Geometry keeps its original (often UTM) coordinates. ``center`` is the
    midpoint of the project bounds — the recommended value to pass to
    ``scene3d.Scene(origin=project.center)`` so positions are shifted into
    float32-safe range at the single serialization boundary (see the Scene
    ``origin`` prop). ``pick-at`` adds it back, so dereferenced positions come
    out in true world coordinates.
    """

    name: str
    center: np.ndarray  # (3,) bounds midpoint; use as Scene(origin=...)
    points: Dict[str, OMFPointSet] = field(default_factory=dict)
    line_sets: Dict[str, OMFLineSet] = field(default_factory=dict)
    surfaces: Dict[str, OMFSurface] = field(default_factory=dict)
    volumes: Dict[str, OMFGridVolume] = field(default_factory=dict)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min, max) world-coordinate bounds over point/line/surface vertices."""
        stacks = [
            e.vertices
            for group in (self.points, self.line_sets, self.surfaces)
            for e in group.values()
        ]
        allv = np.vstack(stacks)
        return allv.min(axis=0), allv.max(axis=0)


def _attribute_arrays(element: Any, location: str) -> Dict[str, np.ndarray]:
    """Collect an OMF element's data arrays for one location ('vertices'/'segments'/'cells')."""
    out: Dict[str, np.ndarray] = {}
    for data in getattr(element, "data", []):
        if data.location == location:
            out[data.name] = np.asarray(data.array.array)
    return out


def load_omf(path: PathLike) -> OMFProject:
    """Load an OMF v1 file into numpy-backed element records (world coords).

    Geometry keeps its original coordinates; the bounds midpoint is returned
    on ``OMFProject.center`` for use as ``scene3d.Scene(origin=...)``. Shifting
    happens once, at the Scene serialization boundary — no loader-side
    re-centering.

    Args:
        path: Path to a ``.omf`` file (OMF v1).

    Returns:
        An :class:`OMFProject` with elements grouped by type, keyed by name.

    Raises:
        ImportError: If the ``omf`` package is not installed.
    """
    try:
        import omf
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Loading OMF files requires the 'omf' package. "
            "Install it with: pip install colight[geo]"
        ) from e

    project = omf.OMFReader(str(path)).get_project()
    project_origin = np.asarray(project.origin, dtype=np.float64)

    points: Dict[str, OMFPointSet] = {}
    line_sets: Dict[str, OMFLineSet] = {}
    surfaces: Dict[str, OMFSurface] = {}
    volumes: Dict[str, OMFGridVolume] = {}

    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)

    def track(v: np.ndarray) -> None:
        nonlocal lo, hi
        lo = np.minimum(lo, v.min(axis=0))
        hi = np.maximum(hi, v.max(axis=0))

    for element in project.elements:
        geometry = element.geometry
        origin = project_origin + np.asarray(geometry.origin, dtype=np.float64)
        kind = type(geometry).__name__
        if kind == "PointSetGeometry":
            vertices = np.asarray(geometry.vertices.array, dtype=np.float64) + origin
            track(vertices)
            points[element.name] = OMFPointSet(
                name=element.name,
                vertices=vertices,
                attributes=_attribute_arrays(element, "vertices"),
            )
        elif kind == "LineSetGeometry":
            vertices = np.asarray(geometry.vertices.array, dtype=np.float64) + origin
            track(vertices)
            line_sets[element.name] = OMFLineSet(
                name=element.name,
                vertices=vertices,
                segments=np.asarray(geometry.segments.array),
                vertex_attributes=_attribute_arrays(element, "vertices"),
                segment_attributes=_attribute_arrays(element, "segments"),
            )
        elif kind == "SurfaceGeometry":
            vertices = np.asarray(geometry.vertices.array, dtype=np.float64) + origin
            track(vertices)
            surfaces[element.name] = OMFSurface(
                name=element.name,
                vertices=vertices,
                triangles=np.asarray(geometry.triangles.array),
                vertex_attributes=_attribute_arrays(element, "vertices"),
            )
        elif kind == "VolumeGridGeometry":
            tensors = tuple(
                np.asarray(getattr(geometry, f"tensor_{axis}"), dtype=np.float64)
                for axis in "uvw"
            )
            axes = np.asarray(
                [geometry.axis_u, geometry.axis_v, geometry.axis_w], dtype=np.float64
            )
            extent = axes.T @ np.array([t.sum() for t in tensors])
            track(np.stack([origin, origin + extent]))
            volumes[element.name] = OMFGridVolume(
                name=element.name,
                corner=origin,
                axes=axes,
                tensors=(tensors[0], tensors[1], tensors[2]),
                cell_attributes=_attribute_arrays(element, "cells"),
            )
        # Other geometry kinds (e.g. VolumeGridGeometry variants we don't
        # know) are skipped; Wolfpass only contains the four above.

    # Geometry stays in world coordinates; ``center`` is the suggested Scene
    # origin (bounds midpoint), applied once at the Scene boundary.
    center = (lo + hi) / 2.0 if np.all(np.isfinite(lo)) else np.zeros(3)

    return OMFProject(
        name=project.name,
        center=center,
        points=points,
        line_sets=line_sets,
        surfaces=surfaces,
        volumes=volumes,
    )
