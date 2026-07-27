"""Tests for per-vertex weighted transform references on scene3d Mesh.

A vertex can be positioned by a weighted combination of named Group transforms
rather than rigidly by one. These tests cover the Python surface: validation of
the three fields (they arrive together, shapes agree, K in range, weights
usable), row normalization, and serialization into the geometry payload.

The GPU-side blend, name resolution and buffer lifecycle are covered by the JS
suite (packages/colight/tests/js/scene3d/transform-refs.test.tsx).
"""

import numpy as np
import pytest

import colight.scene3d as scene3d

# A 4-vertex strip is enough for every shape assertion here.
POSITIONS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    dtype=np.float32,
)
N = len(POSITIONS)
INDICES = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)

RAMP = np.linspace(0.0, 1.0, N, dtype=np.float32)
SLOTS = np.tile([0, 1], (N, 1))
WEIGHTS = np.stack([1.0 - RAMP, RAMP], axis=1)


def mesh(**kwargs):
    return scene3d.Mesh(positions=POSITIONS, indices=INDICES, **kwargs)


def geometry_of(component):
    return component.props["geometry"]


# =============================================================================
# Serialization: the three fields reach the geometry payload as arrays
# =============================================================================


def test_transform_refs_serialize_into_geometry():
    geo = geometry_of(
        mesh(
            transform_refs=["footwall", "hangingwall"],
            transform_indices=SLOTS,
            transform_weights=WEIGHTS,
        )
    )

    assert geo["transform_refs"] == ["footwall", "hangingwall"]
    # Indices and weights ship flattened, row-major, K per vertex.
    assert geo["transform_indices"].shape == (N * 2,)
    assert geo["transform_weights"].shape == (N * 2,)
    assert geo["transform_indices"].dtype == np.float32
    assert geo["transform_weights"].dtype == np.float32
    np.testing.assert_allclose(
        geo["transform_weights"].reshape(N, 2), WEIGHTS, atol=1e-6
    )


def test_transform_refs_renamed_for_js_boundary():
    # Geometry-adjacent data crosses the boundary camelCased, like the other
    # framework keys, so the JS side reads it without a coercion special case.
    props = scene3d._convert_to_js(
        mesh(
            transform_refs=["a", "b"],
            transform_indices=SLOTS,
            transform_weights=WEIGHTS,
        ).props
    )
    geo = props["geometry"]
    assert "transformRefs" in geo
    assert "transformIndices" in geo
    assert "transformWeights" in geo
    assert "transform_refs" not in geo


def test_arrays_participate_in_the_normal_ndarray_path():
    # The values are ndarrays on the component, so they travel the same
    # buffer/diff machinery every other geometry array uses - nothing bespoke.
    geo = geometry_of(
        mesh(
            transform_refs=["a", "b"],
            transform_indices=SLOTS,
            transform_weights=WEIGHTS,
        )
    )
    for key in ("positions", "transform_indices", "transform_weights"):
        assert isinstance(geo[key], np.ndarray), key


def test_mesh_without_refs_carries_no_transform_fields():
    geo = geometry_of(mesh())
    assert "transform_refs" not in geo
    assert "transform_indices" not in geo
    assert "transform_weights" not in geo


def test_single_reference_per_vertex_is_allowed():
    geo = geometry_of(
        mesh(
            transform_refs=["only"],
            transform_indices=np.zeros((N, 1), dtype=np.int32),
            transform_weights=np.ones((N, 1), dtype=np.float32),
        )
    )
    assert geo["transform_indices"].shape == (N,)


def test_one_dimensional_arrays_are_read_as_k_equals_one():
    geo = geometry_of(
        mesh(
            transform_refs=["only"],
            transform_indices=np.zeros(N, dtype=np.int32),
            transform_weights=np.ones(N, dtype=np.float32),
        )
    )
    assert geo["transform_indices"].shape == (N,)


# =============================================================================
# Validation
# =============================================================================


@pytest.mark.parametrize(
    "kwargs",
    [
        {"transform_refs": ["a", "b"]},
        {"transform_indices": SLOTS},
        {"transform_weights": WEIGHTS},
        {"transform_refs": ["a", "b"], "transform_indices": SLOTS},
        {"transform_indices": SLOTS, "transform_weights": WEIGHTS},
    ],
    ids=["refs", "indices", "weights", "refs+indices", "indices+weights"],
)
def test_partial_declaration_is_an_error(kwargs):
    with pytest.raises(ValueError, match="must be supplied together"):
        mesh(**kwargs)


def test_empty_refs_is_an_error():
    with pytest.raises(ValueError, match="at least one Group"):
        mesh(
            transform_refs=[],
            transform_indices=SLOTS,
            transform_weights=WEIGHTS,
        )


def test_duplicate_ref_names_are_an_error():
    with pytest.raises(ValueError, match="unique"):
        mesh(
            transform_refs=["a", "a"],
            transform_indices=SLOTS,
            transform_weights=WEIGHTS,
        )


def test_non_string_refs_are_an_error():
    with pytest.raises(ValueError, match="non-empty strings"):
        mesh(
            transform_refs=["a", 3],  # type: ignore[list-item]
            transform_indices=SLOTS,
            transform_weights=WEIGHTS,
        )


def test_index_and_weight_shape_mismatch_is_an_error():
    with pytest.raises(ValueError, match="must match"):
        mesh(
            transform_refs=["a", "b"],
            transform_indices=SLOTS,
            transform_weights=np.ones((N, 3), dtype=np.float32) / 3.0,
        )


def test_row_count_must_equal_vertex_count():
    with pytest.raises(ValueError, match=f"{N} vertices"):
        mesh(
            transform_refs=["a", "b"],
            transform_indices=np.tile([0, 1], (N + 2, 1)),
            transform_weights=np.full((N + 2, 2), 0.5, dtype=np.float32),
        )


def test_k_above_the_maximum_is_an_error():
    k = scene3d.MAX_TRANSFORM_REFS_PER_VERTEX + 1
    with pytest.raises(ValueError, match="references per"):
        mesh(
            transform_refs=[f"g{i}" for i in range(k)],
            transform_indices=np.tile(np.arange(k), (N, 1)),
            transform_weights=np.full((N, k), 1.0 / k, dtype=np.float32),
        )


def test_slot_outside_refs_is_an_error():
    with pytest.raises(ValueError, match="index transform_refs"):
        mesh(
            transform_refs=["a", "b"],
            transform_indices=np.tile([0, 5], (N, 1)),
            transform_weights=WEIGHTS,
        )


def test_negative_weights_are_an_error():
    weights = WEIGHTS.copy()
    weights[1, 0] = -0.5
    with pytest.raises(ValueError, match="non-negative"):
        mesh(
            transform_refs=["a", "b"],
            transform_indices=SLOTS,
            transform_weights=weights,
        )


def test_all_zero_weight_row_is_a_hard_error():
    # An all-zero row would collapse that vertex to the origin - a silent
    # geometry corruption, so it is refused rather than normalized.
    weights = WEIGHTS.copy()
    weights[2] = [0.0, 0.0]
    with pytest.raises(ValueError, match="non-zero weight"):
        mesh(
            transform_refs=["a", "b"],
            transform_indices=SLOTS,
            transform_weights=weights,
        )


def test_rows_off_by_more_than_tolerance_warn_and_normalize():
    weights = np.full((N, 2), 0.25, dtype=np.float32)  # rows sum to 0.5
    with pytest.warns(UserWarning, match="do not sum to 1"):
        geo = geometry_of(
            mesh(
                transform_refs=["a", "b"],
                transform_indices=SLOTS,
                transform_weights=weights,
            )
        )
    np.testing.assert_allclose(
        geo["transform_weights"].reshape(N, 2).sum(axis=1), 1.0, atol=1e-6
    )


def test_rows_within_tolerance_do_not_warn():
    weights = WEIGHTS.copy()
    weights[0] = [0.5, 0.5005]  # 5e-4 off, inside the 1e-3 tolerance
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        geo = geometry_of(
            mesh(
                transform_refs=["a", "b"],
                transform_indices=SLOTS,
                transform_weights=weights,
            )
        )
    # Normalization still applies, so the shipped rows sum to exactly 1.
    np.testing.assert_allclose(
        geo["transform_weights"].reshape(N, 2).sum(axis=1), 1.0, atol=1e-6
    )
