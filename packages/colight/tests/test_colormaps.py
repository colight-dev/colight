"""Tests for colight.colormaps: value->RGB mapping and legend metadata."""

import numpy as np
import pytest

from colight import colormaps

# Matplotlib viridis anchor values (exact at t=0, 0.5, 1).
VIRIDIS_LO = [0.267004, 0.004874, 0.329415]
VIRIDIS_MID = [0.127568, 0.566949, 0.550556]
VIRIDIS_HI = [0.993248, 0.906157, 0.143936]


class TestApplyColormapContinuous:
    def test_viridis_anchor_values(self):
        rgb = colormaps.apply_colormap([0.0, 0.5, 1.0], "viridis", domain=(0, 1))
        assert rgb.dtype == np.float32
        assert rgb.shape == (3, 3)
        np.testing.assert_allclose(rgb[0], VIRIDIS_LO, atol=1e-6)
        np.testing.assert_allclose(rgb[1], VIRIDIS_MID, atol=1e-6)
        np.testing.assert_allclose(rgb[2], VIRIDIS_HI, atol=1e-6)

    def test_all_continuous_maps_produce_valid_rgb(self):
        values = np.linspace(-1, 2, 7)
        for name in colormaps.CONTINUOUS_CMAPS:
            rgb = colormaps.apply_colormap(values, name, domain=(0, 1))
            assert rgb.shape == (7, 3)
            assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)

    def test_domain_clamping(self):
        rgb = colormaps.apply_colormap([-10.0, 0.0, 5.0, 20.0], domain=(0, 5))
        np.testing.assert_allclose(rgb[0], rgb[1], atol=1e-7)  # clamped low
        np.testing.assert_allclose(rgb[2], rgb[3], atol=1e-7)  # clamped high
        np.testing.assert_allclose(rgb[0], VIRIDIS_LO, atol=1e-6)
        np.testing.assert_allclose(rgb[3], VIRIDIS_HI, atol=1e-6)

    def test_domain_derived_from_values(self):
        rgb = colormaps.apply_colormap([2.0, 4.0])
        np.testing.assert_allclose(rgb[0], VIRIDIS_LO, atol=1e-6)
        np.testing.assert_allclose(rgb[1], VIRIDIS_HI, atol=1e-6)

    def test_nan_gets_nan_color(self):
        rgb = colormaps.apply_colormap([0.0, np.nan, 1.0], domain=(0, 1))
        np.testing.assert_allclose(rgb[1], [0.5, 0.5, 0.5], atol=1e-7)

    def test_custom_nan_color(self):
        rgb = colormaps.apply_colormap([np.nan], domain=(0, 1), nan_color=(1, 0, 1))
        np.testing.assert_allclose(rgb[0], [1, 0, 1], atol=1e-7)

    def test_inf_treated_like_nan(self):
        rgb = colormaps.apply_colormap([np.inf, -np.inf], domain=(0, 1))
        np.testing.assert_allclose(rgb, [[0.5] * 3, [0.5] * 3], atol=1e-7)

    def test_constant_values_do_not_divide_by_zero(self):
        rgb = colormaps.apply_colormap([3.0, 3.0, 3.0])
        assert np.all(np.isfinite(rgb))

    def test_greys_runs_white_to_black(self):
        rgb = colormaps.apply_colormap([0.0, 1.0], "greys", domain=(0, 1))
        np.testing.assert_allclose(rgb[0], [1, 1, 1], atol=1e-6)
        np.testing.assert_allclose(rgb[1], [0, 0, 0], atol=1e-6)

    def test_coolwarm_is_diverging(self):
        rgb = colormaps.apply_colormap([0.0, 0.5, 1.0], "coolwarm", domain=(0, 1))
        assert rgb[0][2] > rgb[0][0]  # blue end
        assert rgb[2][0] > rgb[2][2]  # red end
        assert np.all(rgb[1] > 0.75)  # near-white middle

    def test_bad_domain_raises(self):
        with pytest.raises(ValueError, match="max > min"):
            colormaps.apply_colormap([0.0], domain=(1, 1))

    def test_unknown_cmap_raises(self):
        with pytest.raises(ValueError, match="unknown colormap"):
            colormaps.apply_colormap([0.0], "sunset")

    def test_non_1d_values_raise(self):
        with pytest.raises(ValueError, match="1D"):
            colormaps.apply_colormap(np.zeros((2, 2)))


class TestApplyColormapCategorical:
    def test_codes_index_palette(self):
        rgb = colormaps.apply_colormap([0, 1, 2], "tab10")
        np.testing.assert_allclose(
            rgb, colormaps.CATEGORICAL_CMAPS["tab10"][:3], atol=1e-6
        )

    def test_codes_wrap_modulo_palette(self):
        palette = colormaps.CATEGORICAL_CMAPS["tab10"]
        rgb = colormaps.apply_colormap([len(palette)], "tab10")
        np.testing.assert_allclose(rgb[0], palette[0], atol=1e-6)

    def test_invalid_codes_get_nan_color(self):
        rgb = colormaps.apply_colormap([np.nan, -1.0], "tab10")
        np.testing.assert_allclose(rgb, [[0.5] * 3, [0.5] * 3], atol=1e-7)

    def test_is_categorical(self):
        assert colormaps.is_categorical("tab10")
        assert colormaps.is_categorical("okabe_ito")
        assert not colormaps.is_categorical("viridis")


class TestColormapMetadata:
    def test_continuous_metadata(self):
        meta = colormaps.colormap_metadata("viridis", domain=(0, 2.5), label="Cu %")
        assert meta["cmap"] == "viridis"
        assert meta["categorical"] is False
        assert meta["domain"] == [0.0, 2.5]
        assert meta["label"] == "Cu %"
        stops = np.asarray(meta["stops"])
        assert stops.shape == (17, 3)
        np.testing.assert_allclose(stops[0], VIRIDIS_LO, atol=1e-6)
        np.testing.assert_allclose(stops[-1], VIRIDIS_HI, atol=1e-6)

    def test_categorical_metadata_cycles_colors_over_categories(self):
        palette = colormaps.CATEGORICAL_CMAPS["okabe_ito"]
        categories = [f"c{i}" for i in range(len(palette) + 2)]
        meta = colormaps.colormap_metadata("okabe_ito", categories=categories)
        assert meta["categorical"] is True
        assert meta["categories"] == categories
        assert len(meta["colors"]) == len(categories)
        np.testing.assert_allclose(meta["colors"][len(palette)], palette[0], atol=1e-6)
        assert "domain" not in meta and "stops" not in meta

    def test_domain_on_categorical_raises(self):
        with pytest.raises(ValueError, match="continuous"):
            colormaps.colormap_metadata("tab10", domain=(0, 1))

    def test_categories_on_continuous_raises(self):
        with pytest.raises(ValueError, match="categorical"):
            colormaps.colormap_metadata("viridis", categories=["a"])

    def test_metadata_is_json_safe(self):
        import json

        for kwargs in (
            {"cmap": "magma", "domain": (0, 1), "label": "x"},
            {"cmap": "tab10", "categories": ["a", "b"]},
        ):
            json.dumps(colormaps.colormap_metadata(**kwargs))


class TestResolveColorBy:
    def test_resolves_colors_and_metadata(self):
        colors, meta = colormaps.resolve_color_by(
            {"values": [0.0, 1.0, 2.0], "cmap": "viridis", "label": "t"}
        )
        assert colors.shape == (3, 3)
        assert meta["domain"] == [0.0, 2.0]  # derived and reported
        assert meta["label"] == "t"
        np.testing.assert_allclose(colors[0], VIRIDIS_LO, atol=1e-6)

    def test_defaults_to_viridis(self):
        _colors, meta = colormaps.resolve_color_by({"values": [0.0, 1.0]})
        assert meta["cmap"] == "viridis"

    def test_missing_values_raises(self):
        with pytest.raises(ValueError, match="values"):
            colormaps.resolve_color_by({"cmap": "viridis"})

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="unknown color_by keys"):
            colormaps.resolve_color_by({"values": [0.0], "vmax": 1.0})


class TestResolveCategorical:
    """First-class category tables (xmi id-maps idiom): {value, label, color?}."""

    CATS = [
        {"value": 0, "label": "not logged", "color": [0.5, 0.5, 0.5]},
        {"value": 1, "label": "Dacite"},
        {"value": 2, "label": "Andesite"},
    ]

    def test_declared_colors_and_auto_palette(self):
        colors, meta = colormaps.resolve_categorical(
            [0, 1, 2], self.CATS, label="Lithology"
        )
        assert colors.shape == (3, 3)
        # Declared grey for code 0.
        np.testing.assert_allclose(colors[0], [0.5, 0.5, 0.5], atol=1e-6)
        # Auto-palette (tab10 blue) for code 1.
        np.testing.assert_allclose(
            colors[1], colormaps.CATEGORICAL_CMAPS["tab10"][0], atol=1e-6
        )
        assert meta["categorical"] is True
        assert meta["cmap"] == "categorical"
        assert meta["label"] == "Lithology"
        assert [c["label"] for c in meta["categories"]] == [
            "not logged",
            "Dacite",
            "Andesite",
        ]
        assert [c["value"] for c in meta["categories"]] == [0, 1, 2]

    def test_nan_and_unmatched_use_fallback(self):
        colors, meta = colormaps.resolve_categorical([0, 7, float("nan")], self.CATS)
        # Default fallback is mid-grey.
        np.testing.assert_allclose(colors[1], [0.5, 0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(colors[2], [0.5, 0.5, 0.5], atol=1e-6)
        assert meta["fallback"]["label"] == "unmapped"

    def test_custom_fallback(self):
        _colors, meta = colormaps.resolve_categorical(
            [9], self.CATS, fallback={"label": "other", "color": [1.0, 0.0, 0.0]}
        )
        assert meta["fallback"]["label"] == "other"
        assert meta["fallback"]["color"] == [1.0, 0.0, 0.0]

    def test_arbitrary_value_codes(self):
        # Codes need not be 0..K-1 (e.g. sparse lithology ids).
        cats = [{"value": 10, "label": "a"}, {"value": 42, "label": "b"}]
        colors, _meta = colormaps.resolve_categorical([42, 10, 5], cats)
        # code 42 -> b's color, code 5 unmatched -> fallback grey.
        np.testing.assert_allclose(colors[2], [0.5, 0.5, 0.5], atol=1e-6)
        assert not np.allclose(colors[0], colors[1])

    def test_missing_label_raises(self):
        with pytest.raises(ValueError, match="label"):
            colormaps.resolve_categorical([0], [{"value": 0}])

    def test_resolve_color_by_routes_category_table(self):
        colors, meta = colormaps.resolve_color_by(
            {"values": [0, 1], "categories": self.CATS, "label": "L"}
        )
        assert meta["categorical"] is True
        assert "categories" in meta and isinstance(meta["categories"][0], dict)
        assert colors.shape == (2, 3)

    def test_domain_with_category_table_raises(self):
        with pytest.raises(ValueError, match="continuous"):
            colormaps.resolve_color_by(
                {"values": [0], "categories": self.CATS, "domain": (0, 1)}
            )


class TestContinuousLut:
    def test_shape_and_endpoints(self):
        lut = colormaps.continuous_lut("viridis")
        assert lut.shape == (256, 3)
        assert lut.dtype == np.float32
        np.testing.assert_allclose(lut[0], VIRIDIS_LO, atol=1e-6)
        np.testing.assert_allclose(lut[-1], VIRIDIS_HI, atol=1e-6)

    def test_rejects_categorical(self):
        with pytest.raises(ValueError, match="continuous"):
            colormaps.continuous_lut("tab10")


class TestResolveChannel:
    def test_continuous_channel_ships_lut(self):
        colors, legend, colorizer = colormaps.resolve_channel(
            {"values": [0.0, 1.0, 2.0], "cmap": "viridis", "domain": (0, 2)}
        )
        assert colors.shape == (3, 3)
        assert colorizer["kind"] == "continuous"
        assert len(colorizer["lut"]) == 256
        assert colorizer["domain"] == [0.0, 2.0]
        assert legend["categorical"] is False

    def test_categorical_channel_ships_table(self):
        _colors, legend, colorizer = colormaps.resolve_channel(
            {
                "values": [0, 1],
                "categories": [
                    {"value": 0, "label": "a"},
                    {"value": 1, "label": "b"},
                ],
            }
        )
        assert colorizer["kind"] == "categorical"
        assert [c["label"] for c in colorizer["categories"]] == ["a", "b"]
        assert legend["categorical"] is True
