"""Tests for Phase 3: Layer Panel & ROI Management."""
import pytest
import numpy as np

from montaris.layers import (
    ROILayer, ROI_COLORS, LayerStack, generate_unique_roi_name,
)


class TestExpandedROIColors:
    def test_20_colors(self):
        assert len(ROI_COLORS) == 20

    def test_all_valid_rgb(self):
        for c in ROI_COLORS:
            assert len(c) == 3
            for v in c:
                assert 0 <= v <= 255


class TestGenerateUniqueROIName:
    def test_no_conflict(self):
        layers = [ROILayer("ROI 1", 10, 10)]
        assert generate_unique_roi_name("ROI 2", layers) == "ROI 2"

    def test_conflict(self):
        layers = [ROILayer("ROI 1", 10, 10)]
        assert generate_unique_roi_name("ROI 1", layers) == "ROI 1 (2)"

    def test_multiple_conflicts(self):
        layers = [
            ROILayer("ROI 1", 10, 10),
            ROILayer("ROI 1 (2)", 10, 10),
        ]
        assert generate_unique_roi_name("ROI 1", layers) == "ROI 1 (3)"


class TestGlobalOpacityFactor:
    def test_default_factor(self):
        ls = LayerStack()
        assert ls._global_opacity_factor == 1.0

    def test_set_factor(self):
        ls = LayerStack()
        ls._global_opacity_factor = 0.5
        assert ls._global_opacity_factor == 0.5


class TestBothFillMode:
    def test_both_fill_mode_rgba(self):
        from montaris.canvas import _mask_to_rgba
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 255
        rgba = _mask_to_rgba(mask, (255, 0, 0), 128, "both")
        assert rgba.shape == (20, 20, 4)
        # Interior should have some alpha
        assert rgba[10, 10, 3] > 0
        # Edge should have higher alpha
        assert rgba[5, 10, 3] >= rgba[10, 10, 3]


class TestLayerPanelFeatures:
    def test_refresh_shows_pixel_count(self, qapp, app_with_image):
        app = app_with_image
        roi = app.layer_stack.roi_layers[0]
        roi.mask[5:10, 5:10] = 255
        app.layer_panel.refresh()
        item = app.layer_panel.list_widget.item(1)
        assert "px)" in item.text()

    def test_refresh_shows_roi_count(self, qapp, app_with_image):
        app = app_with_image
        app.layer_panel.refresh()
        assert "1 ROIs" in app.layer_panel.header.text()

    def test_refresh_shows_index(self, qapp, app_with_image):
        app = app_with_image
        app.layer_panel.refresh()
        item = app.layer_panel.list_widget.item(1)
        assert item.text().startswith("1.")

    def test_global_opacity_slider(self, qapp, app_with_image):
        app = app_with_image
        app.layer_panel.global_opacity_slider.setValue(50)
        assert app.layer_stack._global_opacity_factor == 0.5

    def test_show_all_toggle(self, qapp, app_with_image):
        app = app_with_image
        roi = app.layer_stack.roi_layers[0]
        app.layer_panel.show_all_cb.setChecked(False)
        assert not roi.visible
        app.layer_panel.show_all_cb.setChecked(True)
        assert roi.visible

    def test_nav_bar_exists(self, qapp, app_with_image):
        app = app_with_image
        assert app.layer_panel.nav_bar is not None

    def test_clear_all_btn_exists(self, qapp, app_with_image):
        assert app_with_image.layer_panel.clear_all_btn is not None

    def test_random_color_btn_exists(self, qapp, app_with_image):
        assert app_with_image.layer_panel.random_color_btn is not None
