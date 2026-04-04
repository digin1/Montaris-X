"""Headed tests for ROI boundary display features.

Tests the Fiji-like boundary system:
- Global fill mode applies to all ROIs
- Thick boundaries with zoom-independent appearance
- Yellow boundaries for all ROIs, cyan for active
- Solid mode hides boundaries (except active cyan)
- User-configurable thickness and colours

Usage:
    QT_QPA_PLATFORM= .venv/bin/pytest tests/test_boundary.py -m headed -s -v
"""
import os
import numpy as np
import pytest

from PySide6.QtWidgets import QApplication

pytestmark = pytest.mark.headed

SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "..", "screenshots")


def _save_screenshot(widget, name):
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    path = os.path.join(SCREENSHOT_DIR, f"{name}.png")
    widget.grab().save(path)
    print(f"  Screenshot saved: {path}")
    return path


class TestGlobalFillMode:
    """Fill mode should apply to all loaded ROIs, not just the selected one."""

    def test_fill_mode_is_global(self, app_with_real_image):
        window = app_with_real_image
        ls = window.layer_stack
        if not ls.roi_layers:
            pytest.skip("No ROIs available")

        # Default is solid
        assert ls.fill_mode == 'solid'

        # Change to boundary — should be global
        ls.fill_mode = 'boundary'
        window.canvas.refresh_overlays()
        QApplication.processEvents()

        # All ROIs should render with the same fill mode
        assert ls.fill_mode == 'boundary'

        _save_screenshot(window.canvas, "boundary_fill_all")

        # Restore
        ls.fill_mode = 'solid'
        window.canvas.refresh_overlays()
        QApplication.processEvents()

    def test_fill_mode_both(self, app_with_real_image):
        window = app_with_real_image
        ls = window.layer_stack

        ls.fill_mode = 'both'
        window.canvas.refresh_overlays()
        QApplication.processEvents()

        _save_screenshot(window.canvas, "boundary_fill_both")

        ls.fill_mode = 'solid'
        window.canvas.refresh_overlays()
        QApplication.processEvents()


class TestBoundaryColors:
    """All ROIs should have yellow boundaries; active ROI should have cyan."""

    def test_default_boundary_colors(self, app_with_real_image):
        window = app_with_real_image
        ls = window.layer_stack
        assert ls.boundary_color == (255, 255, 0), "Default boundary color should be yellow"
        assert ls.active_boundary_color == (0, 255, 255), "Active boundary color should be cyan"

    def test_active_roi_gets_cyan_boundary(self, app_with_real_image):
        window = app_with_real_image
        canvas = window.canvas
        ls = window.layer_stack
        rois = ls.roi_layers
        if not rois:
            pytest.skip("No ROIs available (deleted by earlier test)")

        ls.fill_mode = 'boundary'

        # Select first ROI
        roi = rois[0]
        canvas.set_active_layer(roi)
        canvas._selection.set([roi])
        QApplication.processEvents()

        canvas.refresh_overlays()
        QApplication.processEvents()

        _save_screenshot(canvas, "boundary_active_cyan")

        ls.fill_mode = 'solid'
        canvas.refresh_overlays()
        QApplication.processEvents()

    def test_custom_boundary_color(self, app_with_real_image):
        """User can change boundary colour."""
        window = app_with_real_image
        ls = window.layer_stack

        original_color = ls.boundary_color
        ls.boundary_color = (255, 0, 0)  # red
        ls.fill_mode = 'boundary'
        window.canvas.refresh_overlays()
        QApplication.processEvents()

        assert ls.boundary_color == (255, 0, 0)
        _save_screenshot(window.canvas, "boundary_custom_red")

        # Restore
        ls.boundary_color = original_color
        ls.fill_mode = 'solid'
        window.canvas.refresh_overlays()
        QApplication.processEvents()


class TestBoundaryThickness:
    """Boundary thickness should be configurable and zoom-independent."""

    def test_default_thickness(self, app_with_real_image):
        window = app_with_real_image
        assert window.layer_stack.boundary_thickness == 1

    def test_thickness_configurable(self, app_with_real_image):
        window = app_with_real_image
        ls = window.layer_stack

        ls.boundary_thickness = 5
        ls.fill_mode = 'boundary'
        window.canvas.refresh_overlays()
        QApplication.processEvents()

        _save_screenshot(window.canvas, "boundary_thick_5")

        # The canvas method should compute pixel thickness from zoom
        px = window.canvas._boundary_thickness_px()
        assert px >= 1

        # Restore
        ls.boundary_thickness = 1
        ls.fill_mode = 'solid'
        window.canvas.refresh_overlays()
        QApplication.processEvents()

    def test_thickness_scales_with_zoom(self, app_with_real_image):
        """Thickness in source pixels should increase when zoomed out."""
        window = app_with_real_image
        canvas = window.canvas
        ls = window.layer_stack
        ls.boundary_thickness = 3

        # At 100% zoom
        canvas.resetTransform()
        QApplication.processEvents()
        px_100 = canvas._boundary_thickness_px()

        # Zoom out to ~50%
        canvas.scale(0.5, 0.5)
        QApplication.processEvents()
        px_50 = canvas._boundary_thickness_px()

        # At 50% zoom, thickness should be roughly 2x to maintain screen size
        assert px_50 > px_100, \
            f"Thickness should increase when zoomed out: {px_50} vs {px_100}"

        # Reset
        ls.boundary_thickness = 1
        canvas.fit_to_window()
        QApplication.processEvents()


class TestSolidRemovesOutline:
    """In solid fill mode, non-selected ROIs should have no boundary.
    Selected ROIs should still show cyan boundary."""

    def test_solid_no_boundary_on_unselected(self, app_with_real_image):
        window = app_with_real_image
        canvas = window.canvas
        ls = window.layer_stack

        ls.fill_mode = 'solid'
        # Deselect all
        canvas._selection._layers.clear()
        canvas._active_layer = None
        canvas.refresh_overlays()
        QApplication.processEvents()

        _save_screenshot(canvas, "boundary_solid_no_outline")

    def test_solid_shows_cyan_on_active(self, app_with_real_image):
        window = app_with_real_image
        canvas = window.canvas
        ls = window.layer_stack

        ls.fill_mode = 'solid'
        if not ls.roi_layers:
            pytest.skip("No ROIs available")
        roi = ls.roi_layers[0]
        canvas.set_active_layer(roi)
        canvas._selection.set([roi])
        canvas.refresh_overlays()
        QApplication.processEvents()

        _save_screenshot(canvas, "boundary_solid_active_cyan")


class TestPropertiesPanel:
    """Properties panel should have boundary controls."""

    def test_fill_combo_items(self, app_with_real_image):
        window = app_with_real_image
        panel = window.properties_panel
        items = [panel.fill_mode_combo.itemText(i)
                 for i in range(panel.fill_mode_combo.count())]
        assert "Solid" in items
        assert "Boundary" in items
        assert "Solid + Boundary" in items
        # Old name should NOT be present
        assert "Outline" not in items

    def test_thickness_spinbox_exists(self, app_with_real_image):
        window = app_with_real_image
        panel = window.properties_panel
        assert hasattr(panel, 'thickness_spin')
        assert panel.thickness_spin.minimum() == 1
        assert panel.thickness_spin.maximum() == 10
        assert panel.thickness_spin.value() == 1

    def test_color_buttons_exist(self, app_with_real_image):
        window = app_with_real_image
        panel = window.properties_panel
        assert hasattr(panel, 'boundary_color_btn')
        assert hasattr(panel, 'active_color_btn')


class TestSelectionHighlightCleanup:
    """Old selection highlight items should be cleaned up since boundaries
    are now baked into ROI overlays."""

    def test_no_stale_highlight_items(self, app_with_real_image):
        window = app_with_real_image
        canvas = window.canvas

        # Select a ROI
        if not window.layer_stack.roi_layers:
            pytest.skip("No ROIs available")
        roi = window.layer_stack.roi_layers[0]
        canvas.set_active_layer(roi)
        canvas._selection.set([roi])
        QApplication.processEvents()

        # Old highlight system should have no items
        assert len(canvas._selection_highlight_items) == 0, \
            "Selection highlights should be empty (boundaries baked into overlays)"

        # Deselect
        canvas._selection.clear()
        QApplication.processEvents()
        assert len(canvas._selection_highlight_items) == 0
