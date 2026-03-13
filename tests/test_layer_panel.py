"""Tests for LayerPanel (montaris.widgets.layer_panel)."""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from montaris.app import MontarisApp, apply_dark_theme
from montaris.layers import ImageLayer, ROILayer, ROI_COLORS
from montaris.widgets.layer_panel import LayerPanel, ColorPaletteDialog


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def app_fixture():
    qapp = QApplication.instance() or QApplication(sys.argv)
    apply_dark_theme(qapp)
    win = MontarisApp()
    img = np.zeros((100, 100), dtype=np.uint8)
    win.layer_stack.set_image(ImageLayer("test", img))
    win.canvas.refresh_image()
    QApplication.processEvents()
    yield win
    win.close()


def _reset_rois(win):
    """Clear and recreate a predictable set of ROIs."""
    win.layer_stack.roi_layers.clear()
    for i in range(3):
        roi = ROILayer(f"ROI {i + 1}", 100, 100)
        roi.mask[i * 10:(i + 1) * 10, i * 10:(i + 1) * 10] = 255
        win.layer_stack.roi_layers.append(roi)
    win.layer_panel.refresh()
    QApplication.processEvents()


# ── refresh / item count ─────────────────────────────────────────────


class TestRefresh:
    def test_item_count_matches_rois_plus_image(self, app_fixture):
        """After refresh, list widget has 1 image row + N ROI rows."""
        win = app_fixture
        _reset_rois(win)
        panel = win.layer_panel
        # 1 image item + 3 ROIs
        assert panel.list_widget.count() == 4

    def test_item_count_after_adding_roi(self, app_fixture):
        win = app_fixture
        _reset_rois(win)
        panel = win.layer_panel
        # Add another ROI
        roi = ROILayer("ROI 4", 100, 100)
        win.layer_stack.roi_layers.append(roi)
        panel.refresh()
        QApplication.processEvents()
        assert panel.list_widget.count() == 5

    def test_item_count_after_removing_roi(self, app_fixture):
        win = app_fixture
        _reset_rois(win)
        panel = win.layer_panel
        win.layer_stack.roi_layers.pop()
        panel.refresh()
        QApplication.processEvents()
        # 1 image + 2 ROIs
        assert panel.list_widget.count() == 3

    def test_header_shows_count(self, app_fixture):
        win = app_fixture
        _reset_rois(win)
        panel = win.layer_panel
        assert "3" in panel.header.text()


# ── _get_selected_roi_indices ────────────────────────────────────────


class TestGetSelectedRoiIndices:
    def test_no_selection_returns_empty(self, app_fixture):
        win = app_fixture
        _reset_rois(win)
        panel = win.layer_panel
        panel.list_widget.clearSelection()
        assert panel._get_selected_roi_indices() == []

    def test_single_selection(self, app_fixture):
        win = app_fixture
        _reset_rois(win)
        panel = win.layer_panel
        panel.list_widget.clearSelection()
        # Row 0 is the image; rows 1-3 are ROIs 0-2
        item = panel.list_widget.item(1)
        item.setSelected(True)
        indices = panel._get_selected_roi_indices()
        assert indices == [0]

    def test_multi_selection(self, app_fixture):
        win = app_fixture
        _reset_rois(win)
        panel = win.layer_panel
        panel.list_widget.clearSelection()
        panel.list_widget.item(1).setSelected(True)
        panel.list_widget.item(3).setSelected(True)
        indices = panel._get_selected_roi_indices()
        assert indices == [0, 2]


# ── visibility toggle ────────────────────────────────────────────────


class TestVisibilityToggle:
    def test_uncheck_item_hides_roi(self, app_fixture):
        win = app_fixture
        _reset_rois(win)
        panel = win.layer_panel
        # ROI 0 is at row 1
        item = panel.list_widget.item(1)
        roi = win.layer_stack.roi_layers[0]
        assert roi.visible is True
        # Simulate unchecking
        item.setCheckState(Qt.Unchecked)
        QApplication.processEvents()
        assert roi.visible is False

    def test_check_item_shows_roi(self, app_fixture):
        win = app_fixture
        _reset_rois(win)
        panel = win.layer_panel
        roi = win.layer_stack.roi_layers[0]
        roi.visible = False
        panel.refresh()
        QApplication.processEvents()
        item = panel.list_widget.item(1)
        item.setCheckState(Qt.Checked)
        QApplication.processEvents()
        assert roi.visible is True


# ── _toggle_all_visibility ───────────────────────────────────────────


class TestToggleAllVisibility:
    def test_toggle_all_false(self, app_fixture):
        win = app_fixture
        _reset_rois(win)
        panel = win.layer_panel
        panel._toggle_all_visibility(False)
        for roi in win.layer_stack.roi_layers:
            assert roi.visible is False

    def test_toggle_all_true(self, app_fixture):
        win = app_fixture
        _reset_rois(win)
        panel = win.layer_panel
        # First hide all
        panel._toggle_all_visibility(False)
        # Then show all
        panel._toggle_all_visibility(True)
        for roi in win.layer_stack.roi_layers:
            assert roi.visible is True


# ── _random_color ────────────────────────────────────────────────────


class TestRandomColor:
    def test_assigns_color_tuple(self, app_fixture):
        win = app_fixture
        _reset_rois(win)
        panel = win.layer_panel
        roi = win.layer_stack.roi_layers[0]
        original_color = roi.color
        # Select the first ROI (row 1)
        panel.list_widget.setCurrentRow(1)
        QApplication.processEvents()
        panel._random_color()
        # Color should be a 3-tuple with values in [50, 255]
        assert isinstance(roi.color, tuple)
        assert len(roi.color) == 3
        for ch in roi.color:
            assert 50 <= ch <= 255

    def test_no_selection_is_safe(self, app_fixture):
        win = app_fixture
        _reset_rois(win)
        panel = win.layer_panel
        panel.list_widget.setCurrentRow(0)  # image row, not ROI
        QApplication.processEvents()
        # Should not raise
        panel._random_color()


# ── _duplicate_selected ──────────────────────────────────────────────


class TestDuplicateSelected:
    def test_creates_copy(self, app_fixture):
        win = app_fixture
        _reset_rois(win)
        panel = win.layer_panel
        initial_count = len(win.layer_stack.roi_layers)
        # Select first ROI (row 1)
        panel.list_widget.setCurrentRow(1)
        QApplication.processEvents()
        panel._duplicate_selected()
        assert len(win.layer_stack.roi_layers) == initial_count + 1
        # The copy should be at index 1
        copy = win.layer_stack.roi_layers[1]
        assert "(copy)" in copy.name

    def test_duplicate_preserves_mask(self, app_fixture):
        win = app_fixture
        _reset_rois(win)
        panel = win.layer_panel
        original = win.layer_stack.roi_layers[0]
        panel.list_widget.setCurrentRow(1)
        QApplication.processEvents()
        panel._duplicate_selected()
        copy = win.layer_stack.roi_layers[1]
        np.testing.assert_array_equal(original.mask, copy.mask)

    def test_item_count_after_duplicate(self, app_fixture):
        win = app_fixture
        _reset_rois(win)
        panel = win.layer_panel
        panel.list_widget.setCurrentRow(1)
        QApplication.processEvents()
        panel._duplicate_selected()
        # 1 image + 3 original + 1 copy = 5
        assert panel.list_widget.count() == 5


# ── ColorPaletteDialog ──────────────────────────────────────────────


class TestColorPaletteDialog:
    def test_can_create(self, app_fixture):
        dlg = ColorPaletteDialog((255, 0, 0), parent=app_fixture)
        assert dlg is not None
        assert dlg.windowTitle() == "Choose Color"

    def test_selected_color_initially_none(self, app_fixture):
        dlg = ColorPaletteDialog((0, 255, 0), parent=app_fixture)
        assert dlg.selected_color is None

    def test_pick_sets_color(self, app_fixture):
        dlg = ColorPaletteDialog((0, 0, 255), parent=app_fixture)
        dlg._pick((128, 64, 32))
        assert dlg.selected_color == (128, 64, 32)

    def test_has_palette_buttons(self, app_fixture):
        dlg = ColorPaletteDialog((100, 100, 100), parent=app_fixture)
        # At minimum, the grid should have ROI_COLORS buttons + 1 custom button
        # Total children buttons >= len(ROI_COLORS) + 1
        from PySide6.QtWidgets import QPushButton
        buttons = dlg.findChildren(QPushButton)
        assert len(buttons) >= len(ROI_COLORS) + 1
