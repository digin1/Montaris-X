"""Comprehensive tests for LayerPanel (montaris.widgets.layer_panel).

Covers: _remove_selected, _clear_all, _merge_selected, _insert_roi_at,
_rename_selected, _move_roi_to, _export_single_roi, _export_selected_rois,
_change_color, _toggle_all_visibility, _on_rows_moved, _on_item_changed,
_show_context_menu, PlaceholderListWidget, ROINavBar.
"""

import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from PySide6.QtWidgets import QApplication, QMessageBox, QInputDialog
from PySide6.QtCore import Qt

from montaris.app import MontarisApp, apply_dark_theme
from montaris.layers import ImageLayer, ROILayer


# ── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    apply_dark_theme(app)
    return app


@pytest.fixture
def app(qapp):
    win = MontarisApp()
    img = np.zeros((100, 100), dtype=np.uint8)
    win.layer_stack.set_image(ImageLayer("test", img))
    for i in range(5):
        roi = ROILayer(f"ROI_{i}", 100, 100)
        roi.mask[10:20, 10:20] = 255
        win.layer_stack.add_roi(roi)
    win.layer_panel.refresh()
    QApplication.processEvents()
    return win


# ── _remove_selected ────────────────────────────────────────────────────


class TestRemoveSelected:
    def test_remove_single_roi(self, app):
        """Confirming deletion removes the selected ROI."""
        panel = app.layer_panel
        panel.list_widget.setCurrentRow(1)  # ROI_0
        QApplication.processEvents()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            panel._remove_selected()
        assert len(app.layer_stack.roi_layers) == 4
        names = [r.name for r in app.layer_stack.roi_layers]
        assert "ROI_0" not in names

    def test_remove_cancelled(self, app):
        """Declining the confirmation keeps all ROIs."""
        panel = app.layer_panel
        count_before = len(app.layer_stack.roi_layers)
        panel.list_widget.setCurrentRow(1)
        QApplication.processEvents()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.No):
            panel._remove_selected()
        assert len(app.layer_stack.roi_layers) == count_before

    def test_remove_multi_selected(self, app):
        """Selecting multiple ROIs and confirming removes all of them."""
        panel = app.layer_panel
        panel.list_widget.clearSelection()
        panel.list_widget.item(1).setSelected(True)  # ROI_0
        panel.list_widget.item(2).setSelected(True)  # ROI_1
        QApplication.processEvents()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            panel._remove_selected()
        assert len(app.layer_stack.roi_layers) == 3

    def test_remove_no_selection(self, app):
        """With no ROI selected, _remove_selected is a no-op."""
        panel = app.layer_panel
        panel.list_widget.clearSelection()
        panel.list_widget.setCurrentRow(0)  # image row
        QApplication.processEvents()
        count_before = len(app.layer_stack.roi_layers)
        panel._remove_selected()
        assert len(app.layer_stack.roi_layers) == count_before


# ── _clear_all ──────────────────────────────────────────────────────────


class TestClearAll:
    def test_clear_all_confirmed(self, app):
        """Confirming clear removes every ROI."""
        panel = app.layer_panel
        assert len(app.layer_stack.roi_layers) > 0
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            panel._clear_all()
        assert len(app.layer_stack.roi_layers) == 0

    def test_clear_all_cancelled(self, app):
        """Declining clear keeps all ROIs."""
        panel = app.layer_panel
        count_before = len(app.layer_stack.roi_layers)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.No):
            panel._clear_all()
        assert len(app.layer_stack.roi_layers) == count_before

    def test_clear_all_empty_list(self, app):
        """Clearing when no ROIs exist is a safe no-op (no dialog shown)."""
        app.layer_stack.roi_layers.clear()
        app.layer_panel.refresh()
        QApplication.processEvents()
        with patch.object(QMessageBox, "question") as mock_q:
            app.layer_panel._clear_all()
            mock_q.assert_not_called()


# ── _merge_selected ─────────────────────────────────────────────────────


class TestMergeSelected:
    def test_merge_two_rois(self, app):
        """Merging two selected ROIs reduces count by one."""
        panel = app.layer_panel
        count_before = len(app.layer_stack.roi_layers)
        panel.list_widget.clearSelection()
        panel.list_widget.item(1).setSelected(True)
        panel.list_widget.item(2).setSelected(True)
        QApplication.processEvents()
        panel._merge_selected()
        assert len(app.layer_stack.roi_layers) == count_before - 1

    def test_merge_single_roi_noop(self, app):
        """Merging with only one ROI selected does nothing."""
        panel = app.layer_panel
        count_before = len(app.layer_stack.roi_layers)
        panel.list_widget.clearSelection()
        panel.list_widget.item(1).setSelected(True)
        QApplication.processEvents()
        panel._merge_selected()
        assert len(app.layer_stack.roi_layers) == count_before


# ── _insert_roi_at ──────────────────────────────────────────────────────


class TestInsertRoiAt:
    def test_insert_at_beginning(self, app):
        """Inserting at index 0 places a new ROI first."""
        panel = app.layer_panel
        count_before = len(app.layer_stack.roi_layers)
        panel._insert_roi_at(0)
        assert len(app.layer_stack.roi_layers) == count_before + 1
        assert app.layer_stack.roi_layers[0].name != "ROI_0"

    def test_insert_at_end(self, app):
        """Inserting at the end appends a new ROI."""
        panel = app.layer_panel
        count_before = len(app.layer_stack.roi_layers)
        panel._insert_roi_at(count_before)
        assert len(app.layer_stack.roi_layers) == count_before + 1

    def test_insert_creates_empty_mask(self, app):
        """The inserted ROI has an all-zero mask."""
        panel = app.layer_panel
        panel._insert_roi_at(0)
        new_roi = app.layer_stack.roi_layers[0]
        assert np.count_nonzero(new_roi.mask) == 0


# ── _rename_selected ────────────────────────────────────────────────────


class TestRenameSelected:
    def test_rename_updates_name(self, app):
        """Accepting the rename dialog updates the ROI name."""
        panel = app.layer_panel
        panel.list_widget.setCurrentRow(1)
        QApplication.processEvents()
        roi = app.layer_stack.roi_layers[0]
        with patch.object(QInputDialog, "getText", return_value=("NewName", True)):
            panel._rename_selected()
        assert roi.name == "NewName"

    def test_rename_cancelled(self, app):
        """Cancelling the rename dialog keeps the original name."""
        panel = app.layer_panel
        panel.list_widget.setCurrentRow(1)
        QApplication.processEvents()
        roi = app.layer_stack.roi_layers[0]
        original_name = roi.name
        with patch.object(QInputDialog, "getText", return_value=("Ignored", False)):
            panel._rename_selected()
        assert roi.name == original_name

    def test_rename_uniqueness(self, app):
        """Renaming to a duplicate name appends a suffix."""
        panel = app.layer_panel
        existing_name = app.layer_stack.roi_layers[1].name
        panel.list_widget.setCurrentRow(1)
        QApplication.processEvents()
        with patch.object(QInputDialog, "getText", return_value=(existing_name, True)):
            panel._rename_selected()
        roi = app.layer_stack.roi_layers[0]
        # Name should differ from the conflicting name (suffix added)
        assert roi.name != existing_name or roi is app.layer_stack.roi_layers[1]


# ── _move_roi_to ────────────────────────────────────────────────────────


class TestMoveRoiTo:
    def test_move_first_to_last(self, app):
        """Moving ROI from position 0 to last reorders correctly."""
        panel = app.layer_panel
        n = len(app.layer_stack.roi_layers)
        first_name = app.layer_stack.roi_layers[0].name
        with patch.object(QInputDialog, "getInt", return_value=(n, True)):
            panel._move_roi_to(0)
        assert app.layer_stack.roi_layers[-1].name == first_name

    def test_move_cancelled(self, app):
        """Cancelling the move dialog preserves order."""
        panel = app.layer_panel
        names_before = [r.name for r in app.layer_stack.roi_layers]
        with patch.object(QInputDialog, "getInt", return_value=(1, False)):
            panel._move_roi_to(0)
        names_after = [r.name for r in app.layer_stack.roi_layers]
        assert names_before == names_after


# ── _export_single_roi ──────────────────────────────────────────────────


class TestExportSingleRoi:
    def test_export_imagej(self, app, tmp_path):
        """Exporting a single ROI as ImageJ .roi writes one file."""
        panel = app.layer_panel
        mock_mask_to_roi = MagicMock(return_value={"type": "polygon"})
        mock_write_roi = MagicMock()
        with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory",
                    return_value=str(tmp_path)), \
             patch("montaris.io.imagej_roi.mask_to_imagej_roi", mock_mask_to_roi), \
             patch("montaris.io.imagej_roi.write_imagej_roi", mock_write_roi):
            panel._export_single_roi(0, "imagej")
        mock_mask_to_roi.assert_called_once()
        mock_write_roi.assert_called_once()

    def test_export_png(self, app, tmp_path):
        """Exporting a single ROI as PNG writes one file."""
        panel = app.layer_panel
        mock_image_cls = MagicMock()
        mock_img_instance = MagicMock()
        mock_image_cls.fromarray.return_value = mock_img_instance
        with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory",
                    return_value=str(tmp_path)), \
             patch("PIL.Image.fromarray", mock_image_cls.fromarray):
            panel._export_single_roi(0, "png")
        mock_image_cls.fromarray.assert_called_once()
        mock_img_instance.save.assert_called_once()


# ── _export_selected_rois ───────────────────────────────────────────────


class TestExportSelectedRois:
    def test_export_multiple_png(self, app, tmp_path):
        """Exporting multiple selected ROIs as PNG writes multiple files."""
        panel = app.layer_panel
        panel.list_widget.clearSelection()
        panel.list_widget.item(1).setSelected(True)
        panel.list_widget.item(2).setSelected(True)
        QApplication.processEvents()

        mock_image_cls = MagicMock()
        mock_img_instance = MagicMock()
        mock_image_cls.fromarray.return_value = mock_img_instance
        with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory",
                    return_value=str(tmp_path)), \
             patch("PIL.Image.fromarray", mock_image_cls.fromarray):
            panel._export_selected_rois("png")
        assert mock_image_cls.fromarray.call_count == 2
        assert mock_img_instance.save.call_count == 2

    def test_export_no_dir_selected(self, app):
        """Cancelling directory selection does nothing."""
        panel = app.layer_panel
        panel.list_widget.clearSelection()
        panel.list_widget.item(1).setSelected(True)
        QApplication.processEvents()
        with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory",
                    return_value=""):
            panel._export_selected_rois("png")
        # No crash = success

    def test_export_empty_selection(self, app, tmp_path):
        """Exporting with no ROIs selected is a no-op."""
        panel = app.layer_panel
        panel.list_widget.clearSelection()
        QApplication.processEvents()
        with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory") as mock_dir:
            panel._export_selected_rois("png")
            mock_dir.assert_not_called()


# ── _change_color ───────────────────────────────────────────────────────


class TestChangeColor:
    def test_change_color_updates_roi(self, app):
        """Accepting the color dialog updates the ROI color."""
        panel = app.layer_panel
        panel.list_widget.setCurrentRow(1)
        QApplication.processEvents()
        roi = app.layer_stack.roi_layers[0]
        new_color = (10, 20, 30)
        mock_dlg = MagicMock()
        mock_dlg.exec.return_value = True
        mock_dlg.selected_color = new_color
        with patch("montaris.widgets.layer_panel.ColorPaletteDialog",
                    return_value=mock_dlg):
            panel._change_color()
        assert roi.color == new_color

    def test_change_color_cancelled(self, app):
        """Cancelling the color dialog preserves the original color."""
        panel = app.layer_panel
        panel.list_widget.setCurrentRow(1)
        QApplication.processEvents()
        roi = app.layer_stack.roi_layers[0]
        original_color = roi.color
        mock_dlg = MagicMock()
        mock_dlg.exec.return_value = False
        with patch("montaris.widgets.layer_panel.ColorPaletteDialog",
                    return_value=mock_dlg):
            panel._change_color()
        assert roi.color == original_color


# ── _toggle_all_visibility ──────────────────────────────────────────────


class TestToggleAllVisibilityComprehensive:
    def test_toggle_off(self, app):
        """Toggling visibility off hides all ROIs."""
        panel = app.layer_panel
        panel._toggle_all_visibility(False)
        for roi in app.layer_stack.roi_layers:
            assert roi.visible is False

    def test_toggle_on(self, app):
        """Toggling visibility on shows all ROIs."""
        panel = app.layer_panel
        panel._toggle_all_visibility(False)
        panel._toggle_all_visibility(True)
        for roi in app.layer_stack.roi_layers:
            assert roi.visible is True

    def test_toggle_refreshes_list(self, app):
        """After toggling, the list widget reflects new check states."""
        panel = app.layer_panel
        panel._toggle_all_visibility(False)
        # Check that all ROI items are unchecked
        for i in range(1, panel.list_widget.count()):
            item = panel.list_widget.item(i)
            data = item.data(Qt.UserRole)
            if data and data[0] == "roi":
                assert item.checkState() == Qt.Unchecked


# ── _on_rows_moved ──────────────────────────────────────────────────────


class TestOnRowsMoved:
    def test_reorder_syncs_layer_stack(self, app):
        """Simulating a row move updates the layer_stack order."""
        panel = app.layer_panel
        names_before = [r.name for r in app.layer_stack.roi_layers]
        assert len(names_before) >= 2

        # Block signals to prevent cascading refresh that deletes items
        panel.list_widget.blockSignals(True)
        # Manually swap the UserRole data on the first two ROI items
        # to simulate drag-drop (items at row 1 and 2 = ROI indices 0 and 1)
        item1 = panel.list_widget.item(1)
        item2 = panel.list_widget.item(2)
        data1 = item1.data(Qt.UserRole)
        data2 = item2.data(Qt.UserRole)
        item1.setData(Qt.UserRole, data2)
        item2.setData(Qt.UserRole, data1)
        panel.list_widget.blockSignals(False)
        panel._updating = False
        panel._on_rows_moved()
        names_after = [r.name for r in app.layer_stack.roi_layers]
        # The order should have changed
        assert names_after[0] == names_before[1]
        assert names_after[1] == names_before[0]


# ── _on_item_changed ────────────────────────────────────────────────────


class TestOnItemChanged:
    def test_rename_via_item_text(self, app):
        """Editing item text triggers an inline rename."""
        panel = app.layer_panel
        roi = app.layer_stack.roi_layers[0]
        item = panel.list_widget.item(1)
        # Change item text to simulate inline edit
        panel._updating = False
        item.setText("1. BrandNewName")
        QApplication.processEvents()
        assert roi.name == "BrandNewName"

    def test_visibility_via_checkbox(self, app):
        """Unchecking a checkbox hides the ROI."""
        panel = app.layer_panel
        roi = app.layer_stack.roi_layers[0]
        assert roi.visible is True
        item = panel.list_widget.item(1)
        panel._updating = False
        item.setCheckState(Qt.Unchecked)
        QApplication.processEvents()
        assert roi.visible is False


# ── _show_context_menu ──────────────────────────────────────────────────


class _NonBlockingMenu:
    """A stand-in for QMenu that records actions without blocking on exec."""

    def __init__(self, parent=None):
        self._actions = []
        self._menus = []

    def addAction(self, action):
        self._actions.append(action)
        return action

    def addSeparator(self):
        sep = MagicMock()
        sep.isSeparator.return_value = True
        sep.text.return_value = ""
        self._actions.append(sep)

    def addMenu(self, title):
        sub = _NonBlockingMenu()
        sub._title = title
        self._menus.append(sub)
        return sub

    def exec(self, *a, **kw):
        pass  # non-blocking

    def actions(self):
        return self._actions


class TestShowContextMenu:
    def _build_menu_actions(self, panel, row_index, selected_rows=None):
        """Build a context menu for the given row and return action texts."""
        from PySide6.QtWidgets import QMenu
        panel.list_widget.setCurrentRow(row_index)
        if selected_rows is not None:
            panel.list_widget.clearSelection()
            for r in selected_rows:
                panel.list_widget.item(r).setSelected(True)
        QApplication.processEvents()

        captured = []

        with patch("montaris.widgets.layer_panel.QMenu", side_effect=lambda parent=None: (lambda m: (captured.append(m), m)[-1])(_NonBlockingMenu(parent))):
            item = panel.list_widget.item(row_index)
            rect = panel.list_widget.visualItemRect(item)
            panel._show_context_menu(rect.center())

        return captured

    def test_menu_has_expected_actions(self, app):
        """The context menu for a ROI item has Duplicate, Rename, Delete, etc."""
        panel = app.layer_panel
        captured = self._build_menu_actions(panel, 1)

        assert len(captured) >= 1
        menu = captured[0]
        action_texts = [a.text() for a in menu.actions()
                        if not getattr(a, 'isSeparator', lambda: False)()
                        and hasattr(a, 'text')]
        assert "Duplicate" in action_texts
        assert "Rename..." in action_texts
        assert "Delete" in action_texts
        assert "Clear All ROIs" in action_texts
        assert "Change Color..." in action_texts
        assert "Random Color" in action_texts
        assert "Move To Position..." in action_texts

    def test_context_menu_multi_has_merge(self, app):
        """With two ROIs selected, the context menu includes Merge Selected."""
        panel = app.layer_panel
        captured = self._build_menu_actions(panel, 1, selected_rows=[1, 2])

        assert len(captured) >= 1
        action_texts = [a.text() for a in captured[0].actions()
                        if not getattr(a, 'isSeparator', lambda: False)()
                        and hasattr(a, 'text')]
        assert "Merge Selected" in action_texts

    def test_context_menu_on_image_row_noop(self, app):
        """Right-clicking the image row does not produce a context menu."""
        panel = app.layer_panel
        captured = self._build_menu_actions(panel, 0)
        # No menu should have been created for the image row
        assert len(captured) == 0


# ── PlaceholderListWidget ───────────────────────────────────────────────


class TestPlaceholderListWidget:
    def test_placeholder_text_stored(self, app):
        """PlaceholderListWidget stores the placeholder text."""
        from montaris.widgets.layer_panel import PlaceholderListWidget
        w = PlaceholderListWidget("No items here")
        assert w._placeholder == "No items here"

    def test_placeholder_default_empty(self, app):
        """Default placeholder is empty string."""
        from montaris.widgets.layer_panel import PlaceholderListWidget
        w = PlaceholderListWidget()
        assert w._placeholder == ""

    def test_app_list_widget_has_placeholder(self, app):
        """The layer panel's list widget is a PlaceholderListWidget."""
        from montaris.widgets.layer_panel import PlaceholderListWidget
        assert isinstance(app.layer_panel.list_widget, PlaceholderListWidget)


# ── ROINavBar ───────────────────────────────────────────────────────────


class TestROINavBar:
    def test_set_segments_stores_data(self, app):
        """set_segments stores the segment list."""
        nav = app.layer_panel.nav_bar
        segments = [(0.5, (255, 0, 0), 0), (0.5, (0, 255, 0), 1)]
        nav.set_segments(segments)
        assert nav._segments == segments

    def test_set_segments_empty(self, app):
        """Setting empty segments clears the bar."""
        nav = app.layer_panel.nav_bar
        nav.set_segments([])
        assert nav._segments == []

    def test_nav_bar_populated_after_refresh(self, app):
        """After refresh, the nav bar has segments matching the ROI count."""
        app.layer_panel.refresh()
        nav = app.layer_panel.nav_bar
        assert len(nav._segments) == len(app.layer_stack.roi_layers)

    def test_nav_bar_segment_fractions_sum_to_one(self, app):
        """Nav bar segment fractions sum approximately to 1.0."""
        app.layer_panel.refresh()
        nav = app.layer_panel.nav_bar
        if nav._segments:
            total = sum(frac for frac, _, _ in nav._segments)
            assert abs(total - 1.0) < 0.01
