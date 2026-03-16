"""Comprehensive tests for MontarisApp methods that lack coverage.

Covers: cursor info, sidebar toggles, global opacity, JIT accel toggle,
tool change, layer selection, ROI add/remove/clear, display mode, adjustments,
minimap pan, fix overlaps, close image, load single channel, clear active ROI,
student session, view instructions, display channels, flip/rotate, and more.
"""

import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QPointF, Qt, QUrl, QMimeData, QTimer
from PySide6.QtGui import QDropEvent, QDragEnterEvent, QTransform

from montaris.app import MontarisApp, apply_dark_theme
from montaris.layers import ImageLayer, ROILayer


# ── Fixtures ────────────────────────────────────────────────────────


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
    win.canvas.refresh_image()
    QApplication.processEvents()
    return win


# ── Helpers ──────────────────────────────────────────────────────────


def _add_roi(win, name="ROI 1"):
    """Add a visible ROI with some mask content to the app."""
    h, w = win.layer_stack.image_layer.shape[:2]
    roi = ROILayer(name, w, h)
    roi.mask[10:20, 10:20] = 255
    win.layer_stack.add_roi(roi)
    return roi


# ═══════════════════════════════════════════════════════════════════
# 1. _update_cursor_info (internally named _update_cursor_info)
# ═══════════════════════════════════════════════════════════════════


class TestOnCursorMoved:
    def test_statusbar_shows_coordinates(self, app):
        app._update_cursor_info(42, 17, 128)
        msg = app.statusbar.currentMessage()
        assert "X: 42" in msg
        assert "Y: 17" in msg
        assert "Value: 128" in msg

    def test_statusbar_includes_roi_info_when_active(self, app):
        roi = _add_roi(app, "R1")
        app.canvas._active_layer = roi
        # Query a pixel inside the painted region
        app._update_cursor_info(15, 15, 200)
        msg = app.statusbar.currentMessage()
        assert "ROI: yes" in msg

    def test_statusbar_roi_no_when_outside_mask(self, app):
        roi = _add_roi(app, "R2")
        app.canvas._active_layer = roi
        # Query a pixel outside the painted region
        app._update_cursor_info(0, 0, 0)
        msg = app.statusbar.currentMessage()
        assert "ROI: no" in msg


# ═══════════════════════════════════════════════════════════════════
# 2. _toggle_left_sidebar / _toggle_right_sidebar
# ═══════════════════════════════════════════════════════════════════


class TestToggleLeftSidebar:
    def test_collapse_hides_docks_shows_toolbar(self, app):
        # In offscreen mode isVisible() is always False; use isHidden() instead
        app._left_collapsed = False
        app._toggle_left_sidebar()
        assert app._left_collapsed is True
        assert app._left_toolbar.isHidden() is False   # toolbar requested visible
        assert app._tool_dock.isHidden() is True        # docks hidden
        assert app._minimap_dock.isHidden() is True

    def test_expand_shows_docks_hides_toolbar(self, app):
        app._left_collapsed = True
        app._toggle_left_sidebar()
        assert app._left_collapsed is False
        assert app._left_toolbar.isHidden() is True     # toolbar hidden
        assert app._tool_dock.isHidden() is False       # docks requested visible
        assert app._minimap_dock.isHidden() is False


class TestToggleRightSidebar:
    def test_collapse_hides_docks_shows_toolbar(self, app):
        app._right_collapsed = False
        app._toggle_right_sidebar()
        assert app._right_collapsed is True
        assert app._right_toolbar.isHidden() is False
        assert app._layer_dock.isHidden() is True
        assert app._details_dock.isHidden() is True

    def test_expand_shows_docks_hides_toolbar(self, app):
        app._right_collapsed = True
        app._toggle_right_sidebar()
        assert app._right_collapsed is False
        assert app._right_toolbar.isHidden() is True
        assert app._layer_dock.isHidden() is False
        assert app._details_dock.isHidden() is False


# ═══════════════════════════════════════════════════════════════════
# 3. _on_global_opacity_changed
# ═══════════════════════════════════════════════════════════════════


class TestOnGlobalOpacityChanged:
    def test_sets_opacity_factor(self, app):
        with patch.object(app.canvas, "refresh_overlays_lut_only"):
            app._on_global_opacity_changed(50)
        assert abs(app.layer_stack._global_opacity_factor - 0.5) < 1e-6

    def test_zero_opacity(self, app):
        with patch.object(app.canvas, "refresh_overlays_lut_only"):
            app._on_global_opacity_changed(0)
        assert app.layer_stack._global_opacity_factor == 0.0

    def test_full_opacity(self, app):
        with patch.object(app.canvas, "refresh_overlays_lut_only"):
            app._on_global_opacity_changed(100)
        assert abs(app.layer_stack._global_opacity_factor - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════
# 4. _toggle_accel
# ═══════════════════════════════════════════════════════════════════


class TestToggleAccel:
    def test_enable_calls_set_enabled(self, app):
        mock_set = MagicMock()
        mock_get = MagicMock(return_value="numba")
        with patch.dict("sys.modules", {}), \
             patch("montaris.app.MontarisApp._toggle_accel.__module__", create=True), \
             patch("montaris.core.accel.set_enabled", mock_set, create=True), \
             patch("montaris.core.accel.get_mode", mock_get, create=True), \
             patch("montaris.core.accel.HAS_NUMBA", True, create=True):
            app._toggle_accel(True)
            mock_set.assert_called_once_with(True)

    def test_disable_calls_set_enabled_false(self, app):
        mock_set = MagicMock()
        mock_get = MagicMock(return_value="numpy")
        with patch("montaris.core.accel.set_enabled", mock_set, create=True), \
             patch("montaris.core.accel.get_mode", mock_get, create=True), \
             patch("montaris.core.accel.HAS_NUMBA", True, create=True):
            app._toggle_accel(False)
            mock_set.assert_called_once_with(False)


# ═══════════════════════════════════════════════════════════════════
# 5. _on_tool_changed
# ═══════════════════════════════════════════════════════════════════


class TestOnToolChanged:
    def test_updates_tool_status_label(self, app):
        mock_tool = MagicMock()
        mock_tool.name = "Brush"
        with patch.object(app.canvas, "set_tool"):
            app._on_tool_changed(mock_tool)
        assert "Brush" in app._tool_status_label.text()
        assert app.active_tool is mock_tool

    def test_none_tool_shows_none(self, app):
        with patch.object(app.canvas, "set_tool"):
            app._on_tool_changed(None)
        assert "None" in app._tool_status_label.text()

    def test_move_tool_shows_hint(self, app):
        mock_tool = MagicMock()
        mock_tool.name = "Move (selected)"
        with patch.object(app.canvas, "set_tool"):
            app._on_tool_changed(mock_tool)
        # In offscreen mode, use isHidden() to check explicit visibility state
        assert app._move_hint.isHidden() is False

    def test_non_move_tool_hides_hint(self, app):
        mock_tool = MagicMock()
        mock_tool.name = "Brush"
        with patch.object(app.canvas, "set_tool"):
            app._on_tool_changed(mock_tool)
        assert app._move_hint.isHidden() is True


# ═══════════════════════════════════════════════════════════════════
# 6. _on_layer_selected
# ═══════════════════════════════════════════════════════════════════


class TestOnLayerSelected:
    def test_sets_active_layer_on_canvas(self, app):
        roi = _add_roi(app, "Sel1")
        with patch.object(app.canvas, "set_active_layer") as mock_set, \
             patch.object(app.properties_panel, "set_layer"):
            app._on_layer_selected(roi)
            mock_set.assert_called_once_with(roi)

    def test_updates_properties_panel(self, app):
        roi = _add_roi(app, "Sel2")
        with patch.object(app.canvas, "set_active_layer"), \
             patch.object(app.properties_panel, "set_layer") as mock_prop:
            app._on_layer_selected(roi)
            mock_prop.assert_called_once_with(roi)

    def test_tool_status_includes_layer_name(self, app):
        roi = _add_roi(app, "MyROI")
        mock_tool = MagicMock()
        mock_tool.name = "Brush"
        app.active_tool = mock_tool
        app._on_layer_selected(roi)
        assert "MyROI" in app._tool_status_label.text()


# ═══════════════════════════════════════════════════════════════════
# 7. _on_roi_added
# ═══════════════════════════════════════════════════════════════════


class TestOnRoiAdded:
    def test_creates_roi_and_adds_to_stack(self, app):
        initial_count = len(app.layer_stack.roi_layers)
        with patch.object(app.canvas, "refresh_overlays"), \
             patch.object(app.layer_panel, "refresh"):
            app._on_roi_added()
        assert len(app.layer_stack.roi_layers) == initial_count + 1

    def test_no_image_shows_message(self, app):
        app.layer_stack.image_layer = None
        with patch.object(QMessageBox, "information") as mock_msg:
            app._on_roi_added()
            mock_msg.assert_called_once()
        # Restore image for subsequent tests
        img = np.zeros((100, 100), dtype=np.uint8)
        app.layer_stack.set_image(ImageLayer("test", img))


# ═══════════════════════════════════════════════════════════════════
# 8. _on_roi_removed
# ═══════════════════════════════════════════════════════════════════


class TestOnRoiRemoved:
    def test_removes_roi_at_index(self, app):
        app.layer_stack.roi_layers.clear()
        r1 = _add_roi(app, "Del1")
        r2 = _add_roi(app, "Del2")
        with patch.object(app.canvas, "refresh_overlays"), \
             patch.object(app.layer_panel, "refresh"), \
             patch.object(app.properties_panel, "set_layer"):
            app._on_roi_removed(0)
        assert len(app.layer_stack.roi_layers) == 1
        assert app.layer_stack.roi_layers[0].name == "Del2"

    def test_clears_active_layer(self, app):
        app.layer_stack.roi_layers.clear()
        _add_roi(app, "Del3")
        with patch.object(app.canvas, "refresh_overlays"), \
             patch.object(app.layer_panel, "refresh"), \
             patch.object(app.properties_panel, "set_layer"):
            app._on_roi_removed(0)
        assert app.canvas._active_layer is None


# ═══════════════════════════════════════════════════════════════════
# 9. _on_all_cleared
# ═══════════════════════════════════════════════════════════════════


class TestOnAllCleared:
    def test_clears_active_layer_and_selection(self, app):
        _add_roi(app, "Clear1")
        with patch.object(app.canvas, "refresh_overlays"), \
             patch.object(app.layer_panel, "refresh"), \
             patch.object(app.properties_panel, "set_layer"):
            app._on_all_cleared()
        assert app.canvas._active_layer is None
        assert len(app.canvas._selection._layers) == 0

    def test_calls_refresh(self, app):
        with patch.object(app.canvas, "refresh_overlays") as mock_refresh, \
             patch.object(app.layer_panel, "refresh") as mock_panel, \
             patch.object(app.properties_panel, "set_layer") as mock_props:
            app._on_all_cleared()
            mock_refresh.assert_called_once()
            mock_panel.assert_called_once()
            mock_props.assert_called_once_with(None)


# ═══════════════════════════════════════════════════════════════════
# 10. _on_display_mode_changed
# ═══════════════════════════════════════════════════════════════════


class TestOnDisplayModeChanged:
    def test_sets_compositor_mode(self, app):
        app._composite_mode = False
        with patch.object(app.canvas, "refresh_image"):
            app._on_display_mode_changed("grayscale")
        assert app._compositor.mode == "grayscale"

    def test_refreshes_canvas_single_mode(self, app):
        app._composite_mode = False
        with patch.object(app.canvas, "refresh_image") as mock_ref:
            app._on_display_mode_changed("grayscale")
            mock_ref.assert_called_once()

    def test_refreshes_composite_when_composite_mode(self, app):
        app._composite_mode = True
        with patch.object(app, "_refresh_composite") as mock_comp:
            app._on_display_mode_changed("false_color")
            mock_comp.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# 11. _on_adjustments_changed
# ═══════════════════════════════════════════════════════════════════


class TestOnAdjustmentsChanged:
    def test_stores_adjustments(self, app):
        adj = MagicMock()
        adj.brightness = 10
        adj.contrast = 20
        app._on_adjustments_changed(adj)
        assert app._adjustments is adj
        assert app.canvas._adjustments is adj

    def test_starts_debounce_timer(self, app):
        adj = MagicMock()
        app._on_adjustments_changed(adj)
        assert hasattr(app, '_adj_timer')
        assert app._adj_timer.isActive()


# ═══════════════════════════════════════════════════════════════════
# 12. _on_minimap_pan
# ═══════════════════════════════════════════════════════════════════


class TestOnMinimapPan:
    def test_calls_center_on(self, app):
        with patch.object(app.canvas, "centerOn") as mock_center:
            app._on_minimap_pan(50.0, 75.0)
            mock_center.assert_called_once_with(50.0, 75.0)


# ═══════════════════════════════════════════════════════════════════
# 13. fix_overlaps
# ═══════════════════════════════════════════════════════════════════


class TestFixOverlaps:
    def test_calls_roi_ops_fix_overlaps(self, app):
        with patch("montaris.core.roi_ops.fix_overlaps") as mock_fix, \
             patch.object(app.canvas, "refresh_overlays"):
            app.fix_overlaps("later_wins")
            mock_fix.assert_called_once_with(app.layer_stack.roi_layers, "later_wins")

    def test_updates_statusbar(self, app):
        with patch("montaris.core.roi_ops.fix_overlaps"), \
             patch.object(app.canvas, "refresh_overlays"):
            app.fix_overlaps("earlier_wins")
        assert "earlier_wins" in app.statusbar.currentMessage()


# ═══════════════════════════════════════════════════════════════════
# 14. close_image
# ═══════════════════════════════════════════════════════════════════


class TestCloseImage:
    def test_close_with_keep_rois(self, app):
        """close_image with 'Keep ROIs' option clears image but keeps ROIs."""
        _add_roi(app, "KeepMe")
        with patch("montaris.widgets.alert_modal.AlertModal.confirm", return_value="Keep ROIs"):
            app.close_image()
        # Image should be cleared
        assert app.layer_stack.image_layer is None
        # ROIs should still exist
        assert len(app.layer_stack.roi_layers) > 0
        # Restore for other tests
        img = np.zeros((100, 100), dtype=np.uint8)
        app.layer_stack.set_image(ImageLayer("test", img))

    def test_close_no_image_is_noop(self, app):
        saved_img = app.layer_stack.image_layer
        app.layer_stack.image_layer = None
        app.close_image()  # should not raise
        app.layer_stack.image_layer = saved_img

    def test_close_cancel_keeps_image(self, app):
        _add_roi(app, "NoClose")
        with patch("montaris.widgets.alert_modal.AlertModal.confirm", return_value="Cancel"):
            app.close_image()
        assert app.layer_stack.image_layer is not None

    def test_close_no_rois_skips_dialog(self, app):
        app.layer_stack.roi_layers.clear()
        app.close_image()
        assert app.layer_stack.image_layer is None
        # Restore image
        img = np.zeros((100, 100), dtype=np.uint8)
        app.layer_stack.set_image(ImageLayer("test", img))


# ═══════════════════════════════════════════════════════════════════
# 15. _load_single_channel
# ═══════════════════════════════════════════════════════════════════


class TestLoadSingleChannel:
    def test_creates_layer_and_document(self, app):
        app._documents.clear()
        app._active_doc_index = -1
        data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        skipped = []
        with patch.object(app.canvas, "refresh_image"), \
             patch.object(app.canvas, "fit_to_window"), \
             patch.object(app.layer_panel, "refresh"), \
             patch.object(app.minimap, "set_image"), \
             patch.object(app, "_update_minimap_viewport"), \
             patch.object(app.adjustments_panel, "set_image_data"), \
             patch.object(app.display_panel, "set_channels"), \
             patch.object(app.toast, "show"):
            # Reset image layer to simulate fresh load
            app.layer_stack.image_layer = None
            app.layer_stack.roi_layers.clear()
            app.canvas._active_layer = None
            app._load_single_channel("ch1", data, 1, skipped)
        assert len(app._documents) == 1
        assert app._documents[0].name == "ch1"
        assert app.layer_stack.image_layer is not None
        assert len(skipped) == 0

    def test_downsamples_when_factor_gt_1(self, app):
        app._documents.clear()
        app._active_doc_index = -1
        data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        skipped = []
        with patch.object(app.canvas, "refresh_image"), \
             patch.object(app.canvas, "fit_to_window"), \
             patch.object(app.layer_panel, "refresh"), \
             patch.object(app.minimap, "set_image"), \
             patch.object(app, "_update_minimap_viewport"), \
             patch.object(app.adjustments_panel, "set_image_data"), \
             patch.object(app.display_panel, "set_channels"), \
             patch.object(app.toast, "show"):
            app.layer_stack.image_layer = None
            app.layer_stack.roi_layers.clear()
            app.canvas._active_layer = None
            app._load_single_channel("ch_ds", data, 2, skipped)
        # Image should be 50x50 after 2x downsample
        assert app.layer_stack.image_layer.data.shape == (50, 50)

    def test_skips_dimension_mismatch(self, app):
        app._documents.clear()
        app._active_doc_index = -1
        # Load first channel
        data1 = np.zeros((100, 100), dtype=np.uint8)
        skipped = []
        with patch.object(app.canvas, "refresh_image"), \
             patch.object(app.canvas, "fit_to_window"), \
             patch.object(app.layer_panel, "refresh"), \
             patch.object(app.minimap, "set_image"), \
             patch.object(app, "_update_minimap_viewport"), \
             patch.object(app.adjustments_panel, "set_image_data"), \
             patch.object(app.display_panel, "set_channels"), \
             patch.object(app.toast, "show"):
            app.layer_stack.image_layer = None
            app.layer_stack.roi_layers.clear()
            app.canvas._active_layer = None
            app._load_single_channel("ch1", data1, 1, skipped)
        # Now load a mismatched channel
        data2 = np.zeros((50, 50), dtype=np.uint8)
        with patch.object(app.canvas, "refresh_image"), \
             patch.object(app.minimap, "set_image"), \
             patch.object(app, "_update_minimap_viewport"), \
             patch.object(app.adjustments_panel, "set_image_data"), \
             patch.object(app.display_panel, "set_channels"), \
             patch.object(app.toast, "show"):
            app._load_single_channel("ch2", data2, 1, skipped)
        assert len(skipped) == 1
        assert "ch2" in skipped[0]


# ═══════════════════════════════════════════════════════════════════
# 16. clear_active_roi
# ═══════════════════════════════════════════════════════════════════


class TestClearActiveRoi:
    def test_removes_active_roi(self, app):
        app.layer_stack.roi_layers.clear()
        roi = _add_roi(app, "ToDelete")
        app.canvas._active_layer = roi
        app.canvas._selection._layers.clear()
        with patch.object(app.canvas, "refresh_overlays"), \
             patch.object(app.layer_panel, "refresh"), \
             patch.object(app.properties_panel, "set_layer"):
            app.clear_active_roi()
        assert len(app.layer_stack.roi_layers) == 0

    def test_removes_multiple_selected_rois(self, app):
        app.layer_stack.roi_layers.clear()
        r1 = _add_roi(app, "S1")
        r2 = _add_roi(app, "S2")
        r3 = _add_roi(app, "S3")
        app.canvas._selection._layers = [r1, r3]
        with patch.object(app.canvas, "refresh_overlays"), \
             patch.object(app.layer_panel, "refresh"), \
             patch.object(app.properties_panel, "set_layer"):
            app.clear_active_roi()
        assert len(app.layer_stack.roi_layers) == 1
        assert app.layer_stack.roi_layers[0].name == "S2"

    def test_noop_when_no_active_or_selected(self, app):
        app.layer_stack.roi_layers.clear()
        app.canvas._active_layer = None
        app.canvas._selection._layers = []
        # Should not raise
        app.clear_active_roi()


# ═══════════════════════════════════════════════════════════════════
# 17. _on_student_session_toggled
# ═══════════════════════════════════════════════════════════════════


class TestOnStudentSessionToggled:
    def test_sets_flag_true(self, app):
        app._on_student_session_toggled(True)
        assert app._student_session is True
        # In offscreen mode isVisible() is always False; use isHidden()
        assert app._student_label.isHidden() is False
        assert "Student Session" in app._student_label.text()

    def test_sets_flag_false(self, app):
        app._on_student_session_toggled(False)
        assert app._student_session is False
        assert app._student_label.isHidden() is True


# ═══════════════════════════════════════════════════════════════════
# 18. _view_instructions
# ═══════════════════════════════════════════════════════════════════


class TestViewInstructions:
    def test_no_instructions_shows_info(self, app):
        # Ensure no instructions loaded
        if hasattr(app, '_last_instructions_text'):
            delattr(app, '_last_instructions_text')
        with patch.object(QMessageBox, "information") as mock_msg:
            app._view_instructions()
            mock_msg.assert_called_once()
            assert "No instructions" in str(mock_msg.call_args)

    def test_with_instructions_shows_dialog(self, app):
        app._last_instructions_text = "Test instruction content"
        from PySide6.QtWidgets import QDialog
        with patch.object(QDialog, "show") as mock_show:
            app._view_instructions()
            mock_show.assert_called_once()
            dlg = app._instructions_dlg
            assert not dlg.isModal()
            dlg.close()


# ═══════════════════════════════════════════════════════════════════
# 19. _update_display_channels
# ═══════════════════════════════════════════════════════════════════


class TestUpdateDisplayChannels:
    def test_updates_panel_with_doc_names(self, app):
        from montaris.layers import MontageDocument
        app._documents.clear()
        app._documents.append(MontageDocument(
            name="DAPI",
            image_layer=ImageLayer("DAPI", np.zeros((100, 100), dtype=np.uint8)),
        ))
        app._documents.append(MontageDocument(
            name="GFP",
            image_layer=ImageLayer("GFP", np.zeros((100, 100), dtype=np.uint8)),
        ))
        with patch.object(app.display_panel, "set_channels") as mock_set:
            app._update_display_channels()
            mock_set.assert_called_once_with(["DAPI", "GFP"])

    def test_empty_documents_sends_empty_list(self, app):
        app._documents.clear()
        with patch.object(app.display_panel, "set_channels") as mock_set:
            app._update_display_channels()
            mock_set.assert_called_once_with([])


# ═══════════════════════════════════════════════════════════════════
# 20. flip_horizontal / rotate_90
# ═══════════════════════════════════════════════════════════════════


class TestFlipAndRotate:
    def test_flip_horizontal_negates_m11(self, app):
        # Reset to identity
        app.canvas.setTransform(QTransform())
        original_m11 = app.canvas.transform().m11()
        app.flip_horizontal()
        new_m11 = app.canvas.transform().m11()
        # m11 should flip sign
        assert new_m11 == -original_m11

    def test_double_flip_restores_transform(self, app):
        app.canvas.setTransform(QTransform())
        original = app.canvas.transform()
        app.flip_horizontal()
        app.flip_horizontal()
        restored = app.canvas.transform()
        assert abs(restored.m11() - original.m11()) < 1e-6
        assert abs(restored.m22() - original.m22()) < 1e-6

    def test_rotate_90_changes_transform(self, app):
        app.canvas.setTransform(QTransform())
        original = app.canvas.transform()
        app.rotate_90()
        rotated = app.canvas.transform()
        # After 90-degree rotation, m12 should be non-zero
        assert abs(rotated.m12()) > 0.5


# ═══════════════════════════════════════════════════════════════════
# Extra: _toggle_auto_overlap, layer_undo/redo, _on_composite_toggled
# ═══════════════════════════════════════════════════════════════════


class TestToggleAutoOverlap:
    def test_sets_flag(self, app):
        app._toggle_auto_overlap(True)
        assert app._auto_overlap is True
        app._toggle_auto_overlap(False)
        assert app._auto_overlap is False


class TestLayerUndoRedo:
    def test_layer_undo_no_active_is_noop(self, app):
        app.canvas._active_layer = None
        # Should not raise
        app.layer_undo()

    def test_layer_redo_no_active_is_noop(self, app):
        app.canvas._active_layer = None
        # Should not raise
        app.layer_redo()


class TestOnCompositToggled:
    def test_enables_composite_mode(self, app):
        with patch.object(app, "_refresh_composite"):
            app._on_composite_toggled(True)
        assert app._composite_mode is True

    def test_disables_composite_mode(self, app):
        with patch.object(app.canvas, "set_tint_color"), \
             patch.object(app.canvas, "refresh_image"), \
             patch.object(app, "_get_active_tint", return_value=None):
            app._on_composite_toggled(False)
        assert app._composite_mode is False


class TestOnSaveProgressToggled:
    def test_shows_save_button_when_checked(self, app):
        app._on_save_progress_toggled(True)
        assert app._save_progress_btn.isHidden() is False
        assert app._save_progress_shortcut.isEnabled() is True

    def test_hides_save_button_when_unchecked(self, app):
        app._on_save_progress_toggled(False)
        assert app._save_progress_btn.isHidden() is True
        assert app._save_progress_shortcut.isEnabled() is False


class TestOnSelectionCountChanged:
    def test_zero_selection(self, app):
        app._on_selection_count_changed([])
        assert "cleared" in app.statusbar.currentMessage().lower()

    def test_single_selection(self, app):
        roi = MagicMock()
        roi.name = "TestROI"
        app._on_selection_count_changed([roi])
        assert "TestROI" in app.statusbar.currentMessage()

    def test_multi_selection(self, app):
        r1 = MagicMock()
        r2 = MagicMock()
        r1.name = "A"
        r2.name = "B"
        app._on_selection_count_changed([r1, r2])
        assert "2" in app.statusbar.currentMessage()
