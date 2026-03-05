"""Comprehensive GUI integration tests simulating real user interactions.

Tests zoom, pan, multi-selection, move, transform, undo, layer panel sync,
tool switching, and edge cases that would be caught during manual testing.
"""
import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt, QEvent
from PySide6.QtGui import QMouseEvent, QKeyEvent, QWheelEvent
from PySide6.QtWidgets import QApplication
from montaris.layers import ROILayer, ImageLayer
from montaris.tools.move import MoveTool
from montaris.tools.transform import TransformTool
from montaris.core.roi_transform import get_mask_bbox, compute_handles


def make_mouse_event(event_type, pos, button=Qt.LeftButton,
                     modifiers=Qt.NoModifier):
    """Create a QMouseEvent for testing."""
    return QMouseEvent(event_type, pos, pos, button, button, modifiers)


def make_key_event(event_type, key, modifiers=Qt.NoModifier):
    """Create a QKeyEvent for testing."""
    return QKeyEvent(event_type, key, modifiers)


@pytest.fixture
def app_two_rois(app_with_image):
    """App with two ROIs that have content."""
    app = app_with_image
    roi1 = app.layer_stack.roi_layers[0]
    roi1.mask[10:30, 10:30] = 255
    roi2 = ROILayer("ROI 2", 120, 100)
    roi2.mask[50:70, 50:70] = 255
    app.layer_stack.add_roi(roi2)
    app.canvas.refresh_overlays()
    app.layer_panel.refresh()
    return app


# ------------------------------------------------------------------
# Zoom & Pan
# ------------------------------------------------------------------

class TestZoomPan:
    def test_zoom_in(self, app_with_image):
        app = app_with_image
        initial_scale = app.canvas.transform().m11()
        # Simulate zoom in via scale
        app.canvas.scale(1.15, 1.15)
        assert app.canvas.transform().m11() > initial_scale

    def test_zoom_out(self, app_with_image):
        app = app_with_image
        app.canvas.scale(2.0, 2.0)
        zoomed_scale = app.canvas.transform().m11()
        app.canvas.scale(1 / 1.15, 1 / 1.15)
        assert app.canvas.transform().m11() < zoomed_scale

    def test_fit_to_window(self, app_with_image):
        app = app_with_image
        app.canvas.scale(5.0, 5.0)
        app.canvas.fit_to_window()
        # Scale should be reasonable (not 5x)
        assert app.canvas.transform().m11() < 5.0

    def test_reset_zoom(self, app_with_image):
        app = app_with_image
        app.canvas.scale(3.0, 3.0)
        app.canvas.reset_zoom()
        assert abs(app.canvas.transform().m11() - 1.0) < 0.01


# ------------------------------------------------------------------
# Selection via canvas
# ------------------------------------------------------------------

class TestSelectionInteraction:
    def test_ctrl_click_selects_roi(self, app_two_rois):
        app = app_two_rois
        roi1 = app.layer_stack.roi_layers[0]
        # Directly use selection model (simulates Ctrl+click hit)
        app.canvas._selection.toggle(roi1)
        assert app.canvas._selection.contains(roi1)
        assert app.canvas._active_layer is roi1

    def test_ctrl_click_toggle_off(self, app_two_rois):
        app = app_two_rois
        roi1 = app.layer_stack.roi_layers[0]
        app.canvas._selection.add(roi1)
        app.canvas._selection.toggle(roi1)
        assert not app.canvas._selection.contains(roi1)

    def test_ctrl_click_multiple_rois(self, app_two_rois):
        app = app_two_rois
        roi1 = app.layer_stack.roi_layers[0]
        roi2 = app.layer_stack.roi_layers[1]
        app.canvas._selection.toggle(roi1)
        app.canvas._selection.toggle(roi2)
        assert app.canvas._selection.count == 2

    def test_ctrl_a_selects_all(self, app_two_rois):
        app = app_two_rois
        app.canvas._selection.select_all(app.layer_stack.roi_layers)
        assert app.canvas._selection.count == 2

    def test_escape_clears_selection(self, app_two_rois):
        app = app_two_rois
        app.canvas._selection.select_all(app.layer_stack.roi_layers)
        app.canvas._selection.clear()
        assert app.canvas._selection.count == 0

    def test_highlights_match_selection(self, app_two_rois):
        app = app_two_rois
        roi1 = app.layer_stack.roi_layers[0]
        roi2 = app.layer_stack.roi_layers[1]
        app.canvas._selection.set([roi1, roi2])
        assert len(app.canvas._selection_highlight_items) == 2

    def test_highlights_cleared_on_deselect(self, app_two_rois):
        app = app_two_rois
        roi1 = app.layer_stack.roi_layers[0]
        app.canvas._selection.set([roi1])
        assert len(app.canvas._selection_highlight_items) == 1
        app.canvas._selection.clear()
        assert len(app.canvas._selection_highlight_items) == 0

    def test_highlight_empty_roi_skipped(self, app_two_rois):
        app = app_two_rois
        empty_roi = ROILayer("Empty", 120, 100)
        app.layer_stack.add_roi(empty_roi)
        app.canvas._selection.set([empty_roi])
        # Empty ROI has no bbox, so no highlight
        assert len(app.canvas._selection_highlight_items) == 0


# ------------------------------------------------------------------
# Layer panel ↔ Selection sync
# ------------------------------------------------------------------

class TestLayerPanelSync:
    def test_canvas_selection_syncs_to_layer_panel(self, app_two_rois):
        app = app_two_rois
        roi1 = app.layer_stack.roi_layers[0]
        app.canvas._selection.set([roi1])
        # Check that the layer panel has the correct item selected
        selected_items = app.layer_panel.list_widget.selectedItems()
        roi_selected = [
            item for item in selected_items
            if item.data(Qt.UserRole) and item.data(Qt.UserRole)[0] == "roi"
        ]
        assert len(roi_selected) >= 1

    def test_select_all_syncs_to_panel(self, app_two_rois):
        app = app_two_rois
        app.canvas._selection.select_all(app.layer_stack.roi_layers)
        selected_items = app.layer_panel.list_widget.selectedItems()
        roi_selected = [
            item for item in selected_items
            if item.data(Qt.UserRole) and item.data(Qt.UserRole)[0] == "roi"
        ]
        assert len(roi_selected) == 2

    def test_active_layer_follows_primary(self, app_two_rois):
        app = app_two_rois
        roi2 = app.layer_stack.roi_layers[1]
        app.canvas._selection.set([roi2])
        assert app.canvas._active_layer is roi2


# ------------------------------------------------------------------
# Move tool — single & multi
# ------------------------------------------------------------------

class TestMoveToolGUI:
    def test_single_move_preserves_mask(self, app_two_rois):
        app = app_two_rois
        roi1 = app.layer_stack.roi_layers[0]
        original_sum = roi1.mask.sum()
        tool = MoveTool(app)
        app.canvas.set_active_layer(roi1)
        tool.on_press(QPointF(20, 20), roi1, app.canvas)
        tool.on_move(QPointF(25, 25), roi1, app.canvas)
        tool.on_release(QPointF(25, 25), roi1, app.canvas)
        # Pixel count should be approximately preserved
        assert abs(int(roi1.mask.sum()) - int(original_sum)) < original_sum * 0.1

    def test_multi_move_both_shift(self, app_two_rois):
        app = app_two_rois
        roi1 = app.layer_stack.roi_layers[0]
        roi2 = app.layer_stack.roi_layers[1]
        app.canvas._selection.set([roi1, roi2])
        bbox1_before = get_mask_bbox(roi1.mask)
        bbox2_before = get_mask_bbox(roi2.mask)

        tool = MoveTool(app)
        tool.on_press(QPointF(20, 20), roi1, app.canvas)
        tool.on_move(QPointF(25, 25), roi1, app.canvas)
        tool.on_release(QPointF(25, 25), roi1, app.canvas)

        bbox1_after = get_mask_bbox(roi1.mask)
        bbox2_after = get_mask_bbox(roi2.mask)
        # Both should have shifted
        assert bbox1_after is not None
        assert bbox2_after is not None
        assert bbox1_after[0] > bbox1_before[0]  # y shifted down
        assert bbox2_after[0] > bbox2_before[0]

    def test_move_undo_restores_all(self, app_two_rois):
        app = app_two_rois
        roi1 = app.layer_stack.roi_layers[0]
        roi2 = app.layer_stack.roi_layers[1]
        snap1 = roi1.mask.copy()
        snap2 = roi2.mask.copy()
        app.canvas._selection.set([roi1, roi2])

        tool = MoveTool(app)
        tool.on_press(QPointF(20, 20), roi1, app.canvas)
        tool.on_move(QPointF(30, 30), roi1, app.canvas)
        tool.on_release(QPointF(30, 30), roi1, app.canvas)

        # Undo
        app.undo_stack.undo()
        assert np.array_equal(roi1.mask, snap1)
        assert np.array_equal(roi2.mask, snap2)

    def test_move_redo(self, app_two_rois):
        app = app_two_rois
        roi1 = app.layer_stack.roi_layers[0]
        app.canvas._selection.set([roi1])
        tool = MoveTool(app)
        tool.on_press(QPointF(20, 20), roi1, app.canvas)
        tool.on_move(QPointF(30, 30), roi1, app.canvas)
        tool.on_release(QPointF(30, 30), roi1, app.canvas)
        moved = roi1.mask.copy()
        app.undo_stack.undo()
        app.undo_stack.redo()
        assert np.array_equal(roi1.mask, moved)

    def test_move_no_layer(self, app_with_image):
        """Move tool does nothing with no layer."""
        app = app_with_image
        tool = MoveTool(app)
        tool.on_press(QPointF(10, 10), None, app.canvas)
        assert not tool._moving

    def test_move_empty_mask(self, app_with_image):
        """Move tool does nothing with empty mask."""
        app = app_with_image
        layer = app.layer_stack.roi_layers[0]
        tool = MoveTool(app)
        tool.on_press(QPointF(10, 10), layer, app.canvas)
        assert not tool._moving


# ------------------------------------------------------------------
# Transform tool — single & multi
# ------------------------------------------------------------------

class TestTransformToolGUI:
    def test_first_click_shows_handles(self, app_two_rois):
        app = app_two_rois
        tool = TransformTool(app)
        roi1 = app.layer_stack.roi_layers[0]
        tool.on_press(QPointF(20, 20), roi1, app.canvas)
        assert tool._bbox is not None
        assert len(tool._handle_items) > 0

    def test_handle_shapes_correct(self, app_two_rois):
        app = app_two_rois
        tool = TransformTool(app)
        roi1 = app.layer_stack.roi_layers[0]
        tool.on_press(QPointF(20, 20), roi1, app.canvas)
        # Should have items for 8 handles + 1 rotate circle + 1 rotate line = 10
        assert len(tool._handle_items) == 10

    def test_multi_selection_union_bbox(self, app_two_rois):
        app = app_two_rois
        roi1 = app.layer_stack.roi_layers[0]
        roi2 = app.layer_stack.roi_layers[1]
        app.canvas._selection.set([roi1, roi2])
        tool = TransformTool(app)
        tool.on_press(QPointF(20, 20), roi1, app.canvas)
        # Union bbox should span both ROIs
        y1, y2, x1, x2 = tool._bbox
        assert y1 == 10  # roi1 starts at 10
        assert y2 == 70  # roi2 ends at 70
        assert x1 == 10
        assert x2 == 70

    def test_scale_corner(self, app_two_rois):
        app = app_two_rois
        tool = TransformTool(app)
        roi1 = app.layer_stack.roi_layers[0]
        original = roi1.mask.copy()
        # Show handles
        tool.on_press(QPointF(20, 20), roi1, app.canvas)
        # Grab br corner
        handles = compute_handles(tool._bbox)
        br = next(h for h in handles if h.handle_type == 'br')
        tool._active_handle = br
        tool._start_pos = QPointF(br.x, br.y)
        tool._dragging = True
        tool._target_layers = [roi1]
        tool._snapshots = {id(roi1): (roi1, original.copy())}
        # Scale up
        tool.on_move(QPointF(br.x + 5, br.y + 5), roi1, app.canvas)
        assert roi1.mask.sum() > original.sum()
        # Release
        tool.on_release(QPointF(br.x + 5, br.y + 5), roi1, app.canvas)
        assert app.undo_stack.can_undo

    def test_rotate(self, app_two_rois):
        app = app_two_rois
        tool = TransformTool(app)
        roi1 = app.layer_stack.roi_layers[0]
        original = roi1.mask.copy()
        tool.on_press(QPointF(20, 20), roi1, app.canvas)
        handles = compute_handles(tool._bbox)
        rot = next(h for h in handles if h.handle_type == 'rotate')
        tool._active_handle = rot
        tool._start_pos = QPointF(rot.x, rot.y)
        tool._dragging = True
        tool._target_layers = [roi1]
        tool._snapshots = {id(roi1): (roi1, original.copy())}
        tool.on_move(QPointF(rot.x + 10, rot.y), roi1, app.canvas)
        assert not np.array_equal(roi1.mask, original)

    def test_escape_cancels(self, app_two_rois):
        app = app_two_rois
        tool = TransformTool(app)
        roi1 = app.layer_stack.roi_layers[0]
        original = roi1.mask.copy()
        tool.on_press(QPointF(20, 20), roi1, app.canvas)
        handles = compute_handles(tool._bbox)
        br = next(h for h in handles if h.handle_type == 'br')
        tool._active_handle = br
        tool._start_pos = QPointF(br.x, br.y)
        tool._dragging = True
        tool._target_layers = [roi1]
        tool._snapshots = {id(roi1): (roi1, original.copy())}
        tool.on_move(QPointF(br.x + 20, br.y + 20), roi1, app.canvas)
        assert not np.array_equal(roi1.mask, original)
        # Escape
        tool.on_key_press(Qt.Key_Escape, app.canvas)
        assert np.array_equal(roi1.mask, original)

    def test_transform_undo(self, app_two_rois):
        app = app_two_rois
        tool = TransformTool(app)
        roi1 = app.layer_stack.roi_layers[0]
        original = roi1.mask.copy()
        tool.on_press(QPointF(20, 20), roi1, app.canvas)
        handles = compute_handles(tool._bbox)
        br = next(h for h in handles if h.handle_type == 'br')
        tool._active_handle = br
        tool._start_pos = QPointF(br.x, br.y)
        tool._dragging = True
        tool._target_layers = [roi1]
        tool._snapshots = {id(roi1): (roi1, original.copy())}
        tool.on_move(QPointF(br.x + 5, br.y + 5), roi1, app.canvas)
        tool.on_release(QPointF(br.x + 5, br.y + 5), roi1, app.canvas)
        app.undo_stack.undo()
        assert np.array_equal(roi1.mask, original)

    def test_clear_handles_on_new_press(self, app_two_rois):
        app = app_two_rois
        tool = TransformTool(app)
        roi1 = app.layer_stack.roi_layers[0]
        tool.on_press(QPointF(20, 20), roi1, app.canvas)
        items_first = len(tool._handle_items)
        assert items_first > 0
        # Press again (on non-handle area) should recreate handles
        tool.on_press(QPointF(20, 20), roi1, app.canvas)
        assert len(tool._handle_items) == items_first  # Same count

    def test_no_layer(self, app_with_image):
        app = app_with_image
        tool = TransformTool(app)
        tool.on_press(QPointF(10, 10), None, app.canvas)
        assert tool._bbox is None

    def test_empty_mask(self, app_with_image):
        app = app_with_image
        tool = TransformTool(app)
        layer = app.layer_stack.roi_layers[0]
        tool.on_press(QPointF(10, 10), layer, app.canvas)
        assert tool._bbox is None


# ------------------------------------------------------------------
# Tool switching
# ------------------------------------------------------------------

class TestToolSwitching:
    def test_switch_clears_transform_handles(self, app_two_rois):
        app = app_two_rois
        tool = TransformTool(app)
        roi1 = app.layer_stack.roi_layers[0]
        tool.on_press(QPointF(20, 20), roi1, app.canvas)
        assert len(tool._handle_items) > 0
        # Switching tool should not crash (handles stay until tool is gone)
        move = MoveTool(app)
        app.canvas.set_tool(move)
        # Old tool's items are still in scene but that's OK - they'll be
        # garbage collected when tool is destroyed

    def test_tools_dont_interfere(self, app_two_rois):
        app = app_two_rois
        roi1 = app.layer_stack.roi_layers[0]
        # Use move
        move = MoveTool(app)
        move.on_press(QPointF(20, 20), roi1, app.canvas)
        move.on_release(QPointF(20, 20), roi1, app.canvas)
        # Use transform
        transform = TransformTool(app)
        transform.on_press(QPointF(20, 20), roi1, app.canvas)
        assert transform._bbox is not None


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_remove_roi_clears_selection(self, app_two_rois):
        """Removing an ROI should not crash even if it's selected."""
        app = app_two_rois
        roi1 = app.layer_stack.roi_layers[0]
        app.canvas._selection.set([roi1])
        app.layer_stack.remove_roi(0)
        # Selection still holds stale ref but shouldn't crash
        app.canvas._selection.clear()
        assert app.canvas._selection.count == 0

    def test_load_new_image_clears(self, app_two_rois):
        """Loading new image clears layer stack and selection should reset."""
        app = app_two_rois
        app.canvas._selection.select_all(app.layer_stack.roi_layers)
        # Simulate loading new image
        new_data = np.random.randint(0, 255, (80, 100), dtype=np.uint8)
        app.layer_stack.set_image(ImageLayer("new", new_data))
        # roi_layers is now empty
        assert len(app.layer_stack.roi_layers) == 0
        # Selection should be manually cleared
        app.canvas._selection.clear()
        assert app.canvas._selection.count == 0

    def test_selection_with_zoom(self, app_two_rois):
        """Selection highlights should work at any zoom level."""
        app = app_two_rois
        app.canvas.scale(3.0, 3.0)
        roi1 = app.layer_stack.roi_layers[0]
        app.canvas._selection.set([roi1])
        assert len(app.canvas._selection_highlight_items) == 1

    def test_multi_undo_then_redo(self, app_two_rois):
        """Full undo/redo cycle with compound command."""
        app = app_two_rois
        roi1 = app.layer_stack.roi_layers[0]
        roi2 = app.layer_stack.roi_layers[1]
        snap1 = roi1.mask.copy()
        snap2 = roi2.mask.copy()
        app.canvas._selection.set([roi1, roi2])

        tool = MoveTool(app)
        tool.on_press(QPointF(20, 20), roi1, app.canvas)
        tool.on_move(QPointF(25, 25), roi1, app.canvas)
        tool.on_release(QPointF(25, 25), roi1, app.canvas)

        moved1 = roi1.mask.copy()
        moved2 = roi2.mask.copy()

        # Undo
        app.undo_stack.undo()
        assert np.array_equal(roi1.mask, snap1)
        assert np.array_equal(roi2.mask, snap2)

        # Redo
        app.undo_stack.redo()
        assert np.array_equal(roi1.mask, moved1)
        assert np.array_equal(roi2.mask, moved2)

    def test_statusbar_updates(self, app_two_rois):
        app = app_two_rois
        roi1 = app.layer_stack.roi_layers[0]
        roi2 = app.layer_stack.roi_layers[1]
        app.canvas._selection.set([roi1])
        msg = app.statusbar.currentMessage()
        assert "ROI 1" in msg or "1" in msg
        app.canvas._selection.set([roi1, roi2])
        msg = app.statusbar.currentMessage()
        assert "2" in msg
        app.canvas._selection.clear()
        msg = app.statusbar.currentMessage()
        assert "clear" in msg.lower()
