import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from montaris.core.selection import SelectionModel
from montaris.core.multi_undo import CompoundUndoCommand
from montaris.core.undo import UndoCommand
from montaris.layers import ROILayer
from montaris.tools.move import MoveTool
from montaris.tools.transform import TransformTool


# ------------------------------------------------------------------
# SelectionModel
# ------------------------------------------------------------------

class TestSelectionModel:
    def test_initial_empty(self, qapp):
        m = SelectionModel()
        assert m.layers == []
        assert m.primary is None
        assert m.count == 0

    def test_set_and_get(self, qapp):
        m = SelectionModel()
        r1 = ROILayer("A", 10, 10)
        r2 = ROILayer("B", 10, 10)
        m.set([r1, r2])
        assert m.layers == [r1, r2]
        assert m.primary is r1
        assert m.count == 2

    def test_set_no_signal_if_unchanged(self, qapp):
        m = SelectionModel()
        r1 = ROILayer("A", 10, 10)
        m.set([r1])
        calls = []
        m.changed.connect(lambda layers: calls.append(layers))
        m.set([r1])
        assert len(calls) == 0

    def test_add(self, qapp):
        m = SelectionModel()
        r1 = ROILayer("A", 10, 10)
        m.add(r1)
        assert m.contains(r1)
        assert m.count == 1

    def test_add_duplicate_ignored(self, qapp):
        m = SelectionModel()
        r1 = ROILayer("A", 10, 10)
        m.add(r1)
        m.add(r1)
        assert m.count == 1

    def test_remove(self, qapp):
        m = SelectionModel()
        r1 = ROILayer("A", 10, 10)
        m.add(r1)
        m.remove(r1)
        assert not m.contains(r1)
        assert m.count == 0

    def test_toggle(self, qapp):
        m = SelectionModel()
        r1 = ROILayer("A", 10, 10)
        m.toggle(r1)
        assert m.contains(r1)
        m.toggle(r1)
        assert not m.contains(r1)

    def test_clear(self, qapp):
        m = SelectionModel()
        r1 = ROILayer("A", 10, 10)
        m.add(r1)
        m.clear()
        assert m.count == 0

    def test_clear_empty_no_signal(self, qapp):
        m = SelectionModel()
        calls = []
        m.changed.connect(lambda layers: calls.append(layers))
        m.clear()
        assert len(calls) == 0

    def test_select_all(self, qapp):
        m = SelectionModel()
        r1 = ROILayer("A", 10, 10)
        r2 = ROILayer("B", 10, 10)
        m.select_all([r1, r2])
        assert m.count == 2
        assert m.contains(r1)
        assert m.contains(r2)

    def test_hit_test(self, qapp):
        r1 = ROILayer("A", 50, 50)
        r2 = ROILayer("B", 50, 50)
        r1.mask[10:20, 10:20] = 255
        r2.mask[15:25, 15:25] = 255
        # r2 is later in list (higher z-order), should be returned first
        hit = SelectionModel.hit_test(17, 17, [r1, r2])
        assert hit is r2

    def test_hit_test_miss(self, qapp):
        r1 = ROILayer("A", 50, 50)
        r1.mask[10:20, 10:20] = 255
        hit = SelectionModel.hit_test(30, 30, [r1])
        assert hit is None

    def test_hit_test_out_of_bounds(self, qapp):
        r1 = ROILayer("A", 50, 50)
        r1.mask[10:20, 10:20] = 255
        hit = SelectionModel.hit_test(100, 100, [r1])
        assert hit is None

    def test_changed_signal(self, qapp):
        m = SelectionModel()
        r1 = ROILayer("A", 10, 10)
        calls = []
        m.changed.connect(lambda layers: calls.append(list(layers)))
        m.add(r1)
        assert len(calls) == 1
        assert calls[0] == [r1]


# ------------------------------------------------------------------
# CompoundUndoCommand
# ------------------------------------------------------------------

class TestCompoundUndoCommand:
    def test_undo_redo(self, qapp):
        r1 = ROILayer("A", 20, 20)
        r2 = ROILayer("B", 20, 20)
        r1.mask[5:10, 5:10] = 255
        r2.mask[5:10, 5:10] = 255
        snap1 = r1.mask.copy()
        snap2 = r2.mask.copy()
        r1.mask[:] = 0
        r2.mask[:] = 0

        cmd1 = UndoCommand(r1, (0, 20, 0, 20), snap1, r1.mask.copy())
        cmd2 = UndoCommand(r2, (0, 20, 0, 20), snap2, r2.mask.copy())
        compound = CompoundUndoCommand([cmd1, cmd2])

        # Undo should restore both
        compound.undo()
        assert r1.mask[5:10, 5:10].sum() > 0
        assert r2.mask[5:10, 5:10].sum() > 0

        # Redo should clear both
        compound.redo()
        assert r1.mask.sum() == 0
        assert r2.mask.sum() == 0


# ------------------------------------------------------------------
# Canvas selection (Ctrl+click, Ctrl+A, Escape)
# ------------------------------------------------------------------

class TestCanvasSelection:
    def test_ctrl_a_selects_all(self, app_with_image):
        app = app_with_image
        roi2 = ROILayer("ROI 2", 120, 100)
        app.layer_stack.add_roi(roi2)
        app.canvas._selection.select_all(app.layer_stack.roi_layers)
        assert app.canvas._selection.count == 2

    def test_escape_clears(self, app_with_image):
        app = app_with_image
        roi = app.layer_stack.roi_layers[0]
        app.canvas._selection.add(roi)
        assert app.canvas._selection.count == 1
        app.canvas._selection.clear()
        assert app.canvas._selection.count == 0

    def test_selection_syncs_active_layer(self, app_with_image):
        app = app_with_image
        roi2 = ROILayer("ROI 2", 120, 100)
        app.layer_stack.add_roi(roi2)
        app.canvas._selection.set([roi2])
        assert app.canvas._active_layer is roi2

    def test_selection_highlights_created(self, app_with_image):
        app = app_with_image
        roi = app.layer_stack.roi_layers[0]
        roi.mask[10:30, 10:30] = 255
        app.canvas._selection.set([roi])
        assert len(app.canvas._selection_highlight_items) == 1

    def test_selection_highlights_cleared(self, app_with_image):
        app = app_with_image
        roi = app.layer_stack.roi_layers[0]
        roi.mask[10:30, 10:30] = 255
        app.canvas._selection.set([roi])
        app.canvas._selection.clear()
        assert len(app.canvas._selection_highlight_items) == 0


# ------------------------------------------------------------------
# Multi-selection Move
# ------------------------------------------------------------------

class TestMultiSelectionMove:
    def test_single_layer_move(self, app_with_image):
        app = app_with_image
        tool = MoveTool(app)
        layer = app.layer_stack.roi_layers[0]
        layer.mask[20:40, 20:40] = 255
        tool.on_press(QPointF(30, 30), layer, app.canvas)
        tool.on_move(QPointF(40, 40), layer, app.canvas)
        tool.on_release(QPointF(40, 40), layer, app.canvas)
        # Should have offset (mask unchanged)
        assert layer.offset_x == 10 and layer.offset_y == 10
        # After flattening, mask is at new position
        layer.flatten_offset()
        assert layer.mask[30:50, 30:50].sum() > 0

    def test_multi_layer_move(self, app_with_image):
        app = app_with_image
        roi1 = app.layer_stack.roi_layers[0]
        roi2 = ROILayer("ROI 2", 120, 100)
        app.layer_stack.add_roi(roi2)
        roi1.mask[10:20, 10:20] = 255
        roi2.mask[30:40, 30:40] = 255
        # Select both
        app.canvas._selection.set([roi1, roi2])
        tool = MoveTool(app)
        tool.on_press(QPointF(15, 15), roi1, app.canvas)
        tool.on_move(QPointF(25, 25), roi1, app.canvas)
        tool.on_release(QPointF(25, 25), roi1, app.canvas)
        # Both should have offset of (10, 10)
        assert roi1.offset_x == 10 and roi1.offset_y == 10
        assert roi2.offset_x == 10 and roi2.offset_y == 10
        # After flattening, mask data is at new position
        roi1.flatten_offset()
        roi2.flatten_offset()
        assert roi1.mask[20:30, 20:30].sum() > 0
        assert roi2.mask[40:50, 40:50].sum() > 0

    def test_multi_move_compound_undo(self, app_with_image):
        app = app_with_image
        roi1 = app.layer_stack.roi_layers[0]
        roi2 = ROILayer("ROI 2", 120, 100)
        app.layer_stack.add_roi(roi2)
        roi1.mask[10:20, 10:20] = 255
        roi2.mask[30:40, 30:40] = 255
        app.canvas._selection.set([roi1, roi2])
        tool = MoveTool(app)
        tool.on_press(QPointF(15, 15), roi1, app.canvas)
        tool.on_move(QPointF(25, 25), roi1, app.canvas)
        tool.on_release(QPointF(25, 25), roi1, app.canvas)
        assert roi1.offset_x == 10
        # Single undo should restore both offsets
        app.undo_stack.undo()
        assert roi1.offset_x == 0 and roi1.offset_y == 0
        assert roi2.offset_x == 0 and roi2.offset_y == 0


# ------------------------------------------------------------------
# Multi-selection Transform
# ------------------------------------------------------------------

class TestMultiSelectionTransform:
    def test_single_layer_transform(self, app_with_image):
        app = app_with_image
        tool = TransformTool(app)
        layer = app.layer_stack.roi_layers[0]
        layer.mask[30:60, 30:60] = 255
        # First press to show handles
        tool.on_press(QPointF(45, 45), layer, app.canvas)
        assert tool._bbox is not None

    def test_multi_layer_union_bbox(self, app_with_image):
        app = app_with_image
        tool = TransformTool(app)
        roi1 = app.layer_stack.roi_layers[0]
        roi2 = ROILayer("ROI 2", 120, 100)
        app.layer_stack.add_roi(roi2)
        roi1.mask[10:20, 10:20] = 255
        roi2.mask[60:80, 60:80] = 255
        app.canvas._selection.set([roi1, roi2])
        bbox = tool._compute_union_bbox([roi1, roi2])
        assert bbox == (10, 80, 10, 80)

    def test_escape_cancels_transform(self, app_with_image):
        app = app_with_image
        tool = TransformTool(app)
        layer = app.layer_stack.roi_layers[0]
        layer.mask[30:60, 30:60] = 255
        original = layer.mask.copy()
        # Show handles
        tool.on_press(QPointF(45, 45), layer, app.canvas)
        # Grab a corner handle
        from montaris.core.roi_transform import compute_handles
        handles = compute_handles(tool._bbox)
        corner = next(h for h in handles if h.handle_type == 'tl')
        tool._active_handle = corner
        tool._start_pos = QPointF(corner.x, corner.y)
        tool._dragging = True
        tool._target_layers = [layer]
        tool._snapshots = {id(layer): (layer, original.copy())}
        # Move handle (preview only, mask not rasterized yet)
        tool.on_move(QPointF(corner.x + 10, corner.y + 10), layer, app.canvas)
        # Escape should restore mask to original
        tool.on_key_press(Qt.Key_Escape, app.canvas)
        assert np.array_equal(layer.mask, original)


# ------------------------------------------------------------------
# App integration
# ------------------------------------------------------------------

class TestAppSelection:
    def test_select_all_rois_action(self, app_with_image):
        app = app_with_image
        roi2 = ROILayer("ROI 2", 120, 100)
        app.layer_stack.add_roi(roi2)
        app.layer_panel.refresh()
        app._select_all_rois()
        assert app.canvas._selection.count == 2

    def test_statusbar_shows_count(self, app_with_image):
        app = app_with_image
        roi = app.layer_stack.roi_layers[0]
        app.canvas._selection.set([roi])
        msg = app.statusbar.currentMessage()
        assert "ROI 1" in msg or "Selected" in msg
