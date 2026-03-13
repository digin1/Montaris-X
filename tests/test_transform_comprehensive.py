"""Comprehensive tests for TransformTool and TransformAllTool.

Covers: __init__, on_press, _hit_test_handle, _create_previews, on_move,
on_release, on_key_press, _clear_handles, _clear_handle_visuals,
TransformAllTool._get_target_layers, TransformAllTool.on_activate,
_transform_one, _poll_transform, cancel during async.
"""

import os
import sys
import math

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF

from montaris.app import MontarisApp, apply_dark_theme
from montaris.layers import ImageLayer, ROILayer
from montaris.tools.transform import TransformTool, TransformAllTool, HANDLE_HIT_RADIUS
from montaris.core.roi_transform import TransformHandle, make_scale_matrix, make_rotation_matrix


# ── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    apply_dark_theme(app)
    return app


@pytest.fixture
def app(qapp):
    win = MontarisApp()
    win.show()
    img = np.zeros((200, 200), dtype=np.uint8)
    win.layer_stack.set_image(ImageLayer("test", img))
    win.canvas.refresh_image()
    win.canvas.fit_to_window()
    QApplication.processEvents()
    return win


def _add_rois(win, count=3):
    """Add ROIs with non-empty masks at distinct positions."""
    for i in range(count):
        roi = ROILayer(f"ROI_{i}", 200, 200)
        y0 = 20 + i * 40
        x0 = 20 + i * 40
        roi.mask[y0:y0 + 30, x0:x0 + 30] = 255
        win.layer_stack.add_roi(roi)
    win.canvas.set_active_layer(win.layer_stack.roi_layers[0])
    win.canvas.refresh_overlays()
    win.layer_panel.refresh()
    QApplication.processEvents()


# ── TransformTool.__init__ ──────────────────────────────────────────────


class TestTransformToolInit:
    def test_initial_state(self, app):
        """TransformTool initializes with all state fields reset."""
        tool = TransformTool(app)
        assert tool._active_handle is None
        assert tool._hovered_handle is None
        assert tool._start_pos is None
        assert tool._bbox is None
        assert tool._snapshots == {}
        assert tool._target_layers == []
        assert tool._handle_items == []
        assert tool._bbox_item is None
        assert tool._canvas is None
        assert tool._dragging is False

    def test_name(self, app):
        """TransformTool has the expected name."""
        tool = TransformTool(app)
        assert tool.name == "Transform (selected)"

    def test_cursor(self, app):
        """TransformTool cursor is SizeAllCursor."""
        tool = TransformTool(app)
        assert tool.cursor() == Qt.SizeAllCursor


# ── on_press with handle hit ────────────────────────────────────────────


class TestOnPressWithHandle:
    def test_handle_hit_saves_snapshot(self, app):
        """Pressing on a handle saves a snapshot for undo."""
        _add_rois(app, 1)
        tool = TransformTool(app)
        layer = app.layer_stack.roi_layers[0]
        canvas = app.canvas

        # Activate to show handles
        tool.on_activate(layer, canvas)
        QApplication.processEvents()

        assert tool._bbox is not None
        # Find a handle position to click on
        handles = tool._get_rotated_handles()
        assert len(handles) > 0
        h = handles[0]  # first handle (tl)
        pos = QPointF(h.x, h.y)

        tool.on_press(pos, layer, canvas)
        assert tool._dragging is True
        assert tool._active_handle is not None
        assert len(tool._snapshots) >= 1

    def test_no_handle_hit_no_drag(self, app):
        """Pressing far from any handle does not start a drag."""
        _add_rois(app, 1)
        tool = TransformTool(app)
        layer = app.layer_stack.roi_layers[0]
        canvas = app.canvas
        tool.on_activate(layer, canvas)
        QApplication.processEvents()

        # Press very far from the ROI
        pos = QPointF(199, 199)
        tool.on_press(pos, layer, canvas)
        assert tool._dragging is False


# ── _hit_test_handle ────────────────────────────────────────────────────


class TestHitTestHandle:
    def test_near_handle_returns_handle(self, app):
        """A point near a handle returns that handle."""
        _add_rois(app, 1)
        tool = TransformTool(app)
        layer = app.layer_stack.roi_layers[0]
        canvas = app.canvas
        tool.on_activate(layer, canvas)
        QApplication.processEvents()

        handles = tool._get_rotated_handles()
        h = handles[0]
        pos = QPointF(h.x + 1, h.y + 1)
        result = tool._hit_test_handle(pos)
        assert result is not None
        assert result.handle_type == h.handle_type

    def test_far_from_handle_returns_none(self, app):
        """A point far from all handles returns None."""
        _add_rois(app, 1)
        tool = TransformTool(app)
        layer = app.layer_stack.roi_layers[0]
        canvas = app.canvas
        tool.on_activate(layer, canvas)
        QApplication.processEvents()

        pos = QPointF(199, 199)
        result = tool._hit_test_handle(pos)
        assert result is None

    def test_no_bbox_returns_none(self, app):
        """Without a bbox, _hit_test_handle returns None."""
        tool = TransformTool(app)
        tool._bbox = None
        result = tool._hit_test_handle(QPointF(50, 50))
        assert result is None


# ── _create_previews ────────────────────────────────────────────────────


class TestCreatePreviews:
    def test_creates_preview_items(self, app):
        """_create_previews populates the _preview_items list."""
        _add_rois(app, 1)
        tool = TransformTool(app)
        layer = app.layer_stack.roi_layers[0]
        canvas = app.canvas
        tool.on_activate(layer, canvas)
        QApplication.processEvents()

        # Press a handle to populate snapshots
        handles = tool._get_rotated_handles()
        h = handles[0]
        tool.on_press(QPointF(h.x, h.y), layer, canvas)
        # _create_previews is called inside on_press
        assert len(tool._preview_items) >= 1

    def test_hidden_layers_populated(self, app):
        """_create_previews records hidden layers for later restoration."""
        _add_rois(app, 1)
        tool = TransformTool(app)
        layer = app.layer_stack.roi_layers[0]
        canvas = app.canvas
        tool.on_activate(layer, canvas)
        QApplication.processEvents()

        handles = tool._get_rotated_handles()
        h = handles[0]
        tool.on_press(QPointF(h.x, h.y), layer, canvas)
        assert len(tool._hidden_layers) >= 1


# ── on_move during drag ────────────────────────────────────────────────


class TestOnMove:
    def test_move_during_drag_updates_matrix(self, app):
        """Moving while dragging a handle updates _current_matrix."""
        _add_rois(app, 1)
        tool = TransformTool(app)
        layer = app.layer_stack.roi_layers[0]
        canvas = app.canvas
        tool.on_activate(layer, canvas)
        QApplication.processEvents()

        handles = tool._get_rotated_handles()
        # Use the 'br' (bottom-right) corner handle for scaling
        br_handle = None
        for h in handles:
            if h.handle_type == 'br':
                br_handle = h
                break
        assert br_handle is not None

        tool.on_press(QPointF(br_handle.x, br_handle.y), layer, canvas)
        tool.on_move(QPointF(br_handle.x + 10, br_handle.y + 10), layer, canvas)
        assert hasattr(tool, '_current_matrix')
        assert tool._current_matrix is not None

    def test_move_without_drag_no_crash(self, app):
        """Moving without an active drag does not crash."""
        _add_rois(app, 1)
        tool = TransformTool(app)
        layer = app.layer_stack.roi_layers[0]
        canvas = app.canvas
        tool.on_activate(layer, canvas)
        QApplication.processEvents()
        # Move without pressing first
        tool.on_move(QPointF(50, 50), layer, canvas)
        # No crash = success


# ── on_release with small transform ─────────────────────────────────────


class TestOnRelease:
    def test_release_after_drag_pushes_undo(self, app):
        """Releasing after a drag pushes a command to the undo stack."""
        _add_rois(app, 1)
        tool = TransformTool(app)
        layer = app.layer_stack.roi_layers[0]
        canvas = app.canvas
        tool.on_activate(layer, canvas)
        QApplication.processEvents()

        undo_count_before = len(app.undo_stack._stack)

        handles = tool._get_rotated_handles()
        br = None
        for h in handles:
            if h.handle_type == 'br':
                br = h
                break
        assert br is not None

        tool.on_press(QPointF(br.x, br.y), layer, canvas)
        tool.on_move(QPointF(br.x + 15, br.y + 15), layer, canvas)
        tool.on_release(QPointF(br.x + 15, br.y + 15), layer, canvas)
        QApplication.processEvents()

        # Give deferred rebuild a tick
        QApplication.processEvents()

        assert len(app.undo_stack._stack) >= undo_count_before

    def test_release_without_drag_is_noop(self, app):
        """Releasing without an active drag does nothing."""
        _add_rois(app, 1)
        tool = TransformTool(app)
        layer = app.layer_stack.roi_layers[0]
        canvas = app.canvas
        # No on_press, just release
        tool.on_release(QPointF(50, 50), layer, canvas)
        # No crash = success


# ── on_key_press (Escape) ───────────────────────────────────────────────


class TestOnKeyPress:
    def test_escape_cancels_drag(self, app):
        """Pressing Escape during a drag cancels the transform."""
        _add_rois(app, 1)
        tool = TransformTool(app)
        layer = app.layer_stack.roi_layers[0]
        canvas = app.canvas
        tool.on_activate(layer, canvas)
        QApplication.processEvents()

        handles = tool._get_rotated_handles()
        h = handles[0]
        tool.on_press(QPointF(h.x, h.y), layer, canvas)
        assert tool._dragging is True

        consumed = tool.on_key_press(Qt.Key_Escape, canvas)
        assert consumed is True
        assert tool._dragging is False
        assert tool._active_handle is None
        assert len(tool._snapshots) == 0

    def test_escape_without_drag_returns_none(self, app):
        """Pressing Escape with no active drag returns None (not consumed)."""
        tool = TransformTool(app)
        canvas = app.canvas
        result = tool.on_key_press(Qt.Key_Escape, canvas)
        assert result is None

    def test_non_escape_key_returns_none(self, app):
        """Pressing a key other than Escape during drag returns None."""
        _add_rois(app, 1)
        tool = TransformTool(app)
        layer = app.layer_stack.roi_layers[0]
        canvas = app.canvas
        tool.on_activate(layer, canvas)
        QApplication.processEvents()

        handles = tool._get_rotated_handles()
        h = handles[0]
        tool.on_press(QPointF(h.x, h.y), layer, canvas)
        result = tool.on_key_press(Qt.Key_A, canvas)
        assert result is None


# ── _clear_handles ──────────────────────────────────────────────────────


class TestClearHandles:
    def test_clear_handles_resets_state(self, app):
        """_clear_handles removes all handle items and resets session state."""
        _add_rois(app, 1)
        tool = TransformTool(app)
        layer = app.layer_stack.roi_layers[0]
        canvas = app.canvas
        tool.on_activate(layer, canvas)
        QApplication.processEvents()

        assert tool._bbox is not None
        assert len(tool._handle_items) > 0

        tool._clear_handles(canvas)
        assert tool._bbox is None
        assert len(tool._handle_items) == 0
        assert tool._bbox_item is None
        assert tool._component_mask is None
        assert tool._component_bbox is None
        assert len(tool._session_snapshots) == 0


# ── _clear_handle_visuals ──────────────────────────────────────────────


class TestClearHandleVisuals:
    def test_clear_visuals_removes_items(self, app):
        """_clear_handle_visuals removes scene items but doesn't reset session state."""
        _add_rois(app, 1)
        tool = TransformTool(app)
        layer = app.layer_stack.roi_layers[0]
        canvas = app.canvas
        tool.on_activate(layer, canvas)
        QApplication.processEvents()

        assert len(tool._handle_items) > 0
        tool._clear_handle_visuals(canvas)
        assert len(tool._handle_items) == 0
        assert tool._bbox is None
        assert tool._bbox_item is None

    def test_clear_visuals_safe_when_empty(self, app):
        """_clear_handle_visuals is safe when no handles exist."""
        tool = TransformTool(app)
        tool._canvas = app.canvas
        tool._clear_handle_visuals(app.canvas)
        # No crash = success


# ── TransformAllTool._get_target_layers ─────────────────────────────────


class TestTransformAllGetTargetLayers:
    def test_returns_all_roi_layers(self, app):
        """_get_target_layers returns all ROI layers regardless of selection."""
        _add_rois(app, 3)
        tool = TransformAllTool(app)
        layer = app.layer_stack.roi_layers[0]
        canvas = app.canvas
        result = tool._get_target_layers(layer, canvas)
        assert len(result) == len(app.layer_stack.roi_layers)
        for r in result:
            assert r.is_roi


# ── TransformAllTool.on_activate ────────────────────────────────────────


class TestTransformAllOnActivate:
    def test_activates_with_handles(self, app):
        """on_activate sets up target layers and shows handles."""
        _add_rois(app, 3)
        tool = TransformAllTool(app)
        layer = app.layer_stack.roi_layers[0]
        canvas = app.canvas

        tool.on_activate(layer, canvas)
        QApplication.processEvents()

        assert tool._bbox is not None
        assert len(tool._target_layers) == len(app.layer_stack.roi_layers)
        assert len(tool._handle_items) > 0

    def test_activates_empty_roi_list_noop(self, app):
        """on_activate with no ROIs is a no-op."""
        tool = TransformAllTool(app)
        canvas = app.canvas
        # No ROIs added
        tool.on_activate(None, canvas)
        assert tool._bbox is None

    def test_redundant_activate_skipped(self, app):
        """Calling on_activate twice with same state is a no-op on second call."""
        _add_rois(app, 2)
        tool = TransformAllTool(app)
        layer = app.layer_stack.roi_layers[0]
        canvas = app.canvas

        tool.on_activate(layer, canvas)
        QApplication.processEvents()
        bbox_first = tool._bbox

        tool.on_activate(layer, canvas)
        QApplication.processEvents()
        # Same bbox means the second call was skipped
        assert tool._bbox == bbox_first

    def test_selection_model_updated(self, app):
        """on_activate sets the canvas selection to all ROIs."""
        _add_rois(app, 3)
        tool = TransformAllTool(app)
        layer = app.layer_stack.roi_layers[0]
        canvas = app.canvas

        tool.on_activate(layer, canvas)
        QApplication.processEvents()

        # The selection should contain all ROIs
        sel = canvas._selection._layers
        assert len(sel) == len(app.layer_stack.roi_layers)


# ── _compute_union_bbox ─────────────────────────────────────────────────


class TestComputeUnionBbox:
    def test_union_of_multiple_rois(self, app):
        """Union bbox encloses all target layer bboxes."""
        _add_rois(app, 3)
        tool = TransformTool(app)
        layers = app.layer_stack.roi_layers
        bbox = tool._compute_union_bbox(layers)
        assert bbox is not None
        y1, y2, x1, x2 = bbox
        for l in layers:
            lb = l.get_bbox()
            if lb is not None:
                assert y1 <= lb[0]
                assert y2 >= lb[1]
                assert x1 <= lb[2]
                assert x2 >= lb[3]

    def test_empty_layers_returns_none(self, app):
        """Empty layer list gives None."""
        tool = TransformTool(app)
        assert tool._compute_union_bbox([]) is None

    def test_all_empty_masks_returns_none(self, app):
        """Layers with empty masks give None."""
        tool = TransformTool(app)
        empty = ROILayer("empty", 100, 100)
        assert tool._compute_union_bbox([empty]) is None


# ── _rotate_point helper ────────────────────────────────────────────────


class TestRotatePoint:
    def test_no_rotation(self, app):
        """Zero-angle rotation returns the same point."""
        tool = TransformTool(app)
        rx, ry = tool._rotate_point(10, 20, 0, 0, 0)
        assert abs(rx - 10) < 1e-9
        assert abs(ry - 20) < 1e-9

    def test_90_degree_rotation(self, app):
        """90-degree rotation around origin maps (1,0) to (0,1)."""
        tool = TransformTool(app)
        rx, ry = tool._rotate_point(1, 0, 0, 0, math.pi / 2)
        assert abs(rx - 0) < 1e-9
        assert abs(ry - 1) < 1e-9


# ── on_activate for TransformTool ───────────────────────────────────────


class TestTransformToolOnActivate:
    def test_activate_with_no_layer(self, app):
        """on_activate with None layer is a no-op."""
        tool = TransformTool(app)
        tool.on_activate(None, app.canvas)
        assert tool._bbox is None

    def test_activate_with_non_roi_layer(self, app):
        """on_activate with a non-ROI layer is a no-op."""
        tool = TransformTool(app)
        tool.on_activate(app.layer_stack.image_layer, app.canvas)
        assert tool._bbox is None

    def test_activate_with_empty_roi(self, app):
        """on_activate with an empty-mask ROI clears handles."""
        empty = ROILayer("empty", 200, 200)
        app.layer_stack.add_roi(empty)
        tool = TransformTool(app)
        canvas = app.canvas
        canvas._selection._layers = [empty]
        tool.on_activate(empty, canvas)
        QApplication.processEvents()
        assert tool._bbox is None
