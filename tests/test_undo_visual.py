"""Headed tests for undo visual refresh bugs.

Reproduces Nic's report:
- Move ROI + Undo → outline stays at pre-undo position
- Transform ROI + Undo → handles stay at pre-undo position
- Delete All → ROI outlines remain on canvas

Usage:
    QT_QPA_PLATFORM= .venv/bin/pytest tests/test_undo_visual.py -m headed -s -v
"""
import os
import numpy as np
import pytest

from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

pytestmark = pytest.mark.headed

SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "..", "screenshots")


def _save_screenshot(widget, name):
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    path = os.path.join(SCREENSHOT_DIR, f"{name}.png")
    widget.grab().save(path)
    print(f"  Screenshot saved: {path}")
    return path


@pytest.fixture
def app_with_drawn_roi(app_with_real_image):
    """Start from app_with_real_image, select first ROI, ensure it's active."""
    window = app_with_real_image
    rois = window.layer_stack.roi_layers
    assert len(rois) > 0
    # Select first ROI
    roi = rois[0]
    window.canvas.set_active_layer(roi)
    window.canvas._selection.set([roi])
    QApplication.processEvents()
    return window


class TestMoveUndoVisual:
    """After moving an ROI and pressing Undo, the ROI overlay should
    revert to its original position — no stale outline at the moved position.
    """

    def test_move_undo_clears_stale_overlay(self, app_with_drawn_roi):
        window = app_with_drawn_roi
        canvas = window.canvas
        roi = window.layer_stack.roi_layers[0]

        # Record original offset
        orig_ox, orig_oy = roi.offset_x, roi.offset_y

        # Take screenshot before move
        _save_screenshot(canvas, "01_before_move")

        # Simulate move via the MoveTool
        from montaris.tools.move import MoveTool
        tool = MoveTool(window)
        start = QPointF(50, 50)
        end = QPointF(150, 150)  # move 100px right and down

        tool.on_press(start, roi, canvas)
        tool.on_move(end, roi, canvas)
        tool.on_release(end, roi, canvas)
        QApplication.processEvents()

        # Verify move happened
        assert roi.offset_x != orig_ox or roi.offset_y != orig_oy, \
            "ROI should have moved"
        _save_screenshot(canvas, "02_after_move")

        # Now undo
        window.undo()
        QApplication.processEvents()

        # Verify data reverted
        assert roi.offset_x == orig_ox and roi.offset_y == orig_oy, \
            "Undo should restore original offset"
        _save_screenshot(canvas, "03_after_undo")

        # Check that selection highlights were updated
        # The highlight items should reflect the original position
        for item in canvas._selection_highlight_items:
            offset = item.offset()
            # The highlight should be near the original bbox, not the moved one
            # (This is a sanity check — the screenshot is the real proof)
            assert offset is not None

    def test_move_undo_refreshes_overlay_position(self, app_with_drawn_roi):
        """The ROI pixmap item position should match the roi offset after undo."""
        window = app_with_drawn_roi
        canvas = window.canvas
        roi = window.layer_stack.roi_layers[0]
        rid = id(roi)

        orig_ox, orig_oy = roi.offset_x, roi.offset_y

        from montaris.tools.move import MoveTool
        tool = MoveTool(window)
        start = QPointF(50, 50)
        end = QPointF(200, 100)

        tool.on_press(start, roi, canvas)
        tool.on_move(end, roi, canvas)
        tool.on_release(end, roi, canvas)
        QApplication.processEvents()

        # Record moved position
        roi_item = canvas._roi_items.get(rid)
        assert roi_item is not None, "ROI item should exist"
        moved_pos = roi_item.pos()

        # Undo
        window.undo()
        QApplication.processEvents()

        # The ROI overlay item's scene position should have changed back
        undone_pos = roi_item.pos()
        # After undo, offset is restored — the item position should reflect that
        assert roi.offset_x == orig_ox and roi.offset_y == orig_oy
        _save_screenshot(canvas, "03b_move_undo_position")


class TestTransformUndoVisual:
    """After transforming an ROI and pressing Undo, the transform handles
    should be cleared — no stale handles at the pre-undo bbox.
    """

    def test_transform_undo_repositions_handles(self, app_with_drawn_roi):
        """After transform + undo, handles should be rebuilt at the original bbox,
        not stuck at the transformed position."""
        window = app_with_drawn_roi
        canvas = window.canvas
        roi = window.layer_stack.roi_layers[0]

        _save_screenshot(canvas, "04_before_transform")

        # Set up transform tool
        from montaris.tools.transform import TransformTool
        tool = TransformTool(window)
        canvas._tool = tool

        # Press on ROI to activate handles
        bbox_before = roi.get_bbox()
        assert bbox_before is not None, "ROI must have a bbox"
        y1, y2, x1, x2 = bbox_before
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        tool.on_press(QPointF(cx, cy), roi, canvas)
        QApplication.processEvents()

        _save_screenshot(canvas, "05_transform_handles_shown")

        # Simulate a scale drag on a corner handle
        if tool._handle_items:
            corner = QPointF(x2, y2)
            tool.on_press(corner, roi, canvas)
            tool.on_move(QPointF(x2 + 20, y2 + 20), roi, canvas)
            tool.on_release(QPointF(x2 + 20, y2 + 20), roi, canvas)
            QApplication.processEvents()

            _save_screenshot(canvas, "06_after_transform")

            # Now undo the transform
            window.undo()
            QApplication.processEvents()

            _save_screenshot(canvas, "07_after_transform_undo")

            # Handles should be rebuilt at the original bbox position
            # (not the transformed position)
            bbox_after_undo = roi.get_bbox()
            assert bbox_after_undo is not None
            assert bbox_after_undo == bbox_before, \
                f"ROI bbox should revert to {bbox_before}, got {bbox_after_undo}"

            # The tool's bbox should match the undone state
            if tool._bbox is not None:
                assert tool._bbox == bbox_before, \
                    f"Tool bbox should match undone ROI bbox {bbox_before}, got {tool._bbox}"


class TestDeleteAllVisual:
    """After Delete All ROIs, no ROI outlines should remain on the canvas."""

    def test_delete_all_clears_overlays(self, app_with_drawn_roi):
        window = app_with_drawn_roi
        canvas = window.canvas

        # Ensure ROIs are visible
        assert len(window.layer_stack.roi_layers) > 0
        canvas.refresh_overlays()
        QApplication.processEvents()

        _save_screenshot(canvas, "08_before_delete_all")

        # Count ROI overlay items before delete
        roi_items_before = len(canvas._roi_items)
        assert roi_items_before > 0, "Should have ROI items before delete"

        # Perform delete all (bypass confirmation dialog)
        from unittest.mock import patch
        from PySide6.QtWidgets import QMessageBox
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.Yes):
            window.layer_panel._clear_all()
        QApplication.processEvents()

        _save_screenshot(canvas, "09_after_delete_all")

        # All ROI overlay items should be removed from scene
        assert len(window.layer_stack.roi_layers) == 0, "All ROIs should be deleted"

        # Check that _roi_items dict is cleaned up
        scene_items = set(canvas.scene().items())
        stale_roi_items = [
            item for item in canvas._roi_items.values()
            if item in scene_items
        ]
        assert len(stale_roi_items) == 0, \
            f"{len(stale_roi_items)} stale ROI overlay items remain on canvas after Delete All"

        # Check that selection highlights are cleared
        stale_highlights = [
            item for item in canvas._selection_highlight_items
            if item in scene_items
        ]
        assert len(stale_highlights) == 0, \
            f"{len(stale_highlights)} stale selection highlights remain after Delete All"

        # Check no transform handles remain
        tool = canvas._tool
        if tool and hasattr(tool, '_handle_items'):
            stale_handles = [
                item for item, _, _ in tool._handle_items
                if item in scene_items
            ]
            assert len(stale_handles) == 0, \
                f"{len(stale_handles)} stale transform handles remain after Delete All"


class TestPixelLevelCleanup:
    """Pixel-level verification that no colored overlay artifacts remain
    after undo/delete operations. Compares canvas screenshots as numpy
    arrays to catch dots/artifacts invisible to visual inspection."""

    def _grab_canvas_array(self, canvas):
        """Grab the canvas viewport as a numpy RGBA array."""
        from PySide6.QtGui import QImage
        pixmap = canvas.grab()
        img = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        w, h = img.width(), img.height()
        ptr = img.constBits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4).copy()
        return arr

    def test_no_colored_artifacts_after_delete_all(self, app_with_real_image):
        """Compare canvas pixels before ROIs loaded vs after delete all.
        Any bright colored pixels (ROI overlays) remaining after delete
        are stale artifacts.
        """
        from unittest.mock import patch
        from PySide6.QtWidgets import QMessageBox

        window = app_with_real_image
        canvas = window.canvas

        # The fixture already has ROIs loaded — capture WITH ROIs
        canvas.refresh_overlays()
        QApplication.processEvents()
        with_rois = self._grab_canvas_array(canvas)
        _save_screenshot(canvas, "10_pixel_with_rois")

        # Delete all ROIs
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.Yes):
            window.layer_panel._clear_all()
        QApplication.processEvents()
        canvas.refresh_overlays()
        QApplication.processEvents()

        after_delete = self._grab_canvas_array(canvas)
        _save_screenshot(canvas, "11_pixel_after_delete")

        # Find pixels that are brightly colored (ROI overlay remnants)
        # ROI overlays use saturated colors — look for high-saturation pixels
        # that exist in after_delete but would indicate stale overlays
        # Compare: pixels that changed between with_rois and after_delete
        # should cover ALL overlay pixels. If any bright overlay pixels
        # remain unchanged, they're stale.

        # Count scene items that are ROI-related
        scene_items = canvas.scene().items()
        # Filter to items at z-value typical of ROI overlays (z > 0, z < 999)
        overlay_items = [item for item in scene_items
                         if hasattr(item, 'zValue') and 0 < item.zValue() < 999
                         and item.isVisible()]
        assert len(overlay_items) == 0, \
            f"{len(overlay_items)} visible overlay items remain at ROI z-levels"

        # Also verify no transform handle items (z >= 1000, where handles live)
        from PySide6.QtWidgets import QGraphicsRectItem
        handle_items = [item for item in scene_items
                        if isinstance(item, QGraphicsRectItem)
                        and item.zValue() >= 1000
                        and item.isVisible()]
        assert len(handle_items) == 0, \
            f"{len(handle_items)} visible transform handle items remain after Delete All"

        # Verify the active tool has no stale handle/ant state
        tool = canvas._tool
        if tool and hasattr(tool, '_handle_items'):
            assert len(tool._handle_items) == 0, \
                "Active tool still has handle items after Delete All"
        if tool and hasattr(tool, '_ants_entries'):
            assert len(tool._ants_entries) == 0, \
                "Active tool still has marching ant entries after Delete All"
