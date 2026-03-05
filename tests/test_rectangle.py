import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QGraphicsScene
from PySide6.QtCore import QPointF
from montaris.tools.rectangle import RectangleTool
from montaris.layers import ROILayer
from montaris.core.undo import UndoStack

# Ensure QApplication exists for QGraphicsScene
_qapp = QApplication.instance() or QApplication(sys.argv)


class MockApp:
    def __init__(self):
        self.undo_stack = UndoStack()


class MockCanvas:
    def __init__(self):
        self._scene = QGraphicsScene()

    def scene(self):
        return self._scene

    def refresh_overlays(self):
        pass

    def refresh_active_overlay(self, layer):
        pass


class TestRectangleTool:
    def test_draw_rectangle_fills_correct_area(self):
        """Drawing a rectangle should fill the correct mask region."""
        app = MockApp()
        tool = RectangleTool(app)
        layer = ROILayer("test", 100, 100)
        canvas = MockCanvas()

        assert layer.mask.sum() == 0
        tool.on_press(QPointF(10, 10), layer, canvas)
        tool.on_release(QPointF(50, 50), layer, canvas)

        # The rectangle region should be filled
        assert layer.mask[10:51, 10:51].sum() > 0
        # Outside should remain 0
        assert layer.mask[0:9, 0:9].sum() == 0
        assert layer.mask[52:, 52:].sum() == 0

    def test_rectangle_fills_all_pixels_in_region(self):
        """Every pixel within the rectangle should be 255."""
        app = MockApp()
        tool = RectangleTool(app)
        layer = ROILayer("test", 100, 100)
        canvas = MockCanvas()

        tool.on_press(QPointF(20, 20), layer, canvas)
        tool.on_release(QPointF(40, 40), layer, canvas)

        assert (layer.mask[20:41, 20:41] == 255).all()

    def test_rectangle_with_undo(self):
        """Undo should revert the rectangle fill."""
        app = MockApp()
        tool = RectangleTool(app)
        layer = ROILayer("test", 100, 100)
        canvas = MockCanvas()

        tool.on_press(QPointF(10, 10), layer, canvas)
        tool.on_release(QPointF(30, 30), layer, canvas)

        assert layer.mask.sum() > 0
        assert app.undo_stack.can_undo

        app.undo_stack.undo()
        assert layer.mask.sum() == 0

    def test_rectangle_reversed_corners(self):
        """Drawing from bottom-right to top-left should work correctly."""
        app = MockApp()
        tool = RectangleTool(app)
        layer = ROILayer("test", 100, 100)
        canvas = MockCanvas()

        tool.on_press(QPointF(50, 50), layer, canvas)
        tool.on_release(QPointF(10, 10), layer, canvas)

        assert (layer.mask[10:51, 10:51] == 255).all()

    def test_rectangle_preview_cleanup(self):
        """Preview item should be removed on release."""
        app = MockApp()
        tool = RectangleTool(app)
        layer = ROILayer("test", 100, 100)
        canvas = MockCanvas()

        tool.on_press(QPointF(10, 10), layer, canvas)
        tool.on_move(QPointF(30, 30), layer, canvas)
        # Preview should exist
        assert tool._preview_item is not None

        tool.on_release(QPointF(30, 30), layer, canvas)
        # Preview should be cleaned up
        assert tool._preview_item is None
