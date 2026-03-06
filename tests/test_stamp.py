import numpy as np
import pytest
from PySide6.QtCore import QPointF
from montaris.tools.stamp import StampTool
from montaris.layers import ROILayer
from montaris.core.undo import UndoStack


class MockApp:
    def __init__(self):
        self.undo_stack = UndoStack()


class MockCanvas:
    def refresh_overlays(self):
        pass

    def refresh_active_overlay(self, layer):
        pass

    def stamp_on_roi_pixmap(self, layer, cx, cy, half_w, half_h):
        pass


class TestStampTool:
    def test_stamp_places_square(self):
        """A single stamp press should place a filled square."""
        app = MockApp()
        tool = StampTool(app)
        tool.size = 20
        layer = ROILayer("test", 100, 100)
        canvas = MockCanvas()

        assert layer.mask.sum() == 0
        tool.on_press(QPointF(50, 50), layer, canvas)
        tool.on_release(QPointF(50, 50), layer, canvas)

        # The stamp should create a 20x20 filled square centered at (50, 50)
        half = 20 // 2
        cx, cy = 50, 50
        y1, y2 = cy - half, cy - half + 20
        x1, x2 = cx - half, cx - half + 20
        assert (layer.mask[y1:y2, x1:x2] == 255).all()

    def test_stamp_with_undo(self):
        """Undo should revert the stamp."""
        app = MockApp()
        tool = StampTool(app)
        tool.size = 10
        layer = ROILayer("test", 100, 100)
        canvas = MockCanvas()

        tool.on_press(QPointF(50, 50), layer, canvas)
        tool.on_release(QPointF(50, 50), layer, canvas)

        assert layer.mask.sum() > 0
        assert app.undo_stack.can_undo

        app.undo_stack.undo()
        assert layer.mask.sum() == 0

    def test_stamp_continuous_drag(self):
        """Dragging the stamp should fill along the path."""
        app = MockApp()
        tool = StampTool(app)
        tool.size = 10
        layer = ROILayer("test", 200, 200)
        canvas = MockCanvas()

        tool.on_press(QPointF(20, 50), layer, canvas)
        tool.on_move(QPointF(100, 50), layer, canvas)
        tool.on_release(QPointF(100, 50), layer, canvas)

        # Both start and end positions should have stamps
        assert layer.mask[50, 20] == 255
        assert layer.mask[50, 100] == 255
        # Points along the line should be filled too
        assert layer.mask[50, 60] == 255

    def test_stamp_at_edge(self):
        """Stamp near the edge should clip without error."""
        app = MockApp()
        tool = StampTool(app)
        tool.size = 20
        layer = ROILayer("test", 50, 50)
        canvas = MockCanvas()

        tool.on_press(QPointF(2, 2), layer, canvas)
        tool.on_release(QPointF(2, 2), layer, canvas)

        # Some pixels should be filled
        assert layer.mask.sum() > 0
        # Should not raise any errors

    def test_stamp_no_layer(self):
        """Stamp with no layer should be a no-op."""
        app = MockApp()
        tool = StampTool(app)
        canvas = MockCanvas()

        # Should not raise
        tool.on_press(QPointF(10, 10), None, canvas)
        tool.on_move(QPointF(20, 20), None, canvas)
        tool.on_release(QPointF(20, 20), None, canvas)
