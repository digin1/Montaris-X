import os
import sys
import math

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QGraphicsScene
from PySide6.QtCore import QPointF
from montaris.tools.circle import CircleTool
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


class TestCircleTool:
    def test_draw_circle_fills_roughly_circular_area(self):
        """Drawing a circle should fill a roughly circular region."""
        app = MockApp()
        tool = CircleTool(app)
        layer = ROILayer("test", 200, 200)
        canvas = MockCanvas()

        center = QPointF(100, 100)
        edge = QPointF(120, 100)  # radius = 20
        radius = 20

        tool.on_press(center, layer, canvas)
        tool.on_release(edge, layer, canvas)

        # Center should be filled
        assert layer.mask[100, 100] == 255
        # A point well inside the circle should be filled
        assert layer.mask[100, 110] == 255
        # A point well outside the circle should be 0
        assert layer.mask[100, 150] == 0
        assert layer.mask[0, 0] == 0

        # Check that the filled area is roughly circular:
        # Count filled pixels and compare to expected area (pi * r^2)
        filled = np.count_nonzero(layer.mask)
        expected_area = math.pi * radius * radius
        # Allow 15% tolerance for discrete rasterization
        assert abs(filled - expected_area) / expected_area < 0.15

    def test_circle_with_undo(self):
        """Undo should revert the circle fill."""
        app = MockApp()
        tool = CircleTool(app)
        layer = ROILayer("test", 200, 200)
        canvas = MockCanvas()

        tool.on_press(QPointF(100, 100), layer, canvas)
        tool.on_release(QPointF(130, 100), layer, canvas)

        assert layer.mask.sum() > 0
        assert app.undo_stack.can_undo

        app.undo_stack.undo()
        assert layer.mask.sum() == 0

    def test_circle_zero_radius_no_fill(self):
        """A zero-radius circle (click without drag) should not fill."""
        app = MockApp()
        tool = CircleTool(app)
        layer = ROILayer("test", 100, 100)
        canvas = MockCanvas()

        tool.on_press(QPointF(50, 50), layer, canvas)
        tool.on_release(QPointF(50, 50), layer, canvas)

        assert layer.mask.sum() == 0

    def test_circle_preview_cleanup(self):
        """Preview item should be removed on release."""
        app = MockApp()
        tool = CircleTool(app)
        layer = ROILayer("test", 100, 100)
        canvas = MockCanvas()

        tool.on_press(QPointF(50, 50), layer, canvas)
        tool.on_move(QPointF(70, 50), layer, canvas)
        assert tool._preview_item is not None

        tool.on_release(QPointF(70, 50), layer, canvas)
        assert tool._preview_item is None

    def test_circle_at_edge(self):
        """Circle near the edge should clip to mask bounds without error."""
        app = MockApp()
        tool = CircleTool(app)
        layer = ROILayer("test", 100, 100)
        canvas = MockCanvas()

        tool.on_press(QPointF(5, 5), layer, canvas)
        tool.on_release(QPointF(25, 5), layer, canvas)

        # Should have some filled pixels (partial circle visible)
        assert layer.mask.sum() > 0
        # Center should be filled
        assert layer.mask[5, 5] == 255
