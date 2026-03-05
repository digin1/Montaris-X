import numpy as np
import pytest
from PySide6.QtCore import QPointF
from montaris.tools.bucket_fill import BucketFillTool
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


class TestBucketFill:
    def test_fill_empty_mask_paints_connected_region(self):
        """Filling on an empty (0-valued) pixel should paint 255."""
        app = MockApp()
        tool = BucketFillTool(app)
        layer = ROILayer("test", 50, 50)
        canvas = MockCanvas()

        assert layer.mask.sum() == 0
        tool.on_press(QPointF(25, 25), layer, canvas)
        # The entire mask should be filled since it's all connected 0s
        assert (layer.mask == 255).all()

    def test_fill_painted_area_erases(self):
        """Filling on a painted (>0) pixel should erase that connected component."""
        app = MockApp()
        tool = BucketFillTool(app)
        layer = ROILayer("test", 50, 50)
        canvas = MockCanvas()

        # Paint a rectangle
        layer.mask[10:30, 10:30] = 255
        painted_before = layer.mask.sum()
        assert painted_before > 0

        # Click inside the painted rectangle
        tool.on_press(QPointF(20, 20), layer, canvas)
        # The connected painted area should be erased
        assert layer.mask[10:30, 10:30].sum() == 0

    def test_fill_respects_boundaries(self):
        """Flood fill should not cross a boundary of different value."""
        app = MockApp()
        tool = BucketFillTool(app)
        layer = ROILayer("test", 50, 50)
        canvas = MockCanvas()

        # Create a wall of 255 pixels that isolates the top-left corner
        layer.mask[0:50, 20] = 255  # vertical wall
        layer.mask[20, 0:50] = 255  # horizontal wall

        # Fill the top-left quadrant (0-valued)
        tool.on_press(QPointF(10, 10), layer, canvas)

        # Top-left quadrant should be filled
        assert layer.mask[5, 5] == 255
        # Bottom-right quadrant should remain 0 (behind both walls)
        assert layer.mask[30, 30] == 0

    def test_fill_pushes_undo(self):
        """Fill should push an undo command."""
        app = MockApp()
        tool = BucketFillTool(app)
        layer = ROILayer("test", 30, 30)
        canvas = MockCanvas()

        tool.on_press(QPointF(15, 15), layer, canvas)
        assert app.undo_stack.can_undo
        assert (layer.mask == 255).all()

        app.undo_stack.undo()
        assert (layer.mask == 0).all()

    def test_fill_out_of_bounds(self):
        """Clicking outside mask bounds should not crash."""
        app = MockApp()
        tool = BucketFillTool(app)
        layer = ROILayer("test", 50, 50)
        canvas = MockCanvas()

        # Should not raise
        tool.on_press(QPointF(-5, -5), layer, canvas)
        tool.on_press(QPointF(100, 100), layer, canvas)

    def test_fill_with_tolerance(self):
        """Tolerance should allow filling values within the specified range."""
        app = MockApp()
        tool = BucketFillTool(app)
        tool.tolerance = 10
        layer = ROILayer("test", 20, 20)
        canvas = MockCanvas()

        # Set varying values: top half has value 5 (>0, so fill erases to 0),
        # bottom half has value 100 (big jump, should not be crossed)
        layer.mask[:, :] = 5
        layer.mask[10:, :] = 100  # big jump - should not be crossed

        tool.on_press(QPointF(5, 5), layer, canvas)
        # Top half should be erased to 0 (target was 5, fill_value is 0)
        assert layer.mask[5, 5] == 0
        # Bottom half should remain untouched (100 is far from target 5)
        assert layer.mask[15, 5] == 100
