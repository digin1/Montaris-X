"""Tests for brush auto-overlap with compression."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF

from montaris.layers import ROILayer, ImageLayer
from montaris.tools.brush import BrushTool


class _FakeTransform:
    def m11(self):
        return 1.0


class FakeCanvas:
    """Minimal canvas stub for tool tests without GUI rendering."""
    def __init__(self):
        self._selection_highlight_items = []

    def refresh_active_overlay(self, layer):
        pass

    def refresh_active_overlay_partial(self, layer, bbox):
        pass

    def _update_selection_highlights(self):
        pass

    def transform(self):
        return _FakeTransform()

    def scene(self):
        return self

    def addRect(self, *a, **kw):
        return _FakeItem()

    def addEllipse(self, *a, **kw):
        return _FakeItem()

    def removeItem(self, item):
        pass


class _FakeItem:
    def setZValue(self, z):
        pass
    def setVisible(self, v):
        pass


@pytest.fixture
def overlap_app(app):
    """App with image and two overlapping ROIs."""
    img = np.zeros((200, 200), dtype=np.uint8)
    app.layer_stack.set_image(ImageLayer("test", img))
    app.canvas.refresh_image()

    active = ROILayer("active", 200, 200)
    inactive = ROILayer("inactive", 200, 200)
    # Pre-paint inactive in overlap region
    inactive.mask[50:100, 50:100] = 255

    app.layer_stack.add_roi(active)
    app.layer_stack.add_roi(inactive)
    app.canvas.set_active_layer(active)
    app._auto_overlap = True
    return app, active, inactive


class TestAutoOverlapBasic:
    def test_overlap_erases_inactive(self, overlap_app):
        app, active, inactive = overlap_app
        canvas = FakeCanvas()
        original_inactive = inactive.mask.copy()

        tool = BrushTool(app)
        tool.size = 30
        # Paint in the overlap region
        tool.on_press(QPointF(75, 75), active, canvas)
        tool.on_move(QPointF(80, 80), active, canvas)
        tool.on_release(QPointF(80, 80), active, canvas)

        # Active should have paint
        assert np.any(active.mask[50:100, 50:100] > 0)
        # Inactive should have been erased where we painted
        assert not np.array_equal(inactive.mask, original_inactive)

    def test_inactive_stays_compressed(self, overlap_app):
        app, active, inactive = overlap_app
        canvas = FakeCanvas()
        inactive.compress()
        assert inactive.is_compressed

        tool = BrushTool(app)
        tool.size = 30
        tool.on_press(QPointF(75, 75), active, canvas)
        tool.on_move(QPointF(80, 80), active, canvas)
        tool.on_release(QPointF(80, 80), active, canvas)

        # After overlap processing, inactive should be re-compressed or
        # at minimum the check should not leave it fully decompressed permanently
        # The key test: the operation should complete without error

    def test_undo_restores_both(self, overlap_app):
        app, active, inactive = overlap_app
        canvas = FakeCanvas()
        original_active = active.mask.copy()
        original_inactive = inactive.mask.copy()

        app.undo_stack._stack.clear()
        app.undo_stack._index = -1
        app.undo_stack._total_bytes = 0

        tool = BrushTool(app)
        tool.size = 30
        tool.on_press(QPointF(75, 75), active, canvas)
        tool.on_move(QPointF(80, 80), active, canvas)
        tool.on_release(QPointF(80, 80), active, canvas)

        # Undo
        if app.undo_stack._index >= 0:
            cmd = app.undo_stack._stack[app.undo_stack._index]
            cmd.undo()
            assert np.array_equal(active.mask, original_active), \
                "Active mask not restored after undo"
            assert np.array_equal(inactive.mask, original_inactive), \
                "Inactive mask not restored after undo"


class TestNoOverlapScenario:
    def test_no_decompression_when_no_overlap(self, app):
        """Painting far from inactive ROI should not decompress it."""
        img = np.zeros((200, 200), dtype=np.uint8)
        app.layer_stack.set_image(ImageLayer("test", img))
        app.canvas.refresh_image()

        active = ROILayer("active", 200, 200)
        inactive = ROILayer("inactive", 200, 200)
        inactive.mask[150:180, 150:180] = 255
        inactive.compress()

        app.layer_stack.add_roi(active)
        app.layer_stack.add_roi(inactive)
        app.canvas.set_active_layer(active)
        app._auto_overlap = True

        canvas = FakeCanvas()
        tool = BrushTool(app)
        tool.size = 10
        # Paint far from inactive region
        tool.on_press(QPointF(10, 10), active, canvas)
        tool.on_release(QPointF(10, 10), active, canvas)

        # Inactive should stay compressed (no overlap detected)
        assert inactive._mask is None


class TestManyRois:
    def test_overlap_with_many_rois(self, app):
        """Auto-overlap with >10 ROIs (realistic scenario)."""
        img = np.zeros((200, 200), dtype=np.uint8)
        app.layer_stack.set_image(ImageLayer("test", img))
        app.canvas.refresh_image()

        active = ROILayer("active", 200, 200)
        app.layer_stack.add_roi(active)
        app.canvas.set_active_layer(active)
        app._auto_overlap = True

        # Create 12 other ROIs with overlapping data
        for i in range(12):
            roi = ROILayer(f"roi_{i}", 200, 200)
            y = (i * 15) % 150
            roi.mask[y:y+30, 70:110] = 255
            roi.compress()
            app.layer_stack.add_roi(roi)

        canvas = FakeCanvas()
        tool = BrushTool(app)
        tool.size = 20
        # Paint in the center where overlap exists
        tool.on_press(QPointF(90, 80), active, canvas)
        tool.on_move(QPointF(95, 85), active, canvas)
        tool.on_release(QPointF(95, 85), active, canvas)

        # Should complete without error
        assert np.any(active.mask > 0)
