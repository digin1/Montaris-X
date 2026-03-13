"""Tests for montaris.tools.eraser.EraserTool."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPointF

from montaris.layers import ROILayer
from montaris.tools.eraser import EraserTool
from montaris.core.undo import UndoStack


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _FakeTransform:
    def __init__(self, zoom=1.0):
        self._zoom = zoom

    def m11(self):
        return self._zoom


class _FakeItem:
    def setVisible(self, v):
        pass


class FakeCanvas:
    """Minimal canvas stub for eraser tests without GUI rendering."""

    def __init__(self, zoom=1.0):
        self._zoom = zoom
        self._selection_highlight_items = []

    def refresh_active_overlay(self, layer):
        pass

    def refresh_active_overlay_partial(self, layer, bbox):
        pass

    def _update_selection_highlights(self):
        pass

    def transform(self):
        return _FakeTransform(self._zoom)


class FakeApp:
    def __init__(self):
        self.undo_stack = UndoStack()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    return FakeApp()


@pytest.fixture
def canvas():
    return FakeCanvas(zoom=1.0)


@pytest.fixture
def layer():
    return ROILayer("test", 200, 150)


@pytest.fixture
def painted_layer():
    """Layer with a 40x40 filled block in the center."""
    roi = ROILayer("painted", 200, 150)
    roi.mask[55:95, 80:120] = 255
    return roi


@pytest.fixture
def eraser(app):
    tool = EraserTool(app)
    tool.size = 20
    return tool


# ---------------------------------------------------------------------------
# Tests: _effective_size
# ---------------------------------------------------------------------------

class TestEffectiveSize:
    def test_returns_size_divided_by_zoom(self, eraser):
        eraser.size = 100
        eraser._canvas = FakeCanvas(zoom=2.0)
        assert eraser._effective_size() == 50

    def test_zoom_one_returns_size(self, eraser):
        eraser.size = 60
        eraser._canvas = FakeCanvas(zoom=1.0)
        assert eraser._effective_size() == 60

    def test_no_canvas_returns_raw_size(self, eraser):
        eraser.size = 40
        eraser._canvas = None
        assert eraser._effective_size() == 40

    def test_minimum_is_one(self, eraser):
        eraser.size = 1
        eraser._canvas = FakeCanvas(zoom=100.0)
        assert eraser._effective_size() == 1


# ---------------------------------------------------------------------------
# Tests: _get_circle
# ---------------------------------------------------------------------------

class TestGetCircle:
    def test_returns_bool_mask(self, eraser):
        circle = eraser._get_circle(es=10)
        assert circle.dtype == bool

    def test_shape_matches_size(self, eraser):
        es = 10
        r = es // 2
        circle = eraser._get_circle(es=es)
        expected_side = 2 * r + 1
        assert circle.shape == (expected_side, expected_side)

    def test_caches_result(self, eraser):
        c1 = eraser._get_circle(es=10)
        c2 = eraser._get_circle(es=10)
        assert c1 is c2

    def test_recreates_on_size_change(self, eraser):
        c1 = eraser._get_circle(es=10)
        c2 = eraser._get_circle(es=20)
        assert c1 is not c2
        assert c1.shape != c2.shape

    def test_center_pixel_is_true(self, eraser):
        circle = eraser._get_circle(es=10)
        r = 10 // 2
        assert circle[r, r] is np.bool_(True)


# ---------------------------------------------------------------------------
# Tests: _erase
# ---------------------------------------------------------------------------

class TestErase:
    def test_erases_circular_region(self, eraser, painted_layer, canvas):
        eraser._canvas = canvas
        # Fill the entire mask so we can see the erased hole
        painted_layer.mask[:] = 255
        eraser._erase(QPointF(100, 75), painted_layer)
        # Center should be erased
        assert painted_layer.mask[75, 100] == 0

    def test_stroke_bbox_set(self, eraser, painted_layer, canvas):
        eraser._canvas = canvas
        painted_layer.mask[:] = 255
        eraser._erase(QPointF(50, 50), painted_layer)
        assert eraser._stroke_bbox is not None

    def test_erase_at_edge_no_error(self, eraser, painted_layer, canvas):
        eraser._canvas = canvas
        painted_layer.mask[:] = 255
        # Erase at corner — should not raise
        eraser._erase(QPointF(0, 0), painted_layer)
        eraser._erase(QPointF(199, 149), painted_layer)

    def test_snapshot_captured(self, eraser, painted_layer, canvas):
        eraser._canvas = canvas
        painted_layer.mask[:] = 255
        eraser._erase(QPointF(100, 75), painted_layer)
        assert eraser._snapshot_crop is not None
        assert eraser._snapshot_bbox is not None


# ---------------------------------------------------------------------------
# Tests: _erase_line
# ---------------------------------------------------------------------------

class TestEraseLine:
    def test_erases_along_path(self, eraser, canvas):
        layer = ROILayer("line_test", 200, 150)
        layer.mask[:] = 255
        eraser._canvas = canvas
        p1 = QPointF(20, 75)
        p2 = QPointF(180, 75)
        eraser._erase_line(p1, p2, layer)
        # Midpoint along the line should be erased
        assert layer.mask[75, 100] == 0
        # Both endpoints should be erased
        assert layer.mask[75, 20] == 0
        assert layer.mask[75, 180] == 0

    def test_single_point_line(self, eraser, canvas):
        layer = ROILayer("single", 200, 150)
        layer.mask[:] = 255
        eraser._canvas = canvas
        p = QPointF(50, 50)
        eraser._erase_line(p, p, layer)
        assert layer.mask[50, 50] == 0


# ---------------------------------------------------------------------------
# Tests: full press/move/release cycle
# ---------------------------------------------------------------------------

class TestFullStrokeCycle:
    def test_stroke_creates_undo_command(self, eraser, app, canvas):
        layer = ROILayer("undo_test", 200, 150)
        layer.mask[50:100, 50:100] = 255
        eraser.on_press(QPointF(75, 75), layer, canvas)
        eraser.on_move(QPointF(80, 80), layer, canvas)
        eraser.on_release(QPointF(80, 80), layer, canvas)
        assert app.undo_stack.can_undo

    def test_undo_restores_mask(self, eraser, app, canvas):
        layer = ROILayer("undo_restore", 200, 150)
        layer.mask[50:100, 50:100] = 255
        original = layer.mask.copy()
        eraser.on_press(QPointF(75, 75), layer, canvas)
        eraser.on_move(QPointF(80, 80), layer, canvas)
        eraser.on_release(QPointF(80, 80), layer, canvas)
        # Mask should have changed
        assert not np.array_equal(layer.mask, original)
        # Undo should restore
        app.undo_stack.undo()
        assert np.array_equal(layer.mask, original)

    def test_press_on_non_roi_layer_ignored(self, eraser, app, canvas):
        """Non-ROI layers (e.g. None or image layers) should be silently skipped."""
        eraser.on_press(QPointF(50, 50), None, canvas)
        assert not eraser._erasing

    def test_press_on_layer_without_is_roi_ignored(self, eraser, app, canvas):
        class PlainLayer:
            pass
        eraser.on_press(QPointF(50, 50), PlainLayer(), canvas)
        assert not eraser._erasing


# ---------------------------------------------------------------------------
# Tests: erase on empty mask → no undo command
# ---------------------------------------------------------------------------

class TestEraseOnEmptyMask:
    def test_no_undo_if_no_change(self, eraser, app, canvas):
        """Erasing on an empty mask produces no change and no undo command."""
        layer = ROILayer("empty", 200, 150)
        # mask is already all zeros
        eraser.on_press(QPointF(100, 75), layer, canvas)
        eraser.on_move(QPointF(110, 75), layer, canvas)
        eraser.on_release(QPointF(110, 75), layer, canvas)
        assert not app.undo_stack.can_undo


# ---------------------------------------------------------------------------
# Tests: erase on mask with content → content removed
# ---------------------------------------------------------------------------

class TestEraseOnPaintedMask:
    def test_content_removed_at_stroke(self, eraser, app, canvas):
        layer = ROILayer("painted", 200, 150)
        layer.mask[60:90, 90:110] = 255
        eraser.on_press(QPointF(100, 75), layer, canvas)
        eraser.on_release(QPointF(100, 75), layer, canvas)
        # The center of the painted region should now be erased
        assert layer.mask[75, 100] == 0

    def test_surrounding_pixels_untouched(self, eraser, app, canvas):
        layer = ROILayer("surround", 200, 150)
        layer.mask[:] = 255
        eraser.size = 10  # small eraser
        eraser.on_press(QPointF(100, 75), layer, canvas)
        eraser.on_release(QPointF(100, 75), layer, canvas)
        # Distant pixel should still be 255
        assert layer.mask[0, 0] == 255
        assert layer.mask[149, 199] == 255
