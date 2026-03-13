"""Tests for montaris.tools.polygon.PolygonTool."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt

from montaris.layers import ROILayer
from montaris.tools.polygon import PolygonTool, CLOSE_DISTANCE
from montaris.core.undo import UndoStack


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _FakeTransform:
    def __init__(self, zoom=1.0):
        self._zoom = zoom

    def m11(self):
        return self._zoom


class FakeCanvas:
    """Minimal canvas stub for polygon tests."""

    def __init__(self, zoom=1.0):
        self._zoom = zoom
        self._preview_vertices = None
        self._preview_cleared = False

    def transform(self):
        return _FakeTransform(self._zoom)

    def draw_polygon_preview(self, vertices, hover_point=None):
        self._preview_vertices = list(vertices)

    def clear_polygon_preview(self):
        self._preview_cleared = True
        self._preview_vertices = None

    def refresh_active_overlay(self, layer):
        pass


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
def polygon(app):
    return PolygonTool(app)


# ---------------------------------------------------------------------------
# Tests: Adding vertices via on_press
# ---------------------------------------------------------------------------

class TestAddVertices:
    def test_first_click_adds_vertex(self, polygon, layer, canvas):
        polygon.on_press(QPointF(50, 60), layer, canvas)
        assert len(polygon._vertices) == 1
        assert polygon._vertices[0] == (50, 60)

    def test_multiple_clicks_grow_vertices(self, polygon, layer, canvas):
        polygon.on_press(QPointF(10, 10), layer, canvas)
        polygon.on_press(QPointF(100, 10), layer, canvas)
        polygon.on_press(QPointF(100, 100), layer, canvas)
        assert len(polygon._vertices) == 3

    def test_preview_drawn_on_each_click(self, polygon, layer, canvas):
        polygon.on_press(QPointF(10, 10), layer, canvas)
        assert canvas._preview_vertices is not None
        assert len(canvas._preview_vertices) == 1
        polygon.on_press(QPointF(50, 50), layer, canvas)
        assert len(canvas._preview_vertices) == 2

    def test_none_layer_ignored(self, polygon, canvas):
        polygon.on_press(QPointF(50, 50), None, canvas)
        assert len(polygon._vertices) == 0

    def test_non_roi_layer_ignored(self, polygon, canvas):
        class PlainLayer:
            pass
        polygon.on_press(QPointF(50, 50), PlainLayer(), canvas)
        assert len(polygon._vertices) == 0


# ---------------------------------------------------------------------------
# Tests: Closing polygon by clicking near first vertex
# ---------------------------------------------------------------------------

class TestClosePolygon:
    def test_click_near_start_closes(self, polygon, app, layer, canvas):
        # Add 3 vertices forming a triangle
        polygon.on_press(QPointF(50, 50), layer, canvas)
        polygon.on_press(QPointF(150, 50), layer, canvas)
        polygon.on_press(QPointF(100, 130), layer, canvas)
        # Click near the first vertex (within CLOSE_DISTANCE at zoom=1)
        polygon.on_press(QPointF(50, 50), layer, canvas)
        # Vertices should be cleared after finish
        assert len(polygon._vertices) == 0
        # Undo command should have been pushed
        assert app.undo_stack.can_undo

    def test_click_far_from_start_does_not_close(self, polygon, layer, canvas):
        polygon.on_press(QPointF(50, 50), layer, canvas)
        polygon.on_press(QPointF(150, 50), layer, canvas)
        polygon.on_press(QPointF(100, 130), layer, canvas)
        # Click far from the first vertex
        polygon.on_press(QPointF(140, 130), layer, canvas)
        # Should have 4 vertices, not closed
        assert len(polygon._vertices) == 4


# ---------------------------------------------------------------------------
# Tests: finish() with 3+ vertices fills mask
# ---------------------------------------------------------------------------

class TestFinishWithVertices:
    def test_finish_fills_mask(self, polygon, app, layer, canvas):
        polygon.on_press(QPointF(50, 50), layer, canvas)
        polygon.on_press(QPointF(150, 50), layer, canvas)
        polygon.on_press(QPointF(100, 130), layer, canvas)
        polygon.finish()
        # Interior of the triangle should be filled
        assert layer.mask[70, 100] == 255
        assert app.undo_stack.can_undo

    def test_finish_clears_preview(self, polygon, layer, canvas):
        polygon.on_press(QPointF(50, 50), layer, canvas)
        polygon.on_press(QPointF(150, 50), layer, canvas)
        polygon.on_press(QPointF(100, 130), layer, canvas)
        polygon.finish()
        assert canvas._preview_cleared

    def test_finish_clears_vertices(self, polygon, layer, canvas):
        polygon.on_press(QPointF(50, 50), layer, canvas)
        polygon.on_press(QPointF(150, 50), layer, canvas)
        polygon.on_press(QPointF(100, 130), layer, canvas)
        polygon.finish()
        assert len(polygon._vertices) == 0


# ---------------------------------------------------------------------------
# Tests: finish() with <3 vertices cancels
# ---------------------------------------------------------------------------

class TestFinishTooFewVertices:
    def test_finish_with_zero_vertices(self, polygon, app, canvas):
        polygon._canvas = canvas
        polygon.finish()
        assert not app.undo_stack.can_undo

    def test_finish_with_one_vertex(self, polygon, app, layer, canvas):
        polygon.on_press(QPointF(50, 50), layer, canvas)
        polygon.finish()
        assert not app.undo_stack.can_undo
        assert len(polygon._vertices) == 0

    def test_finish_with_two_vertices(self, polygon, app, layer, canvas):
        polygon.on_press(QPointF(50, 50), layer, canvas)
        polygon.on_press(QPointF(100, 50), layer, canvas)
        polygon.finish()
        assert not app.undo_stack.can_undo
        assert len(polygon._vertices) == 0


# ---------------------------------------------------------------------------
# Tests: on_key_press(Key_Return) finishes polygon
# ---------------------------------------------------------------------------

class TestKeyReturn:
    def test_enter_finishes(self, polygon, app, layer, canvas):
        polygon.on_press(QPointF(50, 50), layer, canvas)
        polygon.on_press(QPointF(150, 50), layer, canvas)
        polygon.on_press(QPointF(100, 130), layer, canvas)
        result = polygon.on_key_press(Qt.Key_Return, canvas)
        assert result is True
        assert len(polygon._vertices) == 0
        assert app.undo_stack.can_undo

    def test_key_enter_also_works(self, polygon, app, layer, canvas):
        polygon.on_press(QPointF(50, 50), layer, canvas)
        polygon.on_press(QPointF(150, 50), layer, canvas)
        polygon.on_press(QPointF(100, 130), layer, canvas)
        result = polygon.on_key_press(Qt.Key_Enter, canvas)
        assert result is True
        assert app.undo_stack.can_undo


# ---------------------------------------------------------------------------
# Tests: on_key_press(Key_Escape) cancels polygon
# ---------------------------------------------------------------------------

class TestKeyEscape:
    def test_escape_cancels(self, polygon, layer, canvas):
        polygon.on_press(QPointF(50, 50), layer, canvas)
        polygon.on_press(QPointF(150, 50), layer, canvas)
        result = polygon.on_key_press(Qt.Key_Escape, canvas)
        assert result is True
        assert len(polygon._vertices) == 0

    def test_escape_clears_preview(self, polygon, layer, canvas):
        polygon.on_press(QPointF(50, 50), layer, canvas)
        polygon.on_key_press(Qt.Key_Escape, canvas)
        assert canvas._preview_cleared

    def test_escape_does_not_fill(self, polygon, app, layer, canvas):
        polygon.on_press(QPointF(50, 50), layer, canvas)
        polygon.on_press(QPointF(150, 50), layer, canvas)
        polygon.on_press(QPointF(100, 130), layer, canvas)
        polygon.on_key_press(Qt.Key_Escape, canvas)
        # Mask should remain empty
        assert np.all(layer.mask == 0)
        assert not app.undo_stack.can_undo


# ---------------------------------------------------------------------------
# Tests: Triangle fill — verify correct pixels
# ---------------------------------------------------------------------------

class TestTriangleFill:
    def test_triangle_interior_filled(self, polygon, layer, canvas):
        # Large triangle: (20,20), (180,20), (100,140)
        polygon.on_press(QPointF(20, 20), layer, canvas)
        polygon.on_press(QPointF(180, 20), layer, canvas)
        polygon.on_press(QPointF(100, 140), layer, canvas)
        polygon.finish()
        # Centroid is at roughly (100, 60) — should be filled
        assert layer.mask[60, 100] == 255

    def test_triangle_exterior_empty(self, polygon, layer, canvas):
        polygon.on_press(QPointF(20, 20), layer, canvas)
        polygon.on_press(QPointF(180, 20), layer, canvas)
        polygon.on_press(QPointF(100, 140), layer, canvas)
        polygon.finish()
        # Top-left corner is outside the triangle
        assert layer.mask[0, 0] == 0
        # Bottom-right corner is outside
        assert layer.mask[149, 199] == 0

    def test_triangle_has_nonzero_pixels(self, polygon, layer, canvas):
        polygon.on_press(QPointF(20, 20), layer, canvas)
        polygon.on_press(QPointF(180, 20), layer, canvas)
        polygon.on_press(QPointF(100, 140), layer, canvas)
        polygon.finish()
        assert np.sum(layer.mask > 0) > 0


# ---------------------------------------------------------------------------
# Tests: Rectangle (4 vertices) fill
# ---------------------------------------------------------------------------

class TestRectangleFill:
    def test_rectangle_fills_interior(self, polygon, layer, canvas):
        # Rectangle: (30,30) → (170,30) → (170,120) → (30,120)
        polygon.on_press(QPointF(30, 30), layer, canvas)
        polygon.on_press(QPointF(170, 30), layer, canvas)
        polygon.on_press(QPointF(170, 120), layer, canvas)
        polygon.on_press(QPointF(30, 120), layer, canvas)
        polygon.finish()
        # Center should be filled
        assert layer.mask[75, 100] == 255
        # Just inside all edges should be filled
        assert layer.mask[31, 31] == 255
        assert layer.mask[119, 169] == 255

    def test_rectangle_exterior_empty(self, polygon, layer, canvas):
        polygon.on_press(QPointF(30, 30), layer, canvas)
        polygon.on_press(QPointF(170, 30), layer, canvas)
        polygon.on_press(QPointF(170, 120), layer, canvas)
        polygon.on_press(QPointF(30, 120), layer, canvas)
        polygon.finish()
        assert layer.mask[0, 0] == 0
        assert layer.mask[149, 199] == 0
        assert layer.mask[25, 100] == 0

    def test_rectangle_area_approximately_correct(self, polygon, layer, canvas):
        polygon.on_press(QPointF(30, 30), layer, canvas)
        polygon.on_press(QPointF(170, 30), layer, canvas)
        polygon.on_press(QPointF(170, 120), layer, canvas)
        polygon.on_press(QPointF(30, 120), layer, canvas)
        polygon.finish()
        filled = np.sum(layer.mask > 0)
        expected = 140 * 90  # approximate area
        # Allow some rasterization tolerance
        assert abs(filled - expected) < expected * 0.05
