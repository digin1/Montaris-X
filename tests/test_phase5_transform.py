"""Tests for Phase 5: Transform, Move & Canvas Interaction."""
import pytest
import numpy as np

from montaris.core.components import (
    get_component_at, label_connected_components, get_component_bbox,
)
from montaris.tools.select import SelectTool
from montaris.tools import TOOL_REGISTRY


class TestConnectedComponents:
    def test_single_component(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:10, 5:10] = 255
        comp = get_component_at(mask, 7, 7)
        assert comp is not None
        assert np.count_nonzero(comp) == 25

    def test_empty_pixel(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:10, 5:10] = 255
        comp = get_component_at(mask, 0, 0)
        assert comp is None

    def test_two_components(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:5, 2:5] = 255
        mask[12:15, 12:15] = 255
        comp1 = get_component_at(mask, 3, 3)
        comp2 = get_component_at(mask, 13, 13)
        assert np.count_nonzero(comp1) == 9
        assert np.count_nonzero(comp2) == 9
        assert not np.any(comp1 & comp2)

    def test_out_of_bounds(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        assert get_component_at(mask, -1, -1) is None
        assert get_component_at(mask, 10, 10) is None


class TestLabelComponents:
    def test_label_two_components(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:5, 2:5] = 255
        mask[12:15, 12:15] = 255
        labels, n = label_connected_components(mask)
        assert n == 2
        assert labels[3, 3] != labels[13, 13]
        assert labels[3, 3] > 0
        assert labels[13, 13] > 0

    def test_empty_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        labels, n = label_connected_components(mask)
        assert n == 0


class TestComponentBbox:
    def test_bbox(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[2:5, 3:7] = 1
        bbox = get_component_bbox(labels, 1)
        assert bbox == (2, 5, 3, 7)

    def test_missing_label(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        assert get_component_bbox(labels, 99) is None


class TestSelectTool:
    def test_registered(self):
        assert 'Select' in TOOL_REGISTRY

    def test_creation(self, qapp, app_with_image):
        tool = SelectTool(app_with_image)
        assert tool.name == "Select"


class TestZoomInOut:
    def test_zoom_in(self, qapp, app_with_image):
        app = app_with_image
        before = app.canvas.transform().m11()
        app.canvas.zoom_in()
        after = app.canvas.transform().m11()
        assert after > before

    def test_zoom_out(self, qapp, app_with_image):
        app = app_with_image
        before = app.canvas.transform().m11()
        app.canvas.zoom_out()
        after = app.canvas.transform().m11()
        assert after < before


class TestHUD:
    def test_hud_exists(self, qapp, app_with_image):
        assert app_with_image.canvas._hud_label is not None
        assert app_with_image.canvas._hud_label.isVisible()


class TestComponentAwareMove:
    def test_move_detects_component(self, qapp, app_with_image):
        from montaris.tools.move import MoveTool
        from PySide6.QtCore import QPointF
        app = app_with_image
        roi = app.layer_stack.roi_layers[0]
        # Create two separate components
        roi.mask[5:10, 5:10] = 255
        roi.mask[50:55, 50:55] = 255

        tool = MoveTool(app)
        tool.on_press(QPointF(7, 7), roi, app.canvas)
        # Should be in component mode (single component < total)
        assert tool._component_mask is not None
        tool.on_release(QPointF(7, 7), roi, app.canvas)
