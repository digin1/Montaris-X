"""Tests for montaris.tools.select.SelectTool."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import numpy as np
import pytest

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF

from montaris.layers import ROILayer
from montaris.tools.select import SelectTool
from montaris.core.selection import SelectionModel


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class FakeLayerStack:
    def __init__(self, roi_layers):
        self.roi_layers = roi_layers


class FakeCanvas:
    def __init__(self, roi_layers):
        self.layer_stack = FakeLayerStack(roi_layers)
        self._selection = SelectionModel()


class FakeApp:
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def app():
    return FakeApp()


@pytest.fixture
def roi_a():
    """ROI with a painted block at (40:80, 40:80)."""
    roi = ROILayer("ROI_A", 200, 150)
    roi.mask[40:80, 40:80] = 255
    roi.visible = True
    return roi


@pytest.fixture
def roi_b():
    """ROI with a painted block at (90:130, 90:130)."""
    roi = ROILayer("ROI_B", 200, 150)
    roi.mask[90:130, 90:130] = 255
    roi.visible = True
    return roi


# ---------------------------------------------------------------------------
# Tests: on_press at position with ROI selects it
# ---------------------------------------------------------------------------

class TestSelectROI:
    def test_click_on_roi_selects_it(self, qapp, app, roi_a):
        canvas = FakeCanvas([roi_a])
        tool = SelectTool(app)
        tool.on_press(QPointF(60, 60), None, canvas)
        assert canvas._selection.layers == [roi_a]

    def test_click_on_second_roi_selects_it(self, qapp, app, roi_a, roi_b):
        canvas = FakeCanvas([roi_a, roi_b])
        tool = SelectTool(app)
        tool.on_press(QPointF(110, 110), None, canvas)
        assert canvas._selection.layers == [roi_b]

    def test_click_switches_selection(self, qapp, app, roi_a, roi_b):
        canvas = FakeCanvas([roi_a, roi_b])
        tool = SelectTool(app)
        # Select ROI A
        tool.on_press(QPointF(60, 60), None, canvas)
        assert canvas._selection.layers == [roi_a]
        # Switch to ROI B
        tool.on_press(QPointF(110, 110), None, canvas)
        assert canvas._selection.layers == [roi_b]


# ---------------------------------------------------------------------------
# Tests: on_press at empty position deselects
# ---------------------------------------------------------------------------

class TestDeselect:
    def test_click_on_empty_clears_selection(self, qapp, app, roi_a):
        canvas = FakeCanvas([roi_a])
        tool = SelectTool(app)
        # First select
        tool.on_press(QPointF(60, 60), None, canvas)
        assert canvas._selection.count == 1
        # Then click on empty area
        tool.on_press(QPointF(5, 5), None, canvas)
        assert canvas._selection.count == 0

    def test_click_on_empty_with_no_rois(self, qapp, app):
        canvas = FakeCanvas([])
        tool = SelectTool(app)
        tool.on_press(QPointF(50, 50), None, canvas)
        assert canvas._selection.count == 0


# ---------------------------------------------------------------------------
# Tests: Delegation to canvas selection model
# ---------------------------------------------------------------------------

class TestSelectionModelDelegation:
    def test_selection_model_set_called(self, qapp, app, roi_a):
        canvas = FakeCanvas([roi_a])
        tool = SelectTool(app)
        signals_received = []
        canvas._selection.changed.connect(lambda layers: signals_received.append(layers))
        tool.on_press(QPointF(60, 60), None, canvas)
        assert len(signals_received) == 1
        assert signals_received[0] == [roi_a]

    def test_selection_model_clear_called(self, qapp, app, roi_a):
        canvas = FakeCanvas([roi_a])
        tool = SelectTool(app)
        # Select first
        tool.on_press(QPointF(60, 60), None, canvas)
        signals_received = []
        canvas._selection.changed.connect(lambda layers: signals_received.append(layers))
        # Deselect
        tool.on_press(QPointF(5, 5), None, canvas)
        assert len(signals_received) == 1
        assert signals_received[0] == []

    def test_hidden_roi_not_selectable(self, qapp, app, roi_a):
        """A hidden ROI should not be hit-testable."""
        roi_a.visible = False
        canvas = FakeCanvas([roi_a])
        tool = SelectTool(app)
        tool.on_press(QPointF(60, 60), None, canvas)
        assert canvas._selection.count == 0
