"""Headful PySide6 tests — run with a real display, not offscreen.

Usage:
    QT_QPA_PLATFORM= .venv/bin/pytest tests/test_headed.py -m headed -s
"""
import os
import time

import numpy as np
import pytest

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

pytestmark = pytest.mark.headed


class TestLoadImageAndRois:
    def test_open_image_loads_data(self, app_with_real_image):
        window = app_with_real_image
        assert window.layer_stack.image_layer is not None
        shape = window.layer_stack.image_layer.shape
        assert shape[0] > 0 and shape[1] > 0

    def test_rois_imported(self, app_with_real_image):
        window = app_with_real_image
        rois = window.layer_stack.roi_layers
        assert len(rois) > 0, "No ROIs were imported from test.zip"

    def test_roi_names_present(self, app_with_real_image):
        window = app_with_real_image
        names = [r.name for r in window.layer_stack.roi_layers]
        # test.zip contains ImageJ ROI files like ACB.roi, AON1.roi, etc.
        assert any("ACB" in n for n in names) or len(names) > 5

    def test_canvas_renders(self, app_with_real_image):
        window = app_with_real_image
        QApplication.processEvents()
        # Canvas should have a non-zero size
        assert window.canvas.width() > 0
        assert window.canvas.height() > 0

    def test_select_roi_layer(self, app_with_real_image):
        window = app_with_real_image
        rois = window.layer_stack.roi_layers
        if rois:
            window.canvas.set_active_layer(rois[0])
            QApplication.processEvents()
            assert window.canvas._active_layer is rois[0]

    def test_roi_masks_have_pixels(self, app_with_real_image):
        window = app_with_real_image
        rois = window.layer_stack.roi_layers
        nonempty = [r for r in rois if np.any(r.mask)]
        assert len(nonempty) > 0, "All ROI masks are empty"
