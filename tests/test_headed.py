"""Headful PySide6 tests — run with a real display, not offscreen.

Usage:
    QT_QPA_PLATFORM= .venv/bin/pytest tests/test_headed.py -m headed -s
"""
import os
import time
from unittest.mock import patch

import numpy as np
import pytest

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication, QMessageBox

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


class TestNoSelectionWarnings:
    """Verify that ROI operations warn the user when no ROI is selected.

    Reproduces the issue Nic reported: "no warning message appears when
    trying to Duplicate with no ROI selected."
    """

    def test_duplicate_no_selection_warns(self, app_with_real_image):
        """Clicking Duplicate with no ROI selected should show a warning."""
        window = app_with_real_image
        panel = window.layer_panel
        # Deselect everything
        panel.list_widget.clearSelection()
        panel.list_widget.setCurrentRow(-1)
        QApplication.processEvents()

        roi_count_before = len(window.layer_stack.roi_layers)

        with patch.object(QMessageBox, 'information') as mock_info:
            panel._duplicate_selected()
            QApplication.processEvents()

        roi_count_after = len(window.layer_stack.roi_layers)
        assert roi_count_after == roi_count_before, \
            "Duplicate should not create a new ROI when nothing is selected"
        mock_info.assert_called_once(), \
            "Duplicate should show a warning when no ROI is selected"

    def test_merge_no_selection_warns(self, app_with_real_image):
        """Clicking Merge with <2 ROIs selected should show a warning."""
        window = app_with_real_image
        panel = window.layer_panel
        panel.list_widget.clearSelection()
        panel.list_widget.setCurrentRow(-1)
        QApplication.processEvents()

        with patch.object(QMessageBox, 'information') as mock_info:
            panel._merge_selected()
            QApplication.processEvents()

        mock_info.assert_called_once(), \
            "Merge should show a warning when fewer than 2 ROIs are selected"

    def test_rename_no_selection_warns(self, app_with_real_image):
        """Clicking Rename with no ROI selected should show a warning."""
        window = app_with_real_image
        panel = window.layer_panel
        panel.list_widget.clearSelection()
        panel.list_widget.setCurrentRow(-1)
        QApplication.processEvents()

        with patch.object(QMessageBox, 'information') as mock_info:
            panel._rename_selected()
            QApplication.processEvents()

        mock_info.assert_called_once(), \
            "Rename should show a warning when no ROI is selected"
