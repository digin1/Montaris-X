"""Tests for MontarisApp helper methods: upscale, export bbox, drag-drop,
select-all, and flatten offsets."""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QUrl, QMimeData, QPointF

from montaris.app import MontarisApp, apply_dark_theme
from montaris.layers import ImageLayer, ROILayer


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def app_fixture():
    qapp = QApplication.instance() or QApplication(sys.argv)
    apply_dark_theme(qapp)
    win = MontarisApp()
    img = np.zeros((100, 100), dtype=np.uint8)
    win.layer_stack.set_image(ImageLayer("test", img))
    win.canvas.refresh_image()
    QApplication.processEvents()
    yield win
    win.close()


# ── _upscale_mask_if_needed ──────────────────────────────────────────


class TestUpscaleMaskIfNeeded:
    def test_factor_1_returns_same_mask(self, app_fixture):
        """With downsample_factor=1 the mask is returned unchanged."""
        win = app_fixture
        win._downsample_factor = 1
        mask = np.ones((50, 50), dtype=np.uint8) * 255
        result = win._upscale_mask_if_needed(mask)
        assert result is mask

    def test_factor_2_doubles_size(self, app_fixture):
        """With downsample_factor=2 the mask dimensions are doubled."""
        win = app_fixture
        win._downsample_factor = 2
        # Ensure no document clips the result
        saved = win._active_doc_index
        win._active_doc_index = -1
        try:
            mask = np.ones((25, 30), dtype=np.uint8) * 255
            result = win._upscale_mask_if_needed(mask)
            assert result.shape == (50, 60)
            assert result.dtype == mask.dtype
        finally:
            win._downsample_factor = 1
            win._active_doc_index = saved

    def test_factor_4_quadruples_size(self, app_fixture):
        """With downsample_factor=4 the mask dimensions are 4x larger."""
        win = app_fixture
        win._downsample_factor = 4
        saved = win._active_doc_index
        win._active_doc_index = -1
        try:
            mask = np.ones((10, 15), dtype=np.uint8) * 128
            result = win._upscale_mask_if_needed(mask)
            assert result.shape == (40, 60)
        finally:
            win._downsample_factor = 1
            win._active_doc_index = saved


# ── _get_export_bbox ─────────────────────────────────────────────────


class TestGetExportBbox:
    def _make_roi_with_bbox(self, bbox):
        """Create a small ROI whose get_bbox() returns *bbox*."""
        roi = ROILayer("tmp", 100, 100)
        if bbox is not None:
            y1, y2, x1, x2 = bbox
            roi.mask[y1:y2, x1:x2] = 255
        return roi

    def test_no_upscale_returns_raw_bbox(self, app_fixture):
        win = app_fixture
        win._downsample_factor = 1
        roi = self._make_roi_with_bbox((10, 20, 30, 50))
        bbox = win._get_export_bbox(roi, upscale=False)
        assert bbox == (10, 20, 30, 50)

    def test_upscale_with_factor_2(self, app_fixture):
        win = app_fixture
        win._downsample_factor = 2
        saved = win._active_doc_index
        win._active_doc_index = -1
        try:
            roi = self._make_roi_with_bbox((5, 10, 15, 25))
            bbox = win._get_export_bbox(roi, upscale=True)
            assert bbox == (10, 20, 30, 50)
        finally:
            win._downsample_factor = 1
            win._active_doc_index = saved

    def test_upscale_false_ignores_factor(self, app_fixture):
        """Even with a high factor, upscale=False should not scale."""
        win = app_fixture
        win._downsample_factor = 4
        saved = win._active_doc_index
        win._active_doc_index = -1
        try:
            roi = self._make_roi_with_bbox((2, 5, 3, 8))
            bbox = win._get_export_bbox(roi, upscale=False)
            assert bbox == (2, 5, 3, 8)
        finally:
            win._downsample_factor = 1
            win._active_doc_index = saved

    def test_none_bbox_returns_none(self, app_fixture):
        """An empty ROI should yield None regardless of upscale."""
        win = app_fixture
        roi = self._make_roi_with_bbox(None)
        assert win._get_export_bbox(roi, upscale=True) is None
        assert win._get_export_bbox(roi, upscale=False) is None


# ── Drag-and-drop ────────────────────────────────────────────────────


def _make_mock_drop_event(file_paths):
    """Build a mock drop event carrying local file paths (avoids native crash)."""
    mime = QMimeData()
    urls = [QUrl.fromLocalFile(p) for p in file_paths]
    mime.setUrls(urls)
    event = MagicMock()
    event.mimeData.return_value = mime
    return event


def _make_mock_drag_enter_event(file_paths):
    """Build a mock drag-enter event carrying URLs."""
    mime = QMimeData()
    urls = [QUrl.fromLocalFile(p) for p in file_paths]
    mime.setUrls(urls)
    event = MagicMock()
    event.mimeData.return_value = mime
    return event


class TestDragAndDrop:
    def test_drop_image_calls_open_image(self, app_fixture):
        win = app_fixture
        event = _make_mock_drop_event(["/tmp/photo.tif"])
        with patch.object(win, "open_image") as mock_open:
            win.dropEvent(event)
            mock_open.assert_called_once_with(["/tmp/photo.tif"])

    def test_drop_zip_calls_import_roi_zip(self, app_fixture):
        win = app_fixture
        event = _make_mock_drop_event(["/tmp/rois.zip"])
        with patch.object(win, "import_roi_zip") as mock_import:
            win.dropEvent(event)
            mock_import.assert_called_once_with("/tmp/rois.zip")

    def test_drop_unsupported_shows_toast(self, app_fixture):
        win = app_fixture
        event = _make_mock_drop_event(["/tmp/data.csv"])
        with patch.object(win.toast, "show") as mock_toast:
            win.dropEvent(event)
            mock_toast.assert_called_once()
            args = mock_toast.call_args[0]
            assert "unsupported" in args[0].lower() or "Unsupported" in args[0]

    def test_drag_enter_accepts_urls(self, app_fixture):
        win = app_fixture
        event = _make_mock_drag_enter_event(["/tmp/photo.tif"])
        win.dragEnterEvent(event)
        event.acceptProposedAction.assert_called_once()

    def test_drop_image_and_zip_together(self, app_fixture):
        """When both image and zip are dropped, both handlers fire."""
        win = app_fixture
        event = _make_mock_drop_event(["/tmp/photo.png", "/tmp/rois.zip"])
        with patch.object(win, "open_image") as mock_open, \
             patch.object(win, "import_roi_zip") as mock_import:
            win.dropEvent(event)
            mock_open.assert_called_once_with(["/tmp/photo.png"])
            mock_import.assert_called_once_with("/tmp/rois.zip")


# ── _select_all_rois ────────────────────────────────────────────────


class TestSelectAllRois:
    def test_selects_visible_rois(self, app_fixture):
        win = app_fixture
        # Add two visible ROIs
        win.layer_stack.roi_layers.clear()
        r1 = ROILayer("A", 100, 100)
        r2 = ROILayer("B", 100, 100)
        r1.visible = True
        r2.visible = True
        win.layer_stack.roi_layers.extend([r1, r2])

        with patch.object(win.canvas._selection, "select_all") as mock_sel:
            win._select_all_rois()
            mock_sel.assert_called_once()
            selected = mock_sel.call_args[0][0]
            assert len(selected) == 2

    def test_skips_hidden_rois(self, app_fixture):
        win = app_fixture
        win.layer_stack.roi_layers.clear()
        r1 = ROILayer("A", 100, 100)
        r2 = ROILayer("B", 100, 100)
        r1.visible = True
        r2.visible = False
        win.layer_stack.roi_layers.extend([r1, r2])

        with patch.object(win.canvas._selection, "select_all") as mock_sel:
            win._select_all_rois()
            selected = mock_sel.call_args[0][0]
            assert len(selected) == 1
            assert selected[0] is r1


# ── _flatten_roi_offsets ─────────────────────────────────────────────


class TestFlattenRoiOffsets:
    def test_clears_offsets(self, app_fixture):
        win = app_fixture
        win.layer_stack.roi_layers.clear()
        r1 = ROILayer("A", 100, 100)
        r1.mask[10:20, 10:20] = 255
        r1.offset_x = 5
        r1.offset_y = 3
        r2 = ROILayer("B", 100, 100)
        r2.mask[30:40, 30:40] = 255
        r2.offset_x = -2
        r2.offset_y = 7
        win.layer_stack.roi_layers.extend([r1, r2])

        win._flatten_roi_offsets()

        assert r1.offset_x == 0
        assert r1.offset_y == 0
        assert r2.offset_x == 0
        assert r2.offset_y == 0

    def test_no_rois_is_safe(self, app_fixture):
        win = app_fixture
        win.layer_stack.roi_layers.clear()
        # Should not raise
        win._flatten_roi_offsets()
