"""Tests for Phase 6: Multiple Montage Support."""
import pytest
import numpy as np

from montaris.layers import MontageDocument, ImageLayer, ROILayer


class TestMontageDocument:
    def test_creation(self):
        data = np.zeros((50, 60), dtype=np.uint8)
        img = ImageLayer("test", data)
        doc = MontageDocument(name="test.tif", image_layer=img)
        assert doc.name == "test.tif"
        assert doc.image_layer is img
        assert doc.roi_layers == []
        assert doc.downsample_factor == 1
        assert doc.original_shape is None
        assert doc.adjustments['brightness'] == 0.0

    def test_with_rois(self):
        data = np.zeros((50, 60), dtype=np.uint8)
        img = ImageLayer("test", data)
        roi = ROILayer("roi1", 60, 50)
        doc = MontageDocument(name="test.tif", image_layer=img, roi_layers=[roi])
        assert len(doc.roi_layers) == 1

    def test_downsample_factor(self):
        data = np.zeros((50, 60), dtype=np.uint8)
        img = ImageLayer("test", data)
        doc = MontageDocument(
            name="test.tif", image_layer=img,
            downsample_factor=4, original_shape=(200, 240),
        )
        assert doc.downsample_factor == 4
        assert doc.original_shape == (200, 240)


class TestMultiDocument:
    def test_documents_list(self, qapp, app):
        assert hasattr(app, '_documents')
        assert app._documents == []
        assert app._active_doc_index == -1

    def test_doc_combo_exists(self, qapp, app):
        assert hasattr(app, '_doc_combo')

    def test_save_current_document_no_crash(self, qapp, app):
        """Calling save with no documents should not crash."""
        app._save_current_document()

    def test_switch_to_invalid(self, qapp, app):
        """Switching to invalid index should be a no-op."""
        app._switch_to_document(-1)
        app._switch_to_document(999)


class TestUpscaleMask:
    def test_no_upscale_needed(self, qapp, app_with_image):
        app = app_with_image
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        result = app._upscale_mask_if_needed(mask)
        assert result is mask  # identity when factor=1

    def test_upscale_2x(self, qapp, app_with_image):
        app = app_with_image
        app._downsample_factor = 2
        mask = np.zeros((50, 60), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        result = app._upscale_mask_if_needed(mask)
        assert result.shape[0] >= 100
        assert result.shape[1] >= 120
        app._downsample_factor = 1
