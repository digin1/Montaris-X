"""Tests for ROILayer memory features: shape, compression, crop, duplicate."""
import numpy as np
import pytest

from montaris.layers import ROILayer, LayerStack, ImageLayer
from montaris.core.rle import rle_encode, rle_decode


class TestShapeProperty:
    def test_shape_returns_correct_dims(self):
        roi = ROILayer("test", 120, 100)
        assert roi.shape == (100, 120)

    def test_shape_on_compressed_layer(self):
        roi = ROILayer("test", 120, 100)
        roi.mask[10:20, 10:20] = 255
        roi.compress()
        assert roi.is_compressed
        # shape must work without decompression
        assert roi.shape == (100, 120)
        assert roi._mask is None  # must NOT have decompressed

    def test_shape_after_decompress(self):
        roi = ROILayer("test", 80, 60)
        roi.compress()
        _ = roi.mask  # triggers decompress
        assert roi.shape == (60, 80)


class TestCompressDecompress:
    def test_compress_frees_mask(self):
        roi = ROILayer("test", 100, 80)
        roi.mask[5:15, 5:15] = 255
        roi.compress()
        assert roi.is_compressed
        assert roi._mask is None
        assert roi._rle_data is not None

    def test_decompress_restores_mask(self):
        roi = ROILayer("test", 100, 80)
        roi.mask[5:15, 5:15] = 255
        original = roi.mask.copy()
        roi.compress()
        restored = roi.mask  # auto-decompresses
        assert not roi.is_compressed
        assert np.array_equal(restored, original)

    def test_compress_decompress_cycle(self):
        roi = ROILayer("test", 200, 150)
        roi.mask[20:40, 30:70] = 128
        original = roi.mask.copy()
        for _ in range(5):
            roi.compress()
            assert roi.is_compressed
            _ = roi.mask
            assert not roi.is_compressed
        assert np.array_equal(roi.mask, original)

    def test_empty_mask_compress(self):
        roi = ROILayer("test", 50, 50)
        roi.compress()
        assert roi.is_compressed
        decompressed = roi.mask
        assert np.all(decompressed == 0)


class TestGetMaskCrop:
    def test_crop_decompressed(self):
        roi = ROILayer("test", 100, 80)
        roi.mask[10:20, 10:20] = 255
        crop = roi.get_mask_crop((10, 20, 10, 20))
        assert crop.shape == (10, 10)
        assert np.all(crop == 255)

    def test_crop_compressed(self):
        roi = ROILayer("test", 100, 80)
        roi.mask[10:20, 10:20] = 255
        roi.compress()
        crop = roi.get_mask_crop((10, 20, 10, 20))
        assert crop.shape == (10, 10)
        assert np.all(crop == 255)
        # Should still be compressed (no full decompression)
        assert roi._mask is None

    def test_crop_outside_painted(self):
        roi = ROILayer("test", 100, 80)
        roi.mask[10:20, 10:20] = 255
        roi.compress()
        crop = roi.get_mask_crop((50, 60, 50, 60))
        assert np.all(crop == 0)


class TestGetBbox:
    def test_bbox_nonempty(self):
        roi = ROILayer("test", 100, 80)
        roi.mask[10:20, 30:50] = 255
        bbox = roi.get_bbox()
        assert bbox == (10, 20, 30, 50)

    def test_bbox_empty(self):
        roi = ROILayer("test", 100, 80)
        bbox = roi.get_bbox()
        assert bbox is None

    def test_bbox_caching(self):
        roi = ROILayer("test", 100, 80)
        roi.mask[10:20, 10:20] = 255
        bbox1 = roi.get_bbox()
        bbox2 = roi.get_bbox()
        assert bbox1 == bbox2

    def test_invalidate_bbox(self):
        roi = ROILayer("test", 100, 80)
        roi.mask[10:20, 10:20] = 255
        roi.get_bbox()
        assert roi._bbox_valid
        roi.invalidate_bbox()
        assert not roi._bbox_valid


class TestNewPattern:
    def test_roi_new_init(self):
        """ROILayer.__new__() pattern used in import/duplicate paths."""
        roi = ROILayer.__new__(ROILayer)
        roi.name = "imported"
        roi._mask_shape = (80, 100)
        roi._mask = np.zeros((80, 100), dtype=np.uint8)
        roi._rle_data = None
        roi._bbox_valid = False
        roi._cached_bbox = None
        roi.offset_x = 0
        roi.offset_y = 0
        roi.color = (255, 0, 0)
        roi.opacity = 128
        roi.visible = True
        roi.fill_mode = "solid"
        roi._dirty_rect = None

        assert roi.shape == (80, 100)
        assert roi.mask.shape == (80, 100)


class TestDuplicateRoi:
    def test_duplicate_preserves_data(self):
        ls = LayerStack()
        ls.set_image(ImageLayer("img", np.zeros((80, 100), dtype=np.uint8)))
        roi = ROILayer("original", 100, 80)
        roi.mask[10:30, 20:40] = 200
        ls.add_roi(roi)
        ls.duplicate_roi(0)

        assert len(ls.roi_layers) == 2
        dup = ls.roi_layers[1]
        assert np.array_equal(dup.mask, roi.mask)
        assert dup.name != roi.name or dup.color != roi.color

    def test_duplicate_compressed(self):
        ls = LayerStack()
        ls.set_image(ImageLayer("img", np.zeros((80, 100), dtype=np.uint8)))
        roi = ROILayer("original", 100, 80)
        roi.mask[10:30, 20:40] = 200
        roi.compress()
        ls.add_roi(roi)
        ls.duplicate_roi(0)

        dup = ls.roi_layers[1]
        # Duplicate of compressed should also be compressed
        assert dup._rle_data is not None
        # Data should match after decompression
        assert np.array_equal(dup.mask, roi.mask)


class TestFlattenOffset:
    def test_flatten_moves_content(self):
        roi = ROILayer("test", 100, 80)
        roi.mask[10:20, 10:20] = 255
        roi.offset_x = 5
        roi.offset_y = 3
        result = roi.flatten_offset()
        assert result is True
        assert roi.offset_x == 0
        assert roi.offset_y == 0
        # Content should now be at (13:23, 15:25)
        assert roi.mask[13, 15] == 255

    def test_flatten_no_offset(self):
        roi = ROILayer("test", 100, 80)
        roi.mask[10:20, 10:20] = 255
        original = roi.mask.copy()
        result = roi.flatten_offset()
        assert result is True
        assert np.array_equal(roi.mask, original)
