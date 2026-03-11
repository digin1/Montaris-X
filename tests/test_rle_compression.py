import numpy as np
import pytest
from montaris.core.rle import rle_encode, rle_decode
from montaris.layers import ROILayer, LayerStack


class TestRLEEncodeDecode:
    def test_rle_roundtrip(self):
        """Encode/decode preserves exact mask data."""
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[10:30, 50:80] = 255
        mask[60:70, 100:150] = 128
        data, shape = rle_encode(mask)
        result = rle_decode(data, shape)
        np.testing.assert_array_equal(result, mask)

    def test_rle_empty_mask(self):
        """All-zeros mask compresses to minimal bytes."""
        mask = np.zeros((500, 500), dtype=np.uint8)
        data, shape = rle_encode(mask)
        # Single run: 1 pair of 5 bytes
        assert len(data) == 5
        result = rle_decode(data, shape)
        np.testing.assert_array_equal(result, mask)

    def test_rle_compression_ratio(self):
        """Sparse mask achieves >100x compression."""
        mask = np.zeros((1000, 1000), dtype=np.uint8)
        # ~1% fill
        mask[100:110, 200:300] = 255
        raw_size = mask.nbytes
        data, shape = rle_encode(mask)
        compressed_size = len(data)
        ratio = raw_size / compressed_size
        assert ratio > 100

    def test_rle_full_mask(self):
        """Fully filled mask roundtrips correctly."""
        mask = np.full((50, 50), 255, dtype=np.uint8)
        data, shape = rle_encode(mask)
        result = rle_decode(data, shape)
        np.testing.assert_array_equal(result, mask)

    def test_rle_empty_array(self):
        """Zero-length array handled gracefully."""
        mask = np.zeros((0, 0), dtype=np.uint8)
        data, shape = rle_encode(mask)
        assert data == b''
        result = rle_decode(data, shape)
        assert result.shape == (0, 0)


class TestROILayerCompression:
    def test_roi_compress_decompress(self):
        """ROILayer.compress() + .mask access restores correctly."""
        roi = ROILayer("test", 200, 100)
        roi.mask[10:20, 30:50] = 255
        original = roi.mask.copy()
        roi.compress()
        assert roi.is_compressed
        assert roi._mask is None
        # Accessing .mask decompresses
        np.testing.assert_array_equal(roi.mask, original)
        assert not roi.is_compressed

    def test_roi_shape_when_compressed(self):
        """.shape works without decompressing."""
        roi = ROILayer("test", 200, 100)
        assert roi.shape == (100, 200)
        roi.compress()
        assert roi.shape == (100, 200)
        assert roi.is_compressed  # still compressed

    def test_roi_mask_setter_clears_rle(self):
        """Setting .mask clears ._rle_data."""
        roi = ROILayer("test", 200, 100)
        roi.compress()
        assert roi.is_compressed
        new_mask = np.ones((100, 200), dtype=np.uint8) * 128
        roi.mask = new_mask
        assert not roi.is_compressed
        assert roi._rle_data is None
        np.testing.assert_array_equal(roi.mask, new_mask)

    def test_get_mask_crop_compressed(self):
        """get_mask_crop on compressed layer returns correct crop without caching full mask."""
        roi = ROILayer("test", 200, 100)
        roi.mask[10:20, 30:50] = 255
        expected = roi.mask[10:20, 30:50].copy()
        roi.compress()
        crop = roi.get_mask_crop((10, 20, 30, 50))
        np.testing.assert_array_equal(crop, expected)
        # Should still be compressed (no caching of full mask)
        assert roi.is_compressed

    def test_get_mask_crop_uncompressed(self):
        """get_mask_crop on uncompressed layer returns view."""
        roi = ROILayer("test", 200, 100)
        roi.mask[10:20, 30:50] = 255
        crop = roi.get_mask_crop((10, 20, 30, 50))
        np.testing.assert_array_equal(crop, roi.mask[10:20, 30:50])
        assert not roi.is_compressed

    def test_is_compressed_property(self):
        """is_compressed reflects state correctly."""
        roi = ROILayer("test", 100, 100)
        assert not roi.is_compressed
        roi.compress()
        assert roi.is_compressed
        _ = roi.mask  # decompress
        assert not roi.is_compressed


class TestLayerStackCompressInactive:
    def test_compress_inactive(self, qapp):
        """compress_inactive() compresses all except active."""
        stack = LayerStack()
        r1 = ROILayer("A", 100, 100)
        r2 = ROILayer("B", 100, 100)
        r3 = ROILayer("C", 100, 100)
        r1.mask[0:10, 0:10] = 255
        r2.mask[20:30, 20:30] = 255
        r3.mask[40:50, 40:50] = 255
        stack.roi_layers = [r1, r2, r3]

        stack.compress_inactive(active_layer=r2)

        assert r1.is_compressed
        assert not r2.is_compressed
        assert r3.is_compressed

    def test_compress_inactive_none_active(self, qapp):
        """compress_inactive(None) compresses all layers."""
        stack = LayerStack()
        r1 = ROILayer("A", 100, 100)
        r2 = ROILayer("B", 100, 100)
        stack.roi_layers = [r1, r2]

        stack.compress_inactive(active_layer=None)

        assert r1.is_compressed
        assert r2.is_compressed
