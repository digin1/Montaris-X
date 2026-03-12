"""Tests for rle_decode_crop() — memory-efficient partial mask decoding."""
import numpy as np
import pytest

from montaris.core.rle import rle_encode, rle_decode, rle_decode_crop


@pytest.fixture
def painted_mask():
    """100x100 mask with a 20x20 painted block at (30:50, 40:60)."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:50, 40:60] = 255
    return mask


@pytest.fixture
def encoded(painted_mask):
    return rle_encode(painted_mask)


class TestCropMatchesFullDecode:
    def test_interior_crop(self, encoded, painted_mask):
        data, shape = encoded
        bbox = (35, 45, 45, 55)  # interior of painted block
        crop = rle_decode_crop(data, shape, bbox)
        expected = painted_mask[35:45, 45:55]
        assert np.array_equal(crop, expected)

    def test_full_image_crop(self, encoded, painted_mask):
        data, shape = encoded
        bbox = (0, 100, 0, 100)
        crop = rle_decode_crop(data, shape, bbox)
        assert np.array_equal(crop, painted_mask)

    def test_arbitrary_crop(self, encoded, painted_mask):
        data, shape = encoded
        bbox = (20, 60, 30, 70)
        crop = rle_decode_crop(data, shape, bbox)
        expected = painted_mask[20:60, 30:70]
        assert np.array_equal(crop, expected)


class TestEdgeCrops:
    def test_top_left_corner(self, encoded, painted_mask):
        data, shape = encoded
        bbox = (0, 10, 0, 10)
        crop = rle_decode_crop(data, shape, bbox)
        expected = painted_mask[0:10, 0:10]
        assert np.array_equal(crop, expected)
        assert crop.shape == (10, 10)

    def test_bottom_right_corner(self, encoded, painted_mask):
        data, shape = encoded
        bbox = (90, 100, 90, 100)
        crop = rle_decode_crop(data, shape, bbox)
        expected = painted_mask[90:100, 90:100]
        assert np.array_equal(crop, expected)

    def test_single_pixel(self, encoded, painted_mask):
        data, shape = encoded
        # Inside painted region
        bbox = (35, 36, 45, 46)
        crop = rle_decode_crop(data, shape, bbox)
        assert crop.shape == (1, 1)
        assert crop[0, 0] == 255

    def test_single_pixel_outside(self, encoded, painted_mask):
        data, shape = encoded
        bbox = (0, 1, 0, 1)
        crop = rle_decode_crop(data, shape, bbox)
        assert crop.shape == (1, 1)
        assert crop[0, 0] == 0


class TestEmptyMask:
    def test_empty_mask_crop(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        data, shape = rle_encode(mask)
        bbox = (10, 20, 10, 20)
        crop = rle_decode_crop(data, shape, bbox)
        assert crop.shape == (10, 10)
        assert np.all(crop == 0)


class TestCropOutsidePaintedRegion:
    def test_crop_above_painted(self, encoded):
        data, shape = encoded
        bbox = (0, 25, 0, 100)  # entirely above painted block
        crop = rle_decode_crop(data, shape, bbox)
        assert np.all(crop == 0)

    def test_crop_right_of_painted(self, encoded):
        data, shape = encoded
        bbox = (0, 100, 70, 100)  # entirely right of painted block
        crop = rle_decode_crop(data, shape, bbox)
        assert np.all(crop == 0)


class TestRoundtrip:
    def test_encode_decode_crop_roundtrip(self):
        """Encode, then decode_crop at multiple locations, verify each."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[10:30, 10:30] = 128
        mask[50:80, 100:150] = 255
        mask[150:190, 20:60] = 64
        data, shape = rle_encode(mask)

        crops = [
            (0, 40, 0, 40),
            (40, 100, 90, 160),
            (140, 200, 10, 70),
            (0, 200, 0, 200),
        ]
        for bbox in crops:
            y1, y2, x1, x2 = bbox
            crop = rle_decode_crop(data, shape, bbox)
            expected = mask[y1:y2, x1:x2]
            assert np.array_equal(crop, expected), f"Mismatch at bbox {bbox}"

    def test_various_densities(self):
        """Test with sparse, dense, and full masks."""
        rng = np.random.RandomState(42)
        for density in [0.01, 0.1, 0.5, 0.9, 1.0]:
            mask = (rng.random((80, 80)) < density).astype(np.uint8) * 255
            data, shape = rle_encode(mask)
            bbox = (10, 60, 10, 60)
            crop = rle_decode_crop(data, shape, bbox)
            expected = mask[10:60, 10:60]
            assert np.array_equal(crop, expected), f"Failed at density={density}"
