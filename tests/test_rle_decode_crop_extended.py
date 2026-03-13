"""Extended tests for rle_decode_crop() — numpy fallback path.

Forces the numpy fallback by patching the accel import to raise ImportError,
ensuring the pure-numpy code path in rle.py is fully exercised.
"""
import numpy as np
import pytest
from unittest.mock import patch

from montaris.core.rle import rle_encode, rle_decode, rle_decode_crop


# ---------------------------------------------------------------------------
# Helper: force numpy fallback by making the accel import fail
# ---------------------------------------------------------------------------

def _decode_crop_numpy(data, shape, bbox):
    """Call rle_decode_crop with the accel module blocked."""
    with patch.dict("sys.modules", {"montaris.core.accel": None}):
        return rle_decode_crop(data, shape, bbox)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def painted_mask():
    """100x100 mask with a 20x20 painted block at rows 30-50, cols 40-60."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:50, 40:60] = 255
    return mask


@pytest.fixture
def encoded(painted_mask):
    return rle_encode(painted_mask)


# ---------------------------------------------------------------------------
# 1. Basic crop within a painted region
# ---------------------------------------------------------------------------

class TestBasicCropWithinPainted:
    def test_interior_crop_returns_all_painted(self, encoded, painted_mask):
        data, shape = encoded
        bbox = (35, 45, 45, 55)
        crop = _decode_crop_numpy(data, shape, bbox)
        expected = painted_mask[35:45, 45:55]
        np.testing.assert_array_equal(crop, expected)
        assert crop.dtype == np.uint8

    def test_overlapping_crop_has_partial_paint(self, encoded, painted_mask):
        data, shape = encoded
        bbox = (25, 40, 35, 50)  # overlaps top-left corner of painted block
        crop = _decode_crop_numpy(data, shape, bbox)
        expected = painted_mask[25:40, 35:50]
        np.testing.assert_array_equal(crop, expected)


# ---------------------------------------------------------------------------
# 2. Crop entirely outside painted area -> all zeros
# ---------------------------------------------------------------------------

class TestCropOutsidePaintedArea:
    def test_crop_above_painted_region(self, encoded):
        data, shape = encoded
        bbox = (0, 20, 0, 100)
        crop = _decode_crop_numpy(data, shape, bbox)
        assert np.all(crop == 0)
        assert crop.shape == (20, 100)

    def test_crop_below_painted_region(self, encoded):
        data, shape = encoded
        bbox = (60, 100, 0, 100)
        crop = _decode_crop_numpy(data, shape, bbox)
        assert np.all(crop == 0)

    def test_crop_left_of_painted_region(self, encoded):
        data, shape = encoded
        bbox = (30, 50, 0, 30)
        crop = _decode_crop_numpy(data, shape, bbox)
        assert np.all(crop == 0)

    def test_crop_right_of_painted_region(self, encoded):
        data, shape = encoded
        bbox = (30, 50, 70, 100)
        crop = _decode_crop_numpy(data, shape, bbox)
        assert np.all(crop == 0)


# ---------------------------------------------------------------------------
# 3. Crop spanning multiple RLE runs
# ---------------------------------------------------------------------------

class TestCropSpanningMultipleRuns:
    def test_two_disjoint_blocks(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:30] = 255
        mask[10:20, 50:70] = 255
        data, shape = rle_encode(mask)
        # Crop across the gap between the two blocks
        bbox = (10, 20, 5, 75)
        crop = _decode_crop_numpy(data, shape, bbox)
        expected = mask[10:20, 5:75]
        np.testing.assert_array_equal(crop, expected)

    def test_alternating_rows(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[0, :] = 255
        mask[2, :] = 255
        mask[4, :] = 255
        data, shape = rle_encode(mask)
        bbox = (0, 5, 0, 20)
        crop = _decode_crop_numpy(data, shape, bbox)
        expected = mask[0:5, 0:20]
        np.testing.assert_array_equal(crop, expected)


# ---------------------------------------------------------------------------
# 4. Empty data -> zeros
# ---------------------------------------------------------------------------

class TestEmptyData:
    def test_empty_bytes_returns_zeros(self):
        bbox = (5, 15, 5, 15)
        crop = _decode_crop_numpy(b'', (100, 100), bbox)
        assert crop.shape == (10, 10)
        assert np.all(crop == 0)

    def test_all_zero_mask_encoded(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        data, shape = rle_encode(mask)
        bbox = (10, 30, 10, 30)
        crop = _decode_crop_numpy(data, shape, bbox)
        assert crop.shape == (20, 20)
        assert np.all(crop == 0)


# ---------------------------------------------------------------------------
# 5. Zero-size bbox (crop_h <= 0 or crop_w <= 0) -> empty
# ---------------------------------------------------------------------------

class TestZeroSizeBbox:
    def test_zero_height(self, encoded):
        data, shape = encoded
        bbox = (40, 40, 10, 50)  # y1 == y2
        crop = _decode_crop_numpy(data, shape, bbox)
        assert crop.shape == (0, 40)

    def test_zero_width(self, encoded):
        data, shape = encoded
        bbox = (10, 50, 30, 30)  # x1 == x2
        crop = _decode_crop_numpy(data, shape, bbox)
        assert crop.shape == (40, 0)

    def test_negative_height_raises(self, encoded):
        data, shape = encoded
        bbox = (50, 30, 10, 50)  # y2 < y1 -> negative crop_h
        with pytest.raises(ValueError):
            _decode_crop_numpy(data, shape, bbox)

    def test_negative_width_raises(self, encoded):
        data, shape = encoded
        bbox = (10, 50, 60, 30)  # x2 < x1 -> negative crop_w
        with pytest.raises(ValueError):
            _decode_crop_numpy(data, shape, bbox)

    def test_both_zero(self, encoded):
        data, shape = encoded
        bbox = (20, 20, 30, 30)
        crop = _decode_crop_numpy(data, shape, bbox)
        assert crop.shape == (0, 0)


# ---------------------------------------------------------------------------
# 6. Runs with value==0 filtered (background runs should be ignored)
# ---------------------------------------------------------------------------

class TestBackgroundRunsFiltered:
    def test_only_background_runs_in_crop(self):
        """When the crop window only contains background, result is zeros."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[40:50, 40:50] = 255  # paint in bottom-right corner
        data, shape = rle_encode(mask)
        bbox = (0, 30, 0, 30)  # entirely in the background area
        crop = _decode_crop_numpy(data, shape, bbox)
        assert np.all(crop == 0)

    def test_value_zero_runs_do_not_overwrite(self):
        """Ensure the zero-valued runs don't accidentally set pixels."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 200
        data, shape = rle_encode(mask)
        bbox = (0, 10, 0, 10)
        crop = _decode_crop_numpy(data, shape, bbox)
        np.testing.assert_array_equal(crop, mask)


# ---------------------------------------------------------------------------
# 7. Single-pixel crop
# ---------------------------------------------------------------------------

class TestSinglePixelCrop:
    def test_single_pixel_painted(self, encoded):
        data, shape = encoded
        bbox = (35, 36, 45, 46)
        crop = _decode_crop_numpy(data, shape, bbox)
        assert crop.shape == (1, 1)
        assert crop[0, 0] == 255

    def test_single_pixel_unpainted(self, encoded):
        data, shape = encoded
        bbox = (0, 1, 0, 1)
        crop = _decode_crop_numpy(data, shape, bbox)
        assert crop.shape == (1, 1)
        assert crop[0, 0] == 0

    def test_single_pixel_edge_of_painted(self, encoded):
        """Bottom-right corner pixel of the painted block."""
        data, shape = encoded
        bbox = (49, 50, 59, 60)
        crop = _decode_crop_numpy(data, shape, bbox)
        assert crop.shape == (1, 1)
        assert crop[0, 0] == 255

    def test_single_pixel_just_outside_painted(self, encoded):
        """One pixel past the bottom-right corner of the painted block."""
        data, shape = encoded
        bbox = (50, 51, 60, 61)
        crop = _decode_crop_numpy(data, shape, bbox)
        assert crop.shape == (1, 1)
        assert crop[0, 0] == 0


# ---------------------------------------------------------------------------
# 8. Full-mask crop (bbox covers entire mask)
# ---------------------------------------------------------------------------

class TestFullMaskCrop:
    def test_full_crop_matches_full_decode(self, encoded, painted_mask):
        data, shape = encoded
        bbox = (0, 100, 0, 100)
        crop = _decode_crop_numpy(data, shape, bbox)
        np.testing.assert_array_equal(crop, painted_mask)

    def test_full_crop_complex_mask(self):
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[5:15, 10:30] = 100
        mask[20:40, 50:70] = 200
        mask[45:55, 0:80] = 50
        data, shape = rle_encode(mask)
        bbox = (0, 60, 0, 80)
        crop = _decode_crop_numpy(data, shape, bbox)
        np.testing.assert_array_equal(crop, mask)


# ---------------------------------------------------------------------------
# 9. Column filtering: runs spanning columns outside bbox are clipped
# ---------------------------------------------------------------------------

class TestColumnFiltering:
    def test_run_spans_left_of_bbox(self):
        """A painted row that extends left of the crop window is clipped."""
        mask = np.zeros((10, 100), dtype=np.uint8)
        mask[5, 10:90] = 255  # wide run across most of the row
        data, shape = rle_encode(mask)
        bbox = (4, 7, 50, 80)
        crop = _decode_crop_numpy(data, shape, bbox)
        expected = mask[4:7, 50:80]
        np.testing.assert_array_equal(crop, expected)

    def test_run_spans_right_of_bbox(self):
        """A painted row extending right of the crop window is clipped."""
        mask = np.zeros((10, 100), dtype=np.uint8)
        mask[5, 10:90] = 255
        data, shape = rle_encode(mask)
        bbox = (4, 7, 10, 40)
        crop = _decode_crop_numpy(data, shape, bbox)
        expected = mask[4:7, 10:40]
        np.testing.assert_array_equal(crop, expected)

    def test_run_wider_than_bbox_both_sides(self):
        """Run extends past both left and right edges of the bbox."""
        mask = np.zeros((5, 100), dtype=np.uint8)
        mask[2, :] = 255  # entire row painted
        data, shape = rle_encode(mask)
        bbox = (1, 4, 30, 60)
        crop = _decode_crop_numpy(data, shape, bbox)
        expected = mask[1:4, 30:60]
        np.testing.assert_array_equal(crop, expected)


# ---------------------------------------------------------------------------
# 10. Large mask with small crop (memory efficiency)
# ---------------------------------------------------------------------------

class TestLargeMaskSmallCrop:
    def test_large_mask_small_interior_crop(self):
        """Decode a crop from a large mask without allocating the full mask."""
        mask = np.zeros((2000, 2000), dtype=np.uint8)
        mask[900:1100, 900:1100] = 255
        data, shape = rle_encode(mask)
        bbox = (950, 1050, 950, 1050)
        crop = _decode_crop_numpy(data, shape, bbox)
        expected = mask[950:1050, 950:1050]
        np.testing.assert_array_equal(crop, expected)
        assert crop.shape == (100, 100)

    def test_large_mask_crop_all_zeros(self):
        """Crop from a region outside the painted area of a large mask."""
        mask = np.zeros((3000, 3000), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        data, shape = rle_encode(mask)
        bbox = (2500, 2600, 2500, 2600)
        crop = _decode_crop_numpy(data, shape, bbox)
        assert np.all(crop == 0)
        assert crop.shape == (100, 100)


# ---------------------------------------------------------------------------
# 11. Roundtrip: rle_encode -> rle_decode_crop matches direct crop of original
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_roundtrip_single_block(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[20:60, 20:60] = 255
        data, shape = rle_encode(mask)
        for bbox in [
            (0, 80, 0, 80),
            (25, 55, 25, 55),
            (0, 40, 0, 40),
            (40, 80, 40, 80),
            (10, 70, 30, 50),
        ]:
            y1, y2, x1, x2 = bbox
            crop = _decode_crop_numpy(data, shape, bbox)
            expected = mask[y1:y2, x1:x2]
            np.testing.assert_array_equal(crop, expected, err_msg=f"bbox={bbox}")

    def test_roundtrip_multiple_blocks(self):
        mask = np.zeros((150, 150), dtype=np.uint8)
        mask[10:30, 10:50] = 128
        mask[60:90, 80:130] = 255
        mask[110:140, 20:60] = 64
        data, shape = rle_encode(mask)
        for bbox in [
            (0, 150, 0, 150),
            (5, 35, 5, 55),
            (55, 95, 75, 135),
            (105, 145, 15, 65),
            (50, 100, 0, 150),
        ]:
            y1, y2, x1, x2 = bbox
            crop = _decode_crop_numpy(data, shape, bbox)
            expected = mask[y1:y2, x1:x2]
            np.testing.assert_array_equal(crop, expected, err_msg=f"bbox={bbox}")

    def test_roundtrip_random_masks(self):
        rng = np.random.RandomState(99)
        for _ in range(5):
            h, w = rng.randint(50, 200), rng.randint(50, 200)
            mask = (rng.random((h, w)) < 0.3).astype(np.uint8) * 255
            data, shape = rle_encode(mask)
            y1 = rng.randint(0, h // 2)
            y2 = rng.randint(h // 2, h)
            x1 = rng.randint(0, w // 2)
            x2 = rng.randint(w // 2, w)
            bbox = (y1, y2, x1, x2)
            crop = _decode_crop_numpy(data, shape, bbox)
            expected = mask[y1:y2, x1:x2]
            np.testing.assert_array_equal(crop, expected, err_msg=f"bbox={bbox}")


# ---------------------------------------------------------------------------
# 12. Multi-value mask (not just 0/255, e.g. 128)
# ---------------------------------------------------------------------------

class TestMultiValueMask:
    def test_value_128(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:20, 10:20] = 128
        data, shape = rle_encode(mask)
        bbox = (5, 25, 5, 25)
        crop = _decode_crop_numpy(data, shape, bbox)
        expected = mask[5:25, 5:25]
        np.testing.assert_array_equal(crop, expected)

    def test_multiple_distinct_values(self):
        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[5:15, 5:15] = 64
        mask[20:30, 20:30] = 128
        mask[40:50, 40:50] = 200
        data, shape = rle_encode(mask)
        bbox = (0, 60, 0, 60)
        crop = _decode_crop_numpy(data, shape, bbox)
        np.testing.assert_array_equal(crop, mask)

    def test_crop_only_hits_one_value(self):
        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[5:15, 5:15] = 64
        mask[40:50, 40:50] = 200
        data, shape = rle_encode(mask)
        bbox = (3, 18, 3, 18)
        crop = _decode_crop_numpy(data, shape, bbox)
        expected = mask[3:18, 3:18]
        np.testing.assert_array_equal(crop, expected)
        # Verify the crop contains only 0 and 64
        unique = set(np.unique(crop))
        assert unique <= {0, 64}

    def test_value_1_minimal(self):
        """Value 1 should not be filtered out (only value 0 is background)."""
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:10, 5:10] = 1
        data, shape = rle_encode(mask)
        bbox = (0, 20, 0, 20)
        crop = _decode_crop_numpy(data, shape, bbox)
        np.testing.assert_array_equal(crop, mask)
