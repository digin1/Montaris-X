"""Extended tests for montaris.core.tile_pyramid — downsample, level data, eviction."""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from montaris.core.tile_pyramid import TilePyramid, TILE_SIZE
from montaris.core.tile_cache import TileCache


# ── Ensure a QApplication exists (QImage needs one) ──────────────────
@pytest.fixture(scope="module", autouse=True)
def _ensure_qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield


# ---------------------------------------------------------------------------
# 1. TilePyramid._downsample
# ---------------------------------------------------------------------------

class TestDownsample:
    def test_2d_shape_halved(self):
        arr = np.random.randint(0, 255, (20, 30), dtype=np.uint8)
        ds = TilePyramid._downsample(arr)
        assert ds.shape == (10, 15)

    def test_3d_shape_halved(self):
        arr = np.random.randint(0, 255, (20, 30, 3), dtype=np.uint8)
        ds = TilePyramid._downsample(arr)
        assert ds.shape == (10, 15, 3)

    def test_2d_values_averaged(self):
        # Constant array: average should equal the constant
        arr = np.full((10, 10), 100, dtype=np.uint8)
        ds = TilePyramid._downsample(arr)
        assert (ds == 100).all()

    def test_2d_checkerboard_average(self):
        # 2x2 blocks of [0,0,200,200] -> average 100
        arr = np.zeros((4, 4), dtype=np.uint8)
        arr[0, 2:4] = 200
        arr[1, 2:4] = 200
        arr[2, 2:4] = 200
        arr[3, 2:4] = 200
        ds = TilePyramid._downsample(arr)
        assert ds.shape == (2, 2)
        # Left column: average of [0,0,0,0] per 2x2 = 0
        assert ds[0, 0] == 0
        assert ds[1, 0] == 0
        # Right column: average of [200,200,200,200] per 2x2 = 200
        assert ds[0, 1] == 200
        assert ds[1, 1] == 200

    def test_3d_values_averaged(self):
        arr = np.full((8, 8, 3), 50, dtype=np.uint8)
        ds = TilePyramid._downsample(arr)
        assert (ds == 50).all()

    def test_dtype_preserved(self):
        arr = np.ones((10, 10), dtype=np.uint8) * 42
        ds = TilePyramid._downsample(arr)
        assert ds.dtype == np.uint8

    def test_odd_dimension_cropped(self):
        # 11x13 -> crops to 10x12 before averaging -> 5x6
        arr = np.random.randint(0, 255, (11, 13), dtype=np.uint8)
        ds = TilePyramid._downsample(arr)
        assert ds.shape == (5, 6)

    def test_odd_dimension_3d(self):
        arr = np.random.randint(0, 255, (11, 13, 4), dtype=np.uint8)
        ds = TilePyramid._downsample(arr)
        assert ds.shape == (5, 6, 4)


# ---------------------------------------------------------------------------
# 2. TilePyramid._get_level_data
# ---------------------------------------------------------------------------

class TestGetLevelData:
    def test_level0_is_original(self):
        data = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
        pyr = TilePyramid(data)
        level0 = pyr._get_level_data(0)
        assert level0 is data

    def test_level1_is_half_size(self):
        data = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
        pyr = TilePyramid(data)
        level1 = pyr._get_level_data(1)
        assert level1.shape == (512, 512)

    def test_levels_built_progressively(self):
        data = np.random.randint(0, 255, (2048, 2048), dtype=np.uint8)
        pyr = TilePyramid(data)
        # Access level 2 directly — should also create level 1
        level2 = pyr._get_level_data(2)
        assert level2.shape == (512, 512)
        # Level 1 should also be available (either cached or recreatable)
        level1 = pyr._get_level_data(1)
        assert level1.shape == (1024, 1024)


# ---------------------------------------------------------------------------
# 3. Level eviction
# ---------------------------------------------------------------------------

class TestLevelEviction:
    def test_eviction_keeps_level0(self):
        """Level 0 is never evicted, even after accessing many levels."""
        # Need an image large enough for multiple levels
        data = np.random.randint(0, 255, (4096, 4096), dtype=np.uint8)
        pyr = TilePyramid(data)
        assert pyr.num_levels > 3  # Should have several levels

        # Access levels 1, 2, 3 to trigger eviction
        pyr._get_level_data(1)
        pyr._get_level_data(2)
        pyr._get_level_data(3)

        # Level 0 must still be present
        assert 0 in pyr._level_data

    def test_eviction_limits_cached_levels(self):
        """Only ~3 levels (0 + 2 others) should be in cache at any time."""
        data = np.random.randint(0, 255, (4096, 4096), dtype=np.uint8)
        pyr = TilePyramid(data)

        # Access all levels
        for lv in range(pyr.num_levels):
            pyr._get_level_data(lv)

        # Should have at most 3 levels cached (level 0 + 2 most recent)
        assert len(pyr._level_data) <= 3

    def test_evicted_level_rebuilt_on_access(self):
        """Accessing an evicted level should rebuild it correctly."""
        data = np.random.randint(0, 255, (4096, 4096), dtype=np.uint8)
        pyr = TilePyramid(data)

        # Access level 1 first
        level1_first = pyr._get_level_data(1)
        expected_shape = level1_first.shape

        # Now access higher levels to potentially evict level 1
        pyr._get_level_data(2)
        pyr._get_level_data(3)

        # Re-access level 1 — should be rebuilt
        level1_again = pyr._get_level_data(1)
        assert level1_again.shape == expected_shape


# ---------------------------------------------------------------------------
# 4. Multi-level pyramid build and access
# ---------------------------------------------------------------------------

class TestMultiLevelPyramid:
    def test_small_image_single_level(self):
        data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        pyr = TilePyramid(data)
        assert pyr.num_levels == 1

    def test_medium_image_levels(self):
        # 1024x1024: 1024 >> 1 = 512 >= 512 -> n=2; 1024 >> 2 = 256 < 512 -> stop
        data = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
        pyr = TilePyramid(data)
        assert pyr.num_levels == 2

    def test_level_size_decreases(self):
        data = np.random.randint(0, 255, (2048, 2048), dtype=np.uint8)
        pyr = TilePyramid(data)
        for lv in range(pyr.num_levels - 1):
            w_cur, h_cur = pyr.level_size(lv)
            w_next, h_next = pyr.level_size(lv + 1)
            assert w_next <= w_cur
            assert h_next <= h_cur

    def test_clamp_level_negative(self):
        data = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
        pyr = TilePyramid(data)
        # Negative level should clamp to 0
        assert pyr._clamp_level(-1) == 0

    def test_clamp_level_too_high(self):
        data = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
        pyr = TilePyramid(data)
        assert pyr._clamp_level(999) == pyr.num_levels - 1

    def test_shared_tile_cache(self):
        """Two pyramids sharing a TileCache should not interfere."""
        cache = TileCache(max_size=100)
        data_a = np.random.randint(0, 255, (600, 800), dtype=np.uint8)
        data_b = np.random.randint(0, 255, (600, 800), dtype=np.uint8)
        pyr_a = TilePyramid(data_a, tile_cache=cache)
        pyr_b = TilePyramid(data_b, tile_cache=cache)

        tile_a = pyr_a.get_tile(0, 0, 0)
        tile_b = pyr_b.get_tile(0, 0, 0)
        assert tile_a is not None
        assert tile_b is not None
