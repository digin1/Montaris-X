"""Tests for TilePyramid and TileCache."""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage

from montaris.core.tile_pyramid import TilePyramid, TILE_SIZE
from montaris.core.tile_cache import TileCache


# ── Ensure a QApplication exists (QImage needs one) ──────────────────
@pytest.fixture(scope="module", autouse=True)
def _ensure_qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield


# =====================================================================
# TileCache tests
# =====================================================================


class TestTileCache:
    def test_put_and_get(self):
        cache = TileCache(max_size=10)
        cache.put("a", 1)
        assert cache.get("a") == 1

    def test_get_missing_returns_none(self):
        cache = TileCache()
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        cache = TileCache(max_size=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_access_promotes(self):
        cache = TileCache(max_size=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.get("a")  # promote "a" so "b" becomes oldest
        cache.put("d", 4)  # should evict "b"
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_clear(self):
        cache = TileCache()
        cache.put("x", 42)
        cache.clear()
        assert cache.size == 0
        assert cache.get("x") is None

    def test_size(self):
        cache = TileCache()
        assert cache.size == 0
        cache.put("a", 1)
        assert cache.size == 1

    def test_overwrite_existing_key(self):
        cache = TileCache(max_size=5)
        cache.put("a", 1)
        cache.put("a", 2)
        assert cache.get("a") == 2
        assert cache.size == 1


# =====================================================================
# TilePyramid tests
# =====================================================================


class TestTilePyramidBasic:
    """Tests on a small (100 x 120) grayscale image."""

    @pytest.fixture
    def small_pyramid(self):
        data = np.random.randint(0, 255, (100, 120), dtype=np.uint8)
        return TilePyramid(data)

    def test_num_levels_small_image(self, small_pyramid):
        # 120x100 – both dims < TILE_SIZE, so only 1 level
        assert small_pyramid.num_levels == 1

    def test_level_size_level0(self, small_pyramid):
        w, h = small_pyramid.level_size(0)
        assert (w, h) == (120, 100)

    def test_get_tile_returns_qimage(self, small_pyramid):
        tile = small_pyramid.get_tile(0, 0, 0)
        assert isinstance(tile, QImage)
        assert tile.width() == 120  # entire image fits in one tile
        assert tile.height() == 100

    def test_get_tile_out_of_range_returns_none(self, small_pyramid):
        tile = small_pyramid.get_tile(0, 10, 10)
        assert tile is None

    def test_tile_is_cached(self, small_pyramid):
        t1 = small_pyramid.get_tile(0, 0, 0)
        t2 = small_pyramid.get_tile(0, 0, 0)
        assert t1 is t2


class TestTilePyramidLarge:
    """Tests on a 2048 x 2048 image with multiple pyramid levels."""

    @pytest.fixture
    def large_pyramid(self):
        data = np.random.randint(0, 255, (2048, 2048), dtype=np.uint8)
        return TilePyramid(data)

    def test_num_levels_large_image(self, large_pyramid):
        # 2048 >> 1 = 1024 >= 512, 2048 >> 2 = 512 >= 512,
        # 2048 >> 3 = 256 < 512 => stops at n=3
        assert large_pyramid.num_levels == 3

    def test_level_size_all_levels(self, large_pyramid):
        assert large_pyramid.level_size(0) == (2048, 2048)
        assert large_pyramid.level_size(1) == (1024, 1024)
        assert large_pyramid.level_size(2) == (512, 512)

    def test_tile_count_level0(self, large_pyramid):
        w, h = large_pyramid.level_size(0)
        ntx = math.ceil(w / TILE_SIZE)
        nty = math.ceil(h / TILE_SIZE)
        assert ntx == 4
        assert nty == 4

    def test_tile_count_level1(self, large_pyramid):
        w, h = large_pyramid.level_size(1)
        ntx = math.ceil(w / TILE_SIZE)
        nty = math.ceil(h / TILE_SIZE)
        assert ntx == 2
        assert nty == 2

    def test_tile_size_full(self, large_pyramid):
        tile = large_pyramid.get_tile(0, 0, 0)
        assert tile.width() == TILE_SIZE
        assert tile.height() == TILE_SIZE

    def test_tile_size_edge(self, large_pyramid):
        # level 0: 4x4 tiles, all exactly 512 for a 2048x2048 image
        tile = large_pyramid.get_tile(0, 3, 3)
        assert tile.width() == TILE_SIZE
        assert tile.height() == TILE_SIZE


class TestLevelForScale:
    @pytest.fixture
    def pyramid(self):
        data = np.random.randint(0, 255, (4096, 4096), dtype=np.uint8)
        return TilePyramid(data)

    def test_full_res_at_scale_1(self, pyramid):
        assert pyramid.level_for_scale(1.0) == 0

    def test_full_res_at_scale_above_1(self, pyramid):
        assert pyramid.level_for_scale(2.0) == 0

    def test_level1_at_half_scale(self, pyramid):
        assert pyramid.level_for_scale(0.5) == 1

    def test_level2_at_quarter_scale(self, pyramid):
        assert pyramid.level_for_scale(0.25) == 2

    def test_very_small_scale_clamps(self, pyramid):
        level = pyramid.level_for_scale(0.001)
        assert level == pyramid.num_levels - 1


class TestTileRange:
    @pytest.fixture
    def pyramid(self):
        data = np.random.randint(0, 255, (2048, 2048), dtype=np.uint8)
        return TilePyramid(data)

    def test_full_viewport(self, pyramid):
        rect = QRectF(0, 0, 2048, 2048)
        tx_min, ty_min, tx_max, ty_max = pyramid.tile_range(0, rect)
        assert tx_min == 0
        assert ty_min == 0
        assert tx_max == 3
        assert ty_max == 3

    def test_partial_viewport(self, pyramid):
        # Viewport covering only the first tile (just inside)
        rect = QRectF(0, 0, 511, 511)
        tx_min, ty_min, tx_max, ty_max = pyramid.tile_range(0, rect)
        assert tx_min == 0
        assert ty_min == 0
        assert tx_max == 0
        assert ty_max == 0

    def test_offset_viewport(self, pyramid):
        rect = QRectF(600, 600, 600, 600)
        tx_min, ty_min, tx_max, ty_max = pyramid.tile_range(0, rect)
        assert tx_min == 1
        assert ty_min == 1

    def test_tile_range_level1(self, pyramid):
        # At level 1, tile_size_scene = 512*2 = 1024
        rect = QRectF(0, 0, 2048, 2048)
        tx_min, ty_min, tx_max, ty_max = pyramid.tile_range(1, rect)
        assert tx_min == 0
        assert ty_min == 0
        assert tx_max == 1
        assert ty_max == 1


class TestTilePyramidRGB:
    """Ensure tiles work for multi-channel images."""

    def test_rgb_tile(self):
        data = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        p = TilePyramid(data)
        tile = p.get_tile(0, 0, 0)
        assert isinstance(tile, QImage)
        assert tile.width() == TILE_SIZE
        assert tile.height() == TILE_SIZE

    def test_rgba_tile(self):
        data = np.random.randint(0, 255, (600, 800, 4), dtype=np.uint8)
        p = TilePyramid(data)
        tile = p.get_tile(0, 0, 0)
        assert isinstance(tile, QImage)

    def test_float_image(self):
        data = np.random.rand(600, 800).astype(np.float32)
        p = TilePyramid(data)
        tile = p.get_tile(0, 0, 0)
        assert isinstance(tile, QImage)
