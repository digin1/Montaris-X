"""Multi-resolution tile pyramid for large images.

Tiles are computed lazily on first access and stored in a TileCache.
Level 0 is the original resolution; each subsequent level is 2x
downsampled from the previous one.
"""

import math
import numpy as np
from PySide6.QtGui import QImage

from montaris.core.tile_cache import TileCache

TILE_SIZE = 512


def _array_tile_to_qimage(tile_arr):
    """Convert a numpy tile array to a QImage.

    Handles 2-D grayscale, 3-D with 1/3/4 channels, and non-uint8 dtypes
    (auto-normalised to 0-255).
    """
    if tile_arr.ndim == 2:
        h, w = tile_arr.shape
        if tile_arr.dtype != np.uint8:
            mn, mx = float(tile_arr.min()), float(tile_arr.max())
            if mx > mn:
                arr = ((tile_arr.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                arr = np.zeros_like(tile_arr, dtype=np.uint8)
        else:
            arr = tile_arr
        arr = np.ascontiguousarray(arr)
        img = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
        return img.copy()

    elif tile_arr.ndim == 3:
        h, w, c = tile_arr.shape
        if tile_arr.dtype != np.uint8:
            mn, mx = float(tile_arr.min()), float(tile_arr.max())
            if mx > mn:
                arr = ((tile_arr.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                arr = np.zeros((h, w, c), dtype=np.uint8)
        else:
            arr = tile_arr

        if c == 1:
            arr = arr[:, :, 0]
            arr = np.ascontiguousarray(arr)
            img = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
            return img.copy()

        arr = np.ascontiguousarray(arr)
        if c == 3:
            img = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
            return img.copy()
        elif c == 4:
            img = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888)
            return img.copy()

    raise ValueError(f"Unsupported tile array shape: {tile_arr.shape}")


class TilePyramid:
    """Multi-resolution tile pyramid for large images.

    * ``TILE_SIZE = 512``
    * Up to 10 levels (level 0 = full resolution)
    * Lazy tile building -- tiles computed on demand
    """

    def __init__(self, image_data, tile_cache=None):
        """
        Parameters
        ----------
        image_data : np.ndarray
            Full-resolution image (H, W) or (H, W, C).
        tile_cache : TileCache, optional
            Shared cache.  A private one is created when *None*.
        """
        self._data = image_data
        self._cache = tile_cache or TileCache()

        h, w = image_data.shape[:2]
        # Number of levels: keep halving until both dims < TILE_SIZE, max 10
        n = 1
        while n < 10 and (w >> n) >= TILE_SIZE and (h >> n) >= TILE_SIZE:
            n += 1
        # Ensure at least 1 level even for very small images
        self._num_levels = max(1, n)

        # Cache for downsampled full arrays at each level (lazy)
        self._level_data = {0: image_data}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_levels(self):
        """Total number of pyramid levels."""
        return self._num_levels

    def level_size(self, level):
        """Return ``(width, height)`` of the image at *level*."""
        level = self._clamp_level(level)
        h, w = self._data.shape[:2]
        factor = 1 << level
        return max(1, w // factor), max(1, h // factor)

    def level_for_scale(self, scale):
        """Given viewport *scale* (pixels-per-scene-unit), return the
        best pyramid level to use.

        ``scale >= 1`` -> level 0 (full res).  Each halving of scale
        increases the level by one.
        """
        if scale <= 0:
            return self._num_levels - 1
        level = max(0, int(math.log2(max(1.0, 1.0 / scale))))
        return min(level, self._num_levels - 1)

    def tile_range(self, level, viewport_rect):
        """Return ``(tx_min, ty_min, tx_max, ty_max)`` -- the inclusive
        range of tile coordinates needed to cover *viewport_rect* at
        *level*.

        *viewport_rect* should be in **scene** coordinates (i.e. level-0
        pixel coordinates).
        """
        level = self._clamp_level(level)
        factor = 1 << level
        tile_size_scene = TILE_SIZE * factor

        # viewport_rect is a QRectF in scene coords
        x1 = viewport_rect.left()
        y1 = viewport_rect.top()
        x2 = viewport_rect.right()
        y2 = viewport_rect.bottom()

        tx_min = max(0, int(math.floor(x1 / tile_size_scene)))
        ty_min = max(0, int(math.floor(y1 / tile_size_scene)))

        w, h = self.level_size(level)
        max_tx = max(0, math.ceil(w / TILE_SIZE) - 1)
        max_ty = max(0, math.ceil(h / TILE_SIZE) - 1)

        tx_max = min(max_tx, int(math.floor(x2 / tile_size_scene)))
        ty_max = min(max_ty, int(math.floor(y2 / tile_size_scene)))

        return tx_min, ty_min, tx_max, ty_max

    def get_tile(self, level, tile_x, tile_y):
        """Return a ``QImage`` for tile ``(tile_x, tile_y)`` at *level*.

        Tiles are computed lazily and cached.
        """
        level = self._clamp_level(level)
        key = (level, tile_x, tile_y)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        level_arr = self._get_level_data(level)
        h, w = level_arr.shape[:2]

        y0 = tile_y * TILE_SIZE
        x0 = tile_x * TILE_SIZE
        y1 = min(y0 + TILE_SIZE, h)
        x1 = min(x0 + TILE_SIZE, w)

        if x0 >= w or y0 >= h:
            return None

        tile_arr = level_arr[y0:y1, x0:x1]
        qimg = _array_tile_to_qimage(tile_arr)
        self._cache.put(key, qimg)
        return qimg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clamp_level(self, level):
        return max(0, min(level, self._num_levels - 1))

    def _get_level_data(self, level):
        """Return the full numpy array for *level*, computing it lazily
        by downsampling from the previous level."""
        if level in self._level_data:
            return self._level_data[level]

        # Build all intermediate levels we don't have yet
        for lv in range(1, level + 1):
            if lv in self._level_data:
                continue
            prev = self._get_level_data(lv - 1)
            self._level_data[lv] = self._downsample(prev)

        return self._level_data[level]

    @staticmethod
    def _downsample(arr):
        """Downsample *arr* by 2x using simple averaging."""
        if arr.ndim == 2:
            h, w = arr.shape
            h2, w2 = h // 2 * 2, w // 2 * 2
            cropped = arr[:h2, :w2]
            return (
                cropped.reshape(h2 // 2, 2, w2 // 2, 2)
                .mean(axis=(1, 3))
                .astype(arr.dtype)
            )
        else:
            h, w, c = arr.shape
            h2, w2 = h // 2 * 2, w // 2 * 2
            cropped = arr[:h2, :w2, :]
            return (
                cropped.reshape(h2 // 2, 2, w2 // 2, 2, c)
                .mean(axis=(1, 3))
                .astype(arr.dtype)
            )
