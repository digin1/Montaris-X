"""Tests for montaris.core.accel — JIT acceleration correctness and fallback."""
import numpy as np
import pytest

from montaris.core.rle import rle_encode, rle_decode_crop
from montaris.core.accel import (
    _rle_decode_crop_numpy,
    _edge_detect_numpy,
    is_enabled,
    set_enabled,
    HAS_NUMBA,
    get_mode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rect_mask(h, w, y1, y2, x1, x2):
    """Create a mask with a filled rectangle."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def _make_circle_mask(h, w, cy, cx, r):
    """Create a mask with a filled circle."""
    mask = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    mask[circle] = 1
    return mask


# ---------------------------------------------------------------------------
# RLE decode crop correctness
# ---------------------------------------------------------------------------

class TestRLEDecodeCrop:
    """Compare JIT rle_decode_crop output to numpy version."""

    def _compare(self, mask, bbox):
        data, shape = rle_encode(mask)
        # Numpy baseline
        result_np = _rle_decode_crop_numpy(data, shape, bbox)
        # Dispatch (uses JIT if available and enabled)
        result_dispatch = rle_decode_crop(data, shape, bbox)
        np.testing.assert_array_equal(result_dispatch, result_np)

    def test_basic_rect(self):
        mask = _make_rect_mask(100, 100, 20, 60, 30, 70)
        self._compare(mask, (10, 70, 20, 80))

    def test_full_bbox(self):
        mask = _make_rect_mask(50, 50, 5, 45, 5, 45)
        self._compare(mask, (0, 50, 0, 50))

    def test_no_overlap(self):
        mask = _make_rect_mask(100, 100, 20, 40, 20, 40)
        self._compare(mask, (60, 90, 60, 90))

    def test_partial_overlap(self):
        mask = _make_rect_mask(100, 100, 10, 50, 10, 50)
        self._compare(mask, (30, 70, 30, 70))

    def test_circle_mask(self):
        mask = _make_circle_mask(200, 200, 100, 100, 50)
        self._compare(mask, (60, 150, 60, 150))

    def test_single_pixel(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 1
        self._compare(mask, (4, 7, 4, 7))

    def test_empty_data(self):
        result = rle_decode_crop(b'', (100, 100), (0, 50, 0, 50))
        assert result.shape == (50, 50)
        assert result.sum() == 0

    def test_large_mask(self):
        mask = _make_rect_mask(500, 500, 100, 400, 100, 400)
        self._compare(mask, (50, 450, 50, 450))


# ---------------------------------------------------------------------------
# Edge detection correctness
# ---------------------------------------------------------------------------

class TestEdgeDetection:
    """Compare JIT edge detection to scipy binary_erosion."""

    def test_rect_edge(self):
        mask = _make_rect_mask(50, 50, 10, 40, 10, 40)
        edge_np = _edge_detect_numpy(mask)

        if HAS_NUMBA and is_enabled():
            from montaris.core.accel import _edge_detect_numba
            edge_jit = _edge_detect_numba(
                np.ascontiguousarray(mask, dtype=np.uint8))
            np.testing.assert_array_equal(edge_jit.astype(bool), edge_np)

    def test_circle_edge(self):
        mask = _make_circle_mask(100, 100, 50, 50, 30)
        edge_np = _edge_detect_numpy(mask)

        if HAS_NUMBA and is_enabled():
            from montaris.core.accel import _edge_detect_numba
            edge_jit = _edge_detect_numba(
                np.ascontiguousarray(mask, dtype=np.uint8))
            np.testing.assert_array_equal(edge_jit.astype(bool), edge_np)

    def test_single_pixel_edge(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 1
        edge_np = _edge_detect_numpy(mask)
        # Single pixel should be entirely edge
        assert edge_np[5, 5]

        if HAS_NUMBA and is_enabled():
            from montaris.core.accel import _edge_detect_numba
            edge_jit = _edge_detect_numba(
                np.ascontiguousarray(mask, dtype=np.uint8))
            assert edge_jit[5, 5] == 1

    def test_empty_mask(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        edge_np = _edge_detect_numpy(mask)
        assert not edge_np.any()


# ---------------------------------------------------------------------------
# RGBA output correctness
# ---------------------------------------------------------------------------

class TestRGBA:
    """Compare JIT RGBA output for all fill modes."""

    def _compare_rgba(self, mask, fill_mode):
        from montaris.core.accel import compute_roi_rgba

        color = (255, 0, 128)
        opacity = 180

        # Force numpy
        set_enabled(False)
        result_np = compute_roi_rgba(mask.copy(), color, opacity, fill_mode,
                                     0, 0, 0)

        # Restore JIT if available
        if HAS_NUMBA:
            set_enabled(True)
            result_jit = compute_roi_rgba(mask.copy(), color, opacity,
                                          fill_mode, 0, 0, 0)
            np.testing.assert_array_equal(result_jit[0], result_np[0])

    def test_solid(self):
        mask = _make_rect_mask(50, 50, 10, 40, 10, 40)
        self._compare_rgba(mask, 'solid')

    def test_outline(self):
        mask = _make_rect_mask(50, 50, 10, 40, 10, 40)
        self._compare_rgba(mask, 'outline')

    def test_both(self):
        mask = _make_rect_mask(50, 50, 10, 40, 10, 40)
        self._compare_rgba(mask, 'both')

    def test_circle_solid(self):
        mask = _make_circle_mask(80, 80, 40, 40, 25)
        self._compare_rgba(mask, 'solid')

    def test_circle_outline(self):
        mask = _make_circle_mask(80, 80, 40, 40, 25)
        self._compare_rgba(mask, 'outline')

    def test_circle_both(self):
        mask = _make_circle_mask(80, 80, 40, 40, 25)
        self._compare_rgba(mask, 'both')

    def test_lod_level(self):
        """RGBA with LOD downsampling should work identically."""
        from montaris.core.accel import compute_roi_rgba
        mask = _make_rect_mask(64, 64, 8, 56, 8, 56)

        set_enabled(False)
        result_np = compute_roi_rgba(mask.copy(), (100, 200, 50), 128,
                                     'solid', 1, 0, 0)
        if HAS_NUMBA:
            set_enabled(True)
            result_jit = compute_roi_rgba(mask.copy(), (100, 200, 50), 128,
                                          'solid', 1, 0, 0)
            np.testing.assert_array_equal(result_jit[0], result_np[0])


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

class TestFallback:
    """Test that the module works when Numba is not available."""

    def test_numpy_mode_works(self):
        """Disabling JIT should still produce correct output."""
        was_enabled = is_enabled()
        set_enabled(False)
        try:
            mask = _make_rect_mask(30, 30, 5, 25, 5, 25)
            data, shape = rle_encode(mask)
            result = rle_decode_crop(data, shape, (0, 30, 0, 30))
            np.testing.assert_array_equal(result, mask)

            assert get_mode() == "numpy"
        finally:
            set_enabled(was_enabled)

    def test_toggle_enabled(self):
        """set_enabled / is_enabled round-trip."""
        was_enabled = is_enabled()
        set_enabled(True)
        assert is_enabled() == HAS_NUMBA  # only True if numba available
        set_enabled(False)
        assert not is_enabled()
        set_enabled(was_enabled)


# ---------------------------------------------------------------------------
# Benchmark (optional — run with pytest -k benchmark)
# ---------------------------------------------------------------------------

class TestBenchmark:
    """Performance benchmarks for JIT vs numpy."""

    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
    def test_rle_decode_speedup(self):
        """JIT rle_decode_crop should be >= 5x faster than numpy."""
        import time

        # Create a 2000x2000 mask with 50 rectangles
        mask = np.zeros((2000, 2000), dtype=np.uint8)
        rng = np.random.RandomState(42)
        for _ in range(50):
            y1, x1 = rng.randint(0, 1800, size=2)
            y2 = min(y1 + rng.randint(20, 200), 2000)
            x2 = min(x1 + rng.randint(20, 200), 2000)
            mask[y1:y2, x1:x2] = 1

        data, shape = rle_encode(mask)
        bbox = (200, 1800, 200, 1800)

        # Warm up JIT
        set_enabled(True)
        rle_decode_crop(data, shape, bbox)

        # Benchmark JIT
        n_iter = 5
        t0 = time.perf_counter()
        for _ in range(n_iter):
            rle_decode_crop(data, shape, bbox)
        jit_time = (time.perf_counter() - t0) / n_iter

        # Benchmark numpy
        set_enabled(False)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            rle_decode_crop(data, shape, bbox)
        np_time = (time.perf_counter() - t0) / n_iter

        set_enabled(True)
        speedup = np_time / jit_time if jit_time > 0 else float('inf')
        print(f"\nRLE decode: numpy={np_time*1000:.1f}ms, "
              f"jit={jit_time*1000:.1f}ms, speedup={speedup:.1f}x")
        assert speedup >= 5, f"Expected >= 5x speedup, got {speedup:.1f}x"

    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
    def test_rgba_speedup(self):
        """JIT RGBA should be >= 3x faster than numpy."""
        import time
        from montaris.core.accel import compute_roi_rgba

        mask = _make_circle_mask(2000, 2000, 1000, 1000, 800)
        color = (255, 0, 128)
        opacity = 180

        # Warm up JIT
        set_enabled(True)
        compute_roi_rgba(mask.copy(), color, opacity, 'both', 0, 0, 0)

        n_iter = 3
        t0 = time.perf_counter()
        for _ in range(n_iter):
            compute_roi_rgba(mask.copy(), color, opacity, 'both', 0, 0, 0)
        jit_time = (time.perf_counter() - t0) / n_iter

        set_enabled(False)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            compute_roi_rgba(mask.copy(), color, opacity, 'both', 0, 0, 0)
        np_time = (time.perf_counter() - t0) / n_iter

        set_enabled(True)
        speedup = np_time / jit_time if jit_time > 0 else float('inf')
        print(f"\nRGBA (both): numpy={np_time*1000:.1f}ms, "
              f"jit={jit_time*1000:.1f}ms, speedup={speedup:.1f}x")
        assert speedup >= 3, f"Expected >= 3x speedup, got {speedup:.1f}x"
