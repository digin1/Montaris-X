"""Extended tests for montaris.core.accel — dispatch, RGBA shapes, edge cases."""

import numpy as np
import pytest

from montaris.core.accel import (
    is_enabled,
    set_enabled,
    get_mode,
    compute_edge,
    compute_roi_rgba,
    warmup,
    HAS_NUMBA,
    ACCEL_MODE,
    _rle_decode_crop_numpy,
    _edge_detect_numpy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_and_restore_enabled():
    """Return the current enabled state so tests can restore it."""
    return is_enabled()


@pytest.fixture(autouse=True)
def _restore_enabled():
    """Ensure every test restores the original enabled flag."""
    was = is_enabled()
    yield
    set_enabled(was)


def _square_mask(size=20, border=4):
    """A square of ones centered in a field of zeros."""
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[border:size - border, border:size - border] = 1
    return mask


# ---------------------------------------------------------------------------
# 1. is_enabled / set_enabled / get_mode
# ---------------------------------------------------------------------------

class TestCapabilityAPI:
    def test_is_enabled_returns_bool(self):
        assert isinstance(is_enabled(), bool)

    def test_set_enabled_true(self):
        set_enabled(True)
        # is_enabled is True only when numba is available
        assert is_enabled() == HAS_NUMBA

    def test_set_enabled_false(self):
        set_enabled(False)
        assert is_enabled() is False

    def test_set_enabled_toggle(self):
        set_enabled(True)
        first = is_enabled()
        set_enabled(False)
        assert is_enabled() is False
        set_enabled(True)
        assert is_enabled() == first

    def test_get_mode_when_disabled(self):
        set_enabled(False)
        assert get_mode() == "numpy"

    def test_get_mode_when_enabled(self):
        set_enabled(True)
        mode = get_mode()
        assert mode in ("cuda", "jit", "numpy")
        if HAS_NUMBA:
            assert mode == ACCEL_MODE
        else:
            assert mode == "numpy"

    def test_get_mode_returns_string(self):
        assert isinstance(get_mode(), str)


# ---------------------------------------------------------------------------
# 2. compute_edge
# ---------------------------------------------------------------------------

class TestComputeEdge:
    def test_square_edge(self):
        mask = _square_mask(20, 4)
        edge = compute_edge(mask)
        assert edge.dtype == bool
        assert edge.shape == mask.shape
        # Interior pixel should NOT be edge
        assert edge[10, 10] is np.bool_(False)
        # Border pixel of the square should be edge
        assert edge[4, 6] is np.bool_(True)

    def test_single_pixel_is_edge(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 1
        edge = compute_edge(mask)
        assert edge[5, 5]

    def test_empty_mask_no_edges(self):
        mask = np.zeros((15, 15), dtype=np.uint8)
        edge = compute_edge(mask)
        assert not edge.any()

    def test_full_mask_only_border_edges(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        edge = compute_edge(mask)
        # Interior (2..7, 2..7) should have no edges
        assert not edge[2:8, 2:8].any()
        # Border row should be edge
        assert edge[0, :].all()
        assert edge[-1, :].all()
        assert edge[:, 0].all()
        assert edge[:, -1].all()

    def test_edge_numpy_fallback(self):
        """Force numpy path and verify it works."""
        set_enabled(False)
        mask = _square_mask(16, 3)
        edge = compute_edge(mask)
        assert edge.dtype == bool
        assert edge.any()

    def test_edge_consistency_both_paths(self):
        """If numba available, both paths should agree."""
        mask = _square_mask(30, 5)
        set_enabled(False)
        edge_np = compute_edge(mask)
        if HAS_NUMBA:
            set_enabled(True)
            edge_jit = compute_edge(mask)
            np.testing.assert_array_equal(edge_jit, edge_np)


# ---------------------------------------------------------------------------
# 3. compute_roi_rgba
# ---------------------------------------------------------------------------

class TestComputeRoiRGBA:
    """Test RGBA output for different fill modes, LOD, and masks."""

    def _call(self, mask, fill_mode, lod=0, color=(255, 0, 128), opacity=180):
        return compute_roi_rgba(mask.copy(), color, opacity, fill_mode,
                                lod, 10, 20)

    # -- solid ---
    def test_solid_shape(self):
        mask = _square_mask()
        rgba, w, h, dx, dy, sf = self._call(mask, 'solid')
        assert rgba.shape == (20, 20, 4)
        assert rgba.dtype == np.uint8

    def test_solid_fills_masked_pixels(self):
        mask = _square_mask(20, 4)
        rgba, *_ = self._call(mask, 'solid', color=(100, 200, 50), opacity=128)
        # Center pixel should be painted
        assert rgba[10, 10, 0] == 100
        assert rgba[10, 10, 1] == 200
        assert rgba[10, 10, 2] == 50
        assert rgba[10, 10, 3] == 128
        # Outside pixel should be transparent
        assert rgba[0, 0, 3] == 0

    def test_solid_empty_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        rgba, *_ = self._call(mask, 'solid')
        assert rgba.sum() == 0

    def test_solid_full_mask(self):
        mask = np.ones((8, 8), dtype=np.uint8)
        rgba, *_ = self._call(mask, 'solid', opacity=200)
        assert (rgba[:, :, 3] == 200).all()

    # -- outline ---
    def test_outline_shape(self):
        mask = _square_mask()
        rgba, w, h, dx, dy, sf = self._call(mask, 'outline')
        assert rgba.shape == (20, 20, 4)

    def test_outline_only_edges_painted(self):
        mask = _square_mask(20, 4)
        rgba, *_ = self._call(mask, 'outline', opacity=180)
        # Interior should be transparent
        assert rgba[10, 10, 3] == 0
        # Edge pixel should be painted
        assert rgba[4, 6, 3] == 180

    def test_outline_empty_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        rgba, *_ = self._call(mask, 'outline')
        assert rgba.sum() == 0

    # -- both ---
    def test_both_shape(self):
        mask = _square_mask()
        rgba, w, h, *_ = self._call(mask, 'both')
        assert rgba.shape == (20, 20, 4)

    def test_both_fill_and_edge_different_alpha(self):
        mask = _square_mask(20, 4)
        rgba, *_ = self._call(mask, 'both', opacity=200)
        fill_alpha = max(1, 200 // 2)  # 100
        edge_alpha = min(255, 200)     # 200
        # Interior pixel: fill alpha
        assert rgba[10, 10, 3] == fill_alpha
        # Edge pixel: edge alpha
        assert rgba[4, 6, 3] == edge_alpha

    def test_both_empty_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        rgba, *_ = self._call(mask, 'both')
        assert rgba.sum() == 0

    # -- LOD ---
    def test_lod0_no_downscale(self):
        mask = _square_mask(32, 4)
        rgba, w, h, dx, dy, sf = self._call(mask, 'solid', lod=0)
        assert sf == 1
        assert w == 32
        assert h == 32

    def test_lod1_halves_dimensions(self):
        mask = _square_mask(32, 4)
        rgba, w, h, dx, dy, sf = self._call(mask, 'solid', lod=1)
        assert sf == 2
        assert w == 16
        assert h == 16
        assert rgba.shape == (16, 16, 4)

    def test_lod_preserves_disp(self):
        mask = _square_mask(32, 4)
        _, _, _, dx, dy, _ = self._call(mask, 'solid', lod=1)
        assert dx == 10
        assert dy == 20

    # -- Return tuple structure ---
    def test_return_tuple_length(self):
        mask = _square_mask()
        result = self._call(mask, 'solid')
        assert len(result) == 6


# ---------------------------------------------------------------------------
# 4. warmup
# ---------------------------------------------------------------------------

class TestWarmup:
    def test_warmup_runs_without_error(self):
        warmup()

    def test_warmup_idempotent(self):
        warmup()
        warmup()


# ---------------------------------------------------------------------------
# 5. Numba / numpy fallback parity for _rle_decode_crop
# ---------------------------------------------------------------------------

class TestRLEDecodeNumbaVsNumpy:
    """If numba is available, compare _rle_decode_crop_numba to numpy fallback."""

    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
    def test_simple_rect(self):
        from montaris.core.rle import rle_encode
        from montaris.core.accel import _rle_decode_crop_numba

        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 1
        data, shape = rle_encode(mask)
        bbox = (3, 18, 3, 18)

        result_np = _rle_decode_crop_numpy(data, shape, bbox)

        dt = np.dtype([('v', 'u1'), ('n', '<u4')])
        pairs = np.frombuffer(data, dtype=dt)
        values = pairs['v'].astype(np.uint8).copy()
        lengths = pairs['n'].astype(np.int64)
        ends = np.cumsum(lengths)
        starts = ends - lengths
        result_numba = _rle_decode_crop_numba(
            values, starts, ends, np.int64(shape[1]),
            bbox[0], bbox[1], bbox[2], bbox[3]
        )
        np.testing.assert_array_equal(result_numba, result_np)

    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
    def test_empty_data(self):
        from montaris.core.accel import _rle_decode_crop_numba

        values = np.array([], dtype=np.uint8)
        starts = np.array([], dtype=np.int64)
        ends = np.array([], dtype=np.int64)
        result = _rle_decode_crop_numba(values, starts, ends,
                                        np.int64(10), 0, 5, 0, 5)
        assert result.shape == (5, 5)
        assert result.sum() == 0

    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
    def test_full_mask(self):
        from montaris.core.rle import rle_encode
        from montaris.core.accel import _rle_decode_crop_numba

        mask = np.ones((10, 10), dtype=np.uint8)
        data, shape = rle_encode(mask)
        bbox = (0, 10, 0, 10)

        result_np = _rle_decode_crop_numpy(data, shape, bbox)

        dt = np.dtype([('v', 'u1'), ('n', '<u4')])
        pairs = np.frombuffer(data, dtype=dt)
        values = pairs['v'].astype(np.uint8).copy()
        lengths = pairs['n'].astype(np.int64)
        ends = np.cumsum(lengths)
        starts = ends - lengths
        result_numba = _rle_decode_crop_numba(
            values, starts, ends, np.int64(10), 0, 10, 0, 10
        )
        np.testing.assert_array_equal(result_numba, result_np)


# ---------------------------------------------------------------------------
# 6. RGBA shape correctness across fill modes
# ---------------------------------------------------------------------------

class TestRGBAShapeCorrectness:
    """Verify RGBA output always has shape (H, W, 4) for various inputs."""

    @pytest.mark.parametrize("fill_mode", ["solid", "outline", "both"])
    def test_shape_is_hw4(self, fill_mode):
        mask = np.zeros((25, 30), dtype=np.uint8)
        mask[5:20, 5:25] = 1
        rgba, w, h, *_ = compute_roi_rgba(
            mask.copy(), (10, 20, 30), 150, fill_mode, 0, 0, 0)
        assert rgba.ndim == 3
        assert rgba.shape[2] == 4
        assert rgba.shape == (25, 30, 4)

    @pytest.mark.parametrize("fill_mode", ["solid", "outline", "both"])
    def test_contiguous(self, fill_mode):
        mask = _square_mask()
        rgba, *_ = compute_roi_rgba(
            mask.copy(), (1, 2, 3), 128, fill_mode, 0, 0, 0)
        assert rgba.flags['C_CONTIGUOUS']


# ---------------------------------------------------------------------------
# 7. Edge detection consistency between numba and scipy
# ---------------------------------------------------------------------------

class TestEdgeConsistency:
    """Verify edge detection produces identical results regardless of backend."""

    def _compare_edges(self, mask):
        set_enabled(False)
        edge_np = compute_edge(mask.copy())
        if HAS_NUMBA:
            set_enabled(True)
            edge_jit = compute_edge(mask.copy())
            np.testing.assert_array_equal(edge_jit, edge_np)

    def test_rect_edge_consistency(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[8:32, 8:32] = 1
        self._compare_edges(mask)

    def test_circle_edge_consistency(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        yy, xx = np.ogrid[:50, :50]
        mask[((yy - 25) ** 2 + (xx - 25) ** 2) <= 15 ** 2] = 1
        self._compare_edges(mask)

    def test_l_shape_edge_consistency(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[5:25, 5:15] = 1
        mask[15:25, 5:25] = 1
        self._compare_edges(mask)
