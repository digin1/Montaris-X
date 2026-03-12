"""Numba JIT acceleration for CPU-bound hotspots.

Provides capability detection, JIT-compiled kernels for RLE decode,
mask→RGBA conversion, and edge detection, with numpy fallbacks.
"""
import numpy as np

# ---------------------------------------------------------------------------
# 1a. Capability detection
# ---------------------------------------------------------------------------

HAS_NUMBA = False
HAS_CUDA = False

try:
    from numba import njit, prange  # noqa: F401
    HAS_NUMBA = True
except ImportError:
    pass

if HAS_NUMBA:
    try:
        from numba import cuda
        if cuda.is_available():
            HAS_CUDA = True
    except Exception:
        pass


def _detect_mode():
    if HAS_CUDA:
        return "cuda"
    if HAS_NUMBA:
        return "jit"
    return "numpy"


ACCEL_MODE = _detect_mode()

_enabled = HAS_NUMBA  # auto-enable if available


def is_enabled():
    """Return True if JIT acceleration is currently active."""
    return _enabled and HAS_NUMBA


def set_enabled(value: bool):
    global _enabled
    _enabled = bool(value)


def get_mode():
    """Return current acceleration mode string."""
    if not _enabled or not HAS_NUMBA:
        return "numpy"
    return ACCEL_MODE


# ---------------------------------------------------------------------------
# 1b. RLE decode crop — Numba JIT kernel
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    @njit(cache=True)
    def _rle_decode_crop_numba(values, starts, ends, w, y1, y2, x1, x2):
        """Decode RLE runs directly into a cropped array.

        Scalar loop — no temp array allocations. Sequential because runs
        can overlap in output position.
        """
        crop_h = y2 - y1
        crop_w = x2 - x1
        crop = np.zeros((crop_h, crop_w), dtype=np.uint8)

        flat_y1 = np.int64(y1) * w
        flat_y2 = np.int64(y2) * w

        for i in range(len(values)):
            v = values[i]
            if v == 0:
                continue
            s = starts[i]
            e = ends[i]
            if e <= flat_y1 or s >= flat_y2:
                continue
            # Clamp to row range
            if s < flat_y1:
                s = flat_y1
            if e > flat_y2:
                e = flat_y2
            for pos in range(s, e):
                col = pos % w
                if col < x1 or col >= x2:
                    continue
                row = pos // w
                crop[row - y1, col - x1] = v

        return crop


def _rle_decode_crop_numpy(data, shape, bbox):
    """Numpy fallback for RLE decode crop (existing implementation)."""
    y1, y2, x1, x2 = bbox
    crop_h, crop_w = y2 - y1, x2 - x1
    if not data or crop_h <= 0 or crop_w <= 0:
        return np.zeros((crop_h, crop_w), dtype=np.uint8)

    h, w = shape
    dt = np.dtype([('v', 'u1'), ('n', '<u4')])
    pairs = np.frombuffer(data, dtype=dt)
    values = pairs['v']
    lengths = pairs['n'].astype(np.int64)

    ends = np.cumsum(lengths)
    starts = ends - lengths

    flat_y1 = np.int64(y1) * w
    flat_y2 = np.int64(y2) * w
    keep = (values > 0) & (ends > flat_y1) & (starts < flat_y2)
    if not keep.any():
        return np.zeros((crop_h, crop_w), dtype=np.uint8)

    v_sel = values[keep]
    s_sel = np.maximum(starts[keep], flat_y1)
    e_sel = np.minimum(ends[keep], flat_y2)

    crop = np.zeros((crop_h, crop_w), dtype=np.uint8)

    for v, s, e in zip(v_sel, s_sel, e_sel):
        pos = np.arange(s, e, dtype=np.int64)
        rows = pos // w
        cols = pos % w
        col_ok = (cols >= x1) & (cols < x2)
        if col_ok.any():
            crop[rows[col_ok] - y1, cols[col_ok] - x1] = v

    return crop


def rle_decode_crop(data, shape, bbox):
    """Dispatch RLE decode crop to JIT or numpy fallback."""
    if is_enabled():
        y1, y2, x1, x2 = bbox
        crop_h, crop_w = y2 - y1, x2 - x1
        if not data or crop_h <= 0 or crop_w <= 0:
            return np.zeros((crop_h, crop_w), dtype=np.uint8)

        h, w = shape
        dt = np.dtype([('v', 'u1'), ('n', '<u4')])
        pairs = np.frombuffer(data, dtype=dt)
        values = pairs['v'].astype(np.uint8).copy()
        lengths = pairs['n'].astype(np.int64)

        ends = np.cumsum(lengths)
        starts = ends - lengths

        return _rle_decode_crop_numba(values, starts, ends,
                                      np.int64(w), y1, y2, x1, x2)

    return _rle_decode_crop_numpy(data, shape, bbox)


# ---------------------------------------------------------------------------
# 1c. Mask → RGBA kernels
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    @njit(cache=True)
    def _mask_to_rgba_solid_numba(mask, r, g, b, alpha):
        """Solid fill: write RGBA where mask > 0."""
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for row in range(h):
            for col in range(w):
                if mask[row, col] > 0:
                    rgba[row, col, 0] = r
                    rgba[row, col, 1] = g
                    rgba[row, col, 2] = b
                    rgba[row, col, 3] = alpha
        return rgba

    @njit(cache=True)
    def _mask_to_rgba_outline_numba(mask, r, g, b, alpha):
        """Outline: fused edge detection + RGBA fill in single pass.

        A pixel is an edge if mask > 0 and any 4-connected neighbor
        is 0 or out-of-bounds.
        """
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for row in range(h):
            for col in range(w):
                if mask[row, col] > 0:
                    is_edge = False
                    if row == 0 or mask[row - 1, col] == 0:
                        is_edge = True
                    elif row == h - 1 or mask[row + 1, col] == 0:
                        is_edge = True
                    elif col == 0 or mask[row, col - 1] == 0:
                        is_edge = True
                    elif col == w - 1 or mask[row, col + 1] == 0:
                        is_edge = True
                    if is_edge:
                        rgba[row, col, 0] = r
                        rgba[row, col, 1] = g
                        rgba[row, col, 2] = b
                        rgba[row, col, 3] = alpha
        return rgba

    @njit(cache=True)
    def _mask_to_rgba_both_numba(mask, r, g, b, fill_alpha, edge_alpha):
        """Both fill + outline: fused edge detection + RGBA in single pass."""
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for row in range(h):
            for col in range(w):
                if mask[row, col] > 0:
                    is_edge = False
                    if row == 0 or mask[row - 1, col] == 0:
                        is_edge = True
                    elif row == h - 1 or mask[row + 1, col] == 0:
                        is_edge = True
                    elif col == 0 or mask[row, col - 1] == 0:
                        is_edge = True
                    elif col == w - 1 or mask[row, col + 1] == 0:
                        is_edge = True
                    if is_edge:
                        rgba[row, col, 0] = r
                        rgba[row, col, 1] = g
                        rgba[row, col, 2] = b
                        rgba[row, col, 3] = edge_alpha
                    else:
                        rgba[row, col, 0] = r
                        rgba[row, col, 1] = g
                        rgba[row, col, 2] = b
                        rgba[row, col, 3] = fill_alpha
        return rgba


# ---------------------------------------------------------------------------
# 1d. Standalone edge detection kernel
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    @njit(cache=True)
    def _edge_detect_numba(mask):
        """4-connected edge detection: pixel is edge if >0 and any
        neighbor is 0 or out-of-bounds.

        Equivalent to scipy binary_erosion with cross structuring element.
        """
        h, w = mask.shape
        edge = np.zeros((h, w), dtype=np.uint8)
        for row in range(h):
            for col in range(w):
                if mask[row, col] > 0:
                    is_edge = False
                    if row == 0 or mask[row - 1, col] == 0:
                        is_edge = True
                    elif row == h - 1 or mask[row + 1, col] == 0:
                        is_edge = True
                    elif col == 0 or mask[row, col - 1] == 0:
                        is_edge = True
                    elif col == w - 1 or mask[row, col + 1] == 0:
                        is_edge = True
                    if is_edge:
                        edge[row, col] = 1
        return edge


def _edge_detect_numpy(mask):
    """Numpy/scipy fallback for edge detection."""
    from scipy.ndimage import binary_erosion
    filled = mask > 0
    return filled ^ binary_erosion(filled)


# ---------------------------------------------------------------------------
# 1e. Dispatch functions
# ---------------------------------------------------------------------------

def compute_edge(mask):
    """Compute boolean edge array for mask > 0 pixels.

    Dispatches to JIT kernel or scipy fallback.
    """
    if is_enabled():
        result = _edge_detect_numba(np.ascontiguousarray(mask, dtype=np.uint8))
        return result.astype(bool)
    return _edge_detect_numpy(mask)


def compute_roi_rgba(mask_crop, color, opacity, fill_mode,
                     lod_level, disp_x, disp_y):
    """Compute ROI RGBA array from a pre-cropped mask.

    Thread-safe: operates only on the owned mask_crop copy.
    Returns (rgba_array, width, height, disp_x, disp_y, scale_factor).
    """
    bh, bw = mask_crop.shape
    r, g, b = color
    effective_opacity = int(opacity)

    scale_factor = 1
    if lod_level > 0:
        factor = 1 << lod_level
        th = (bh // factor) * factor
        tw = (bw // factor) * factor
        if th > 0 and tw > 0:
            mask_crop = mask_crop[:th, :tw].reshape(
                th // factor, factor, tw // factor, factor
            ).max(axis=(1, 3))
            bh, bw = th // factor, tw // factor
            scale_factor = factor

    if is_enabled():
        mask_c = np.ascontiguousarray(mask_crop, dtype=np.uint8)
        if fill_mode == 'outline':
            rgba = _mask_to_rgba_outline_numba(
                mask_c, np.uint8(r), np.uint8(g), np.uint8(b),
                np.uint8(effective_opacity))
        elif fill_mode == 'both':
            fill_alpha = max(1, effective_opacity // 2)
            edge_alpha = min(255, effective_opacity)
            rgba = _mask_to_rgba_both_numba(
                mask_c, np.uint8(r), np.uint8(g), np.uint8(b),
                np.uint8(fill_alpha), np.uint8(edge_alpha))
        else:
            rgba = _mask_to_rgba_solid_numba(
                mask_c, np.uint8(r), np.uint8(g), np.uint8(b),
                np.uint8(effective_opacity))
    else:
        # Numpy fallback
        rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
        if fill_mode == 'outline':
            edge = _edge_detect_numpy(mask_crop)
            rgba[edge] = [r, g, b, effective_opacity]
        elif fill_mode == 'both':
            fill_alpha = max(1, effective_opacity // 2)
            painted = mask_crop > 0
            rgba[painted] = [r, g, b, fill_alpha]
            edge = _edge_detect_numpy(mask_crop)
            rgba[edge] = [r, g, b, min(255, effective_opacity)]
        else:
            painted = mask_crop > 0
            rgba[painted] = [r, g, b, effective_opacity]

    rgba = np.ascontiguousarray(rgba)
    return (rgba, bw, bh, disp_x, disp_y, scale_factor)


# ---------------------------------------------------------------------------
# 1f. Warmup — pre-compile all kernels
# ---------------------------------------------------------------------------

def warmup():
    """Pre-compile all JIT kernels with tiny dummy data.

    Call from a background thread at startup. With cache=True, first-ever
    launch compiles ~10-15s; subsequent loads from .nbi/.nbc cache in ms.
    """
    if not HAS_NUMBA:
        return

    dummy_mask = np.array([[0, 1, 1, 0],
                           [1, 1, 1, 1],
                           [1, 1, 1, 1],
                           [0, 1, 1, 0]], dtype=np.uint8)

    # Warm up mask → RGBA kernels
    _mask_to_rgba_solid_numba(dummy_mask, np.uint8(255), np.uint8(0),
                              np.uint8(0), np.uint8(128))
    _mask_to_rgba_outline_numba(dummy_mask, np.uint8(255), np.uint8(0),
                                np.uint8(0), np.uint8(128))
    _mask_to_rgba_both_numba(dummy_mask, np.uint8(255), np.uint8(0),
                             np.uint8(0), np.uint8(64), np.uint8(128))

    # Warm up edge detection
    _edge_detect_numba(dummy_mask)

    # Warm up RLE decode crop
    values = np.array([0, 1, 0], dtype=np.uint8)
    starts = np.array([0, 5, 12], dtype=np.int64)
    ends = np.array([5, 12, 16], dtype=np.int64)
    _rle_decode_crop_numba(values, starts, ends,
                           np.int64(4), 0, 4, 0, 4)
