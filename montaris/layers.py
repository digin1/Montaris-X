import numpy as np
from dataclasses import dataclass, field
from PySide6.QtCore import QObject, Signal
from montaris.core.rle import rle_encode, rle_decode, rle_decode_crop


def _rle_get_bbox(data, shape):
    """Compute bounding box from RLE data without full decompression.

    Returns (y1, y2, x1, x2) or None if all-zero.
    """
    if not data:
        return None
    h, w = shape
    dt = np.dtype([('v', 'u1'), ('n', '<u4')])
    pairs = np.frombuffer(data, dtype=dt)
    values = pairs['v']
    lengths = pairs['n'].astype(np.int64)

    ends = np.cumsum(lengths)
    starts = ends - lengths

    # Keep only non-zero runs
    nz = values > 0
    if not nz.any():
        return None
    s_nz = starts[nz]
    e_nz = ends[nz] - 1  # inclusive end

    # Convert flat positions to row/col
    first_pos = s_nz[0]
    last_pos = e_nz[-1]
    min_row = int(first_pos // w)
    max_row = int(last_pos // w)

    # For column bounds, check all non-zero runs
    min_col = w
    max_col = 0
    for s, e in zip(s_nz, e_nz):
        c1 = int(s % w)
        c2 = int(e % w)
        r1 = int(s // w)
        r2 = int(e // w)
        if r1 == r2:
            # Run within a single row
            min_col = min(min_col, c1)
            max_col = max(max_col, c2)
        else:
            # Run spans multiple rows — full width coverage possible
            min_col = 0
            max_col = w - 1
            break

    if min_col > max_col:
        return None
    return (min_row, max_row + 1, min_col, max_col + 1)


# --- Napari-compatible label colormap (LAB + low-discrepancy sequence) ---

_LABMIN = np.array([0.0, -86.18302974, -107.85730021])
_LABMAX = np.array([100.0, 98.23305386, 94.47812228])
_LABRNG = _LABMAX - _LABMIN


def _low_discrepancy(dim, n, seed=0.5):
    """Quasi-random sequence in [0,1]^dim using generalized golden ratios."""
    phi1 = 1.6180339887498948482
    phi2 = 1.32471795724474602596
    phi3 = 1.22074408460575947536
    seed = np.broadcast_to(seed, (1, dim))
    g = 1.0 / np.array([phi1, phi2, phi3])
    n_arr = np.arange(n).reshape((n, 1))
    return (seed + (n_arr * g[:dim])) % 1


def _lab_to_rgb_batch(lab):
    """Convert LAB array (N,3) to RGB (N,3) float. Out-of-gamut -> negative."""
    # LAB -> XYZ (D65)
    fy = (lab[:, 0] + 16.0) / 116.0
    fx = lab[:, 1] / 500.0 + fy
    fz = fy - lab[:, 2] / 200.0
    delta = 6.0 / 29.0

    def finv(t):
        mask = t > delta
        out = np.where(mask, t ** 3, 3.0 * delta ** 2 * (t - 4.0 / 29.0))
        return out

    X = 0.95047 * finv(fx)
    Y = 1.00000 * finv(fy)
    Z = 1.08883 * finv(fz)
    # XYZ -> linear sRGB
    r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    rgb = np.stack([r, g, b], axis=1)
    # sRGB gamma
    mask = rgb <= 0.0031308
    rgb = np.where(mask, 12.92 * rgb, 1.055 * np.sign(rgb) * np.abs(rgb) ** (1.0 / 2.4) - 0.055)
    return rgb


def _color_random_lab(n, seed=0.5):
    """Generate n valid RGB colors from quasi-random LAB sampling (napari algorithm)."""
    factor = 6
    rgb = np.zeros((0, 3))
    while len(rgb) < n:
        pts = _low_discrepancy(3, n * factor, seed=seed)
        lab = pts * _LABRNG + _LABMIN
        raw_rgb = _lab_to_rgb_batch(lab)
        valid = np.all((raw_rgb > 0) & (raw_rgb < 1), axis=1)
        rgb = np.clip(raw_rgb[valid], 0, 1)
        factor *= 2
    return rgb[:n]


def _low_discrepancy_image(image, seed=0.5, margin=1.0 / 256):
    """Napari's golden-ratio reorder for maximal consecutive color separation."""
    phi_mod = 0.6180339887498948482
    image_float = np.float32(image)
    image_float = seed + image_float * phi_mod
    image_out = margin + (1 - 2 * margin) * (image_float - np.floor(image_float))
    image_out[image == 0] = 0.0
    return image_out


def _build_color_table(n=512, seed=0.5):
    """Build napari-compatible label color table with low-discrepancy reorder."""
    colors_rgb = _color_random_lab(n + 2, seed=seed)
    colors_rgba = np.concatenate((colors_rgb, np.ones((len(colors_rgb), 1))), axis=1)
    values = np.arange(n + 2)
    randomized = _low_discrepancy_image(values, seed=seed)
    control_points = np.concatenate((
        np.array([0]),
        np.linspace(0.00001, 1 - 0.00001, n + 1),
        np.array([1.0]),
    ))
    indices = np.clip(
        np.searchsorted(control_points, randomized, side='right') - 1,
        0, len(control_points) - 1,
    )
    reordered = colors_rgba[indices][:-1]
    # Quantize to uint8 like napari
    rgb8 = (reordered[:, :3] * 255).astype(np.uint8)
    return [(int(rgb8[i, 0]), int(rgb8[i, 1]), int(rgb8[i, 2])) for i in range(len(rgb8))]


_COLOR_TABLE = None


def _generate_color(index):
    """Return a perceptually distinct color for any ROI index (napari-compatible)."""
    global _COLOR_TABLE
    if _COLOR_TABLE is None:
        _COLOR_TABLE = _build_color_table(512)
    if index < len(_COLOR_TABLE):
        return _COLOR_TABLE[index]
    # Fallback beyond table: extend with golden ratio HSV
    import colorsys
    golden = 0.6180339887498948482
    hue = (index * golden) % 1.0
    sat = 0.7 + 0.3 * ((index % 3) / 2.0)
    val = 0.8 + 0.2 * ((index % 2) / 1.0)
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(r * 255), int(g * 255), int(b * 255))


# Keep ROI_COLORS as the first 20 for any code referencing it
ROI_COLORS = [_generate_color(i) for i in range(20)]


def generate_unique_roi_name(base, existing_layers):
    """Generate a unique ROI name by appending a number if needed."""
    names = {l.name for l in existing_layers}
    if base not in names:
        return base
    i = 2
    while f"{base} ({i})" in names:
        i += 1
    return f"{base} ({i})"


class ImageLayer:
    is_roi = False

    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.visible = True
        self._tile_pyramid = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def tile_pyramid(self):
        """Lazily build and return a :class:`TilePyramid` for this image."""
        if self._tile_pyramid is None:
            from montaris.core.tile_pyramid import TilePyramid
            self._tile_pyramid = TilePyramid(self.data)
        return self._tile_pyramid


class ROILayer:
    is_roi = True

    def __init__(self, name, width, height, color=None):
        self.name = name
        self._mask = np.zeros((height, width), dtype=np.uint8)
        self._rle_data = None      # compressed bytes (mutually exclusive with _mask)
        self._mask_shape = (height, width)
        self.color = color or ROI_COLORS[0]
        self.opacity = 128
        self.visible = True
        self.fill_mode = "solid"  # "solid" or "outline"
        self._dirty_rect = None  # (x, y, w, h) or None
        self._cached_bbox = None
        self._bbox_valid = False
        self.offset_x = 0
        self.offset_y = 0

    @property
    def mask(self):
        if self._mask is None:
            self._mask = rle_decode(self._rle_data, self._mask_shape)
            self._rle_data = None
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value
        self._rle_data = None
        if value is not None:
            self._mask_shape = value.shape
        self.invalidate_bbox()

    @property
    def shape(self):
        return self._mask_shape

    def compress(self):
        """Compress mask to RLE, freeing the numpy array."""
        if self._mask is not None:
            self._rle_data, self._mask_shape = rle_encode(self._mask)
            self._mask = None

    @property
    def is_compressed(self):
        return self._rle_data is not None

    def get_mask_crop(self, bbox):
        """Get a mask crop. If compressed, decompress only the bbox region."""
        if self._mask is not None:
            y1, y2, x1, x2 = bbox
            return self._mask[y1:y2, x1:x2]
        return rle_decode_crop(self._rle_data, self._mask_shape, bbox)

    def invalidate_bbox(self):
        """Mark the cached bounding box as stale."""
        self._bbox_valid = False

    def get_bbox(self):
        """Return bounding box of non-zero mask pixels (cached).

        When compressed, computes bbox from RLE data without full
        decompression to avoid memory spikes.
        """
        if self._bbox_valid:
            return self._cached_bbox
        if self._rle_data is not None:
            self._cached_bbox = _rle_get_bbox(self._rle_data, self._mask_shape)
        else:
            from montaris.core.roi_transform import get_mask_bbox
            self._cached_bbox = get_mask_bbox(self._mask)
        # Only cache non-None; empty masks recompute so direct mask
        # assignment (without invalidate_bbox) still works correctly.
        self._bbox_valid = self._cached_bbox is not None
        return self._cached_bbox

    def mark_dirty(self, rect):
        """Mark a rectangular region as dirty.

        *rect* is ``(x, y, w, h)``.  Successive calls expand the dirty
        region to the union of all supplied rects.
        """
        self.invalidate_bbox()
        if self._dirty_rect is None:
            self._dirty_rect = rect
        else:
            ox, oy, ow, oh = self._dirty_rect
            nx, ny, nw, nh = rect
            x1 = min(ox, nx)
            y1 = min(oy, ny)
            x2 = max(ox + ow, nx + nw)
            y2 = max(oy + oh, ny + nh)
            self._dirty_rect = (x1, y1, x2 - x1, y2 - y1)

    def clear_dirty(self):
        """Reset the dirty region."""
        self._dirty_rect = None

    @property
    def dirty_rect(self):
        """Current dirty rectangle or *None*."""
        return self._dirty_rect

    def flatten_offset(self):
        """Bake offset_x/y into the mask, clipping OOB pixels. Resets offset to 0.

        Returns True if flatten happened, False if refused (fully OOB).
        """
        if self.offset_x == 0 and self.offset_y == 0:
            return True
        dx, dy = self.offset_x, self.offset_y
        bbox = self.get_bbox()
        if bbox is None:
            self.offset_x = 0
            self.offset_y = 0
            return True
        y1, y2, x1, x2 = bbox
        h, w = self.mask.shape
        # Destination region in mask coords
        dy1, dy2 = y1 + dy, y2 + dy
        dx1, dx2 = x1 + dx, x2 + dx
        # Refuse flatten when 100% of pixels would be clipped
        if dy1 >= h or dy2 <= 0 or dx1 >= w or dx2 <= 0:
            return False
        crop = self.mask[y1:y2, x1:x2].copy()
        self.mask[:] = 0
        # Clip to mask bounds
        sy1 = max(0, -dy1)
        sy2 = crop.shape[0] - max(0, dy2 - h)
        sx1 = max(0, -dx1)
        sx2 = crop.shape[1] - max(0, dx2 - w)
        if sy2 > sy1 and sx2 > sx1:
            self.mask[dy1 + sy1:dy1 + sy2, dx1 + sx1:dx1 + sx2] = crop[sy1:sy2, sx1:sx2]
        self.offset_x = 0
        self.offset_y = 0
        self.invalidate_bbox()
        return True

    def has_oob_content(self):
        """Return True if offset puts any mask content outside bounds."""
        if self.offset_x == 0 and self.offset_y == 0:
            return False
        bbox = self.get_bbox()
        if bbox is None:
            return False
        y1, y2, x1, x2 = bbox
        h, w = self.shape
        dy1, dy2 = y1 + self.offset_y, y2 + self.offset_y
        dx1, dx2 = x1 + self.offset_x, x2 + self.offset_x
        return dy1 < 0 or dy2 > h or dx1 < 0 or dx2 > w

    def get_display_bbox(self):
        """Return bounding box shifted by offset (canvas coordinates)."""
        bbox = self.get_bbox()
        if bbox is None:
            return None
        y1, y2, x1, x2 = bbox
        return (y1 + self.offset_y, y2 + self.offset_y,
                x1 + self.offset_x, x2 + self.offset_x)


class LayerStack(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.image_layer = None
        self.roi_layers = []
        self._color_index = 0
        self._global_opacity_factor = 1.0

    def set_image(self, layer):
        self.image_layer = layer
        self.roi_layers.clear()
        self._color_index = 0
        self.changed.emit()

    def next_color(self):
        """Return the next distinct color and advance the index."""
        color = _generate_color(self._color_index)
        self._color_index += 1
        return color

    def add_roi(self, roi):
        if roi.color == ROI_COLORS[0]:
            roi.color = self.next_color()
        self.roi_layers.append(roi)
        self.changed.emit()

    def remove_roi(self, index):
        if 0 <= index < len(self.roi_layers):
            self.roi_layers.pop(index)
            self.changed.emit()

    def get_roi(self, index):
        if 0 <= index < len(self.roi_layers):
            return self.roi_layers[index]
        return None

    def merge_rois(self, indices):
        """Merge multiple ROI layers into one. Keep first, remove rest."""
        if len(indices) < 2:
            return
        target = self.roi_layers[indices[0]]
        target.flatten_offset()
        for idx in indices[1:]:
            roi = self.roi_layers[idx]
            roi.flatten_offset()
            target.mask = np.maximum(target.mask, roi.mask)
        target.invalidate_bbox()
        # Remove merged layers in reverse order
        for idx in sorted(indices[1:], reverse=True):
            self.roi_layers.pop(idx)
        self.changed.emit()

    def duplicate_roi(self, index):
        """Duplicate ROI layer with a new distinct color."""
        if 0 <= index < len(self.roi_layers):
            src = self.roi_layers[index]
            h, w = src.mask.shape
            new_roi = ROILayer(f"{src.name} (copy)", w, h, self.next_color())
            new_roi.mask = src.mask.copy()
            new_roi.opacity = src.opacity
            new_roi.fill_mode = src.fill_mode
            new_roi.offset_x = src.offset_x
            new_roi.offset_y = src.offset_y
            self.roi_layers.insert(index + 1, new_roi)
            self.changed.emit()

    def reorder_roi(self, from_idx, to_idx):
        """Move ROI from one position to another."""
        if 0 <= from_idx < len(self.roi_layers) and 0 <= to_idx < len(self.roi_layers):
            roi = self.roi_layers.pop(from_idx)
            self.roi_layers.insert(to_idx, roi)
            self.changed.emit()

    def insert_roi(self, index, roi):
        """Insert ROI at specific position."""
        if roi.color == ROI_COLORS[0]:
            roi.color = self.next_color()
        self.roi_layers.insert(index, roi)
        self.changed.emit()

    def compress_inactive(self, active_layer=None):
        """Compress all ROI layers except the active one."""
        import time
        from montaris.core.event_logger import EventLogger
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        t0 = time.perf_counter()
        targets = [r for r in self.roi_layers if r is not active_layer and r._mask is not None]
        if not targets:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        if len(targets) > 3:
            # Parallel RLE encode — numpy releases GIL
            from montaris.core.workers import get_pool
            from montaris.core.rle import rle_encode
            futures = [(r, get_pool().submit(rle_encode, r._mask)) for r in targets]
            for roi, fut in futures:
                roi._rle_data, roi._mask_shape = fut.result()
                roi._mask = None
        else:
            for roi in targets:
                roi.compress()
        QApplication.restoreOverrideCursor()
        EventLogger.instance().log("compress", "compress_inactive",
            duration_ms=(time.perf_counter() - t0) * 1000, count=len(targets))


@dataclass
class MontageDocument:
    """Represents a single montage with its image, ROIs, and settings."""
    name: str
    image_layer: ImageLayer
    roi_layers: list = field(default_factory=list)
    adjustments: dict = field(default_factory=lambda: {
        'brightness': 0.0, 'contrast': 0.0, 'exposure': 0.0, 'gamma': 1.0,
    })
    color_index: int = 0
    downsample_factor: int = 1
    original_shape: tuple = None
    tint_color: tuple = None  # (R, G, B) or None for grayscale
