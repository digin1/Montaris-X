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
    def shape(self):
        """Return (height, width) without decompressing the mask."""
        return self._mask_shape

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
        h, w = self.shape
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


class VolumeROILayer:
    """A 3D ROI backed by a shared ``labels_3d`` volume on a MontageDocument.

    Duck-types to :class:`ROILayer` so :class:`LayerPanel` can render/rename/
    recolor it without branching. All storage lives in the document's
    ``labels_3d`` array (one integer ID per ROI) and ``labels_meta`` dict
    (display attributes keyed by ID). That keeps memory to one uint16 volume
    for the whole document regardless of how many 3D ROIs exist, and matches
    napari's Labels-layer format for single-file TIFF export.
    """

    is_roi = True
    is_volume = True

    def __init__(self, doc, label_id):
        self._doc = doc
        self._label_id = int(label_id)
        # 3D ROIs don't carry a position offset like 2D ROIs do — the labels
        # volume already defines the voxel coordinates. Keep the attributes
        # present (zero-valued) so code paths reading offset_x/y don't break.
        self.offset_x = 0
        self.offset_y = 0
        self._dirty_rect = None
        self._cached_bbox = None
        self._bbox_valid = False

    @property
    def label_id(self):
        return self._label_id

    def _meta(self):
        return self._doc.labels_meta[self._label_id]

    @property
    def name(self):
        return self._meta()["name"]

    @name.setter
    def name(self, v):
        self._meta()["name"] = v

    @property
    def color(self):
        return self._meta()["color"]

    @color.setter
    def color(self, v):
        self._meta()["color"] = v

    @property
    def opacity(self):
        return self._meta()["opacity"]

    @opacity.setter
    def opacity(self, v):
        self._meta()["opacity"] = int(v)

    @property
    def visible(self):
        return self._meta()["visible"]

    @visible.setter
    def visible(self, v):
        self._meta()["visible"] = bool(v)

    @property
    def fill_mode(self):
        return self._meta()["fill_mode"]

    @fill_mode.setter
    def fill_mode(self, v):
        self._meta()["fill_mode"] = v

    @property
    def shape(self):
        """Return ``(Y, X)`` so 2D compositor code reads a familiar 2D size."""
        lab = self._doc.labels_3d
        if lab is None:
            img = self._doc.image_layer.data if self._doc.image_layer else None
            return img.shape[:2] if img is not None else (0, 0)
        return lab.shape[1:]

    @property
    def volume_shape(self):
        """Return ``(Z, Y, X)`` of the backing labels volume, or ``None``."""
        lab = self._doc.labels_3d
        return None if lab is None else lab.shape

    @property
    def is_compressed(self):
        return False

    @property
    def dirty_rect(self):
        return self._dirty_rect

    def clear_dirty(self):
        self._dirty_rect = None

    def mask_slice(self, z):
        """Return the 2D binary mask for Z-slice ``z`` as uint8 (0/1)."""
        lab = self._doc.labels_3d
        if lab is None or not (0 <= z < lab.shape[0]):
            y, x = self.shape
            return np.zeros((y, x), dtype=np.uint8)
        return (lab[z] == self._label_id).astype(np.uint8)

    @property
    def mask(self):
        """2D mask = current slice's mask. Uses ``_doc.active_z`` if present.

        Exists so code paths written against :class:`ROILayer` (ImageJ export,
        the 2D compositor, bbox queries) see a 2D uint8 mask without caring
        this ROI is actually 3D.
        """
        z = int(getattr(self._doc, 'active_z', 0) or 0)
        return self.mask_slice(z)

    def mask_volume(self):
        """Return the full 3D binary mask ``(Z, Y, X)`` as uint8 (0/1)."""
        lab = self._doc.labels_3d
        if lab is None:
            return None
        return (lab == self._label_id).astype(np.uint8)

    def get_mask_crop(self, bbox):
        y1, y2, x1, x2 = bbox
        return self.mask[y1:y2, x1:x2]

    def invalidate_bbox(self):
        self._bbox_valid = False

    def get_bbox(self):
        """Bbox of this ROI's footprint on the current Z-slice, or None."""
        if self._bbox_valid:
            return self._cached_bbox
        from montaris.core.roi_transform import get_mask_bbox
        self._cached_bbox = get_mask_bbox(self.mask)
        self._bbox_valid = self._cached_bbox is not None
        return self._cached_bbox

    def get_display_bbox(self):
        bbox = self.get_bbox()
        if bbox is None:
            return None
        y1, y2, x1, x2 = bbox
        return (y1 + self.offset_y, y2 + self.offset_y,
                x1 + self.offset_x, x2 + self.offset_x)

    def get_volume_bbox(self):
        """Full 3D bbox ``(z1, z2, y1, y2, x1, x2)`` of this label, or None.

        Used by the 3D nav bar + properties panel so sizing is reported in
        whole-volume units rather than the current-slice 2D bbox.
        """
        lab = self._doc.labels_3d
        if lab is None:
            return None
        mask = (lab == self._label_id)
        if not mask.any():
            return None
        zs = np.where(mask.any(axis=(1, 2)))[0]
        ys = np.where(mask.any(axis=(0, 2)))[0]
        xs = np.where(mask.any(axis=(0, 1)))[0]
        return (int(zs[0]), int(zs[-1]) + 1,
                int(ys[0]), int(ys[-1]) + 1,
                int(xs[0]), int(xs[-1]) + 1)

    def voxel_count(self):
        """Total voxel count for this label across the whole volume."""
        lab = self._doc.labels_3d
        if lab is None:
            return 0
        return int(np.count_nonzero(lab == self._label_id))

    def compress(self):
        """No-op. The shared labels volume is the canonical storage."""
        return

    def flatten_offset(self):
        return True

    def has_oob_content(self):
        return False

    def mark_dirty(self, rect):
        self.invalidate_bbox()
        if self._dirty_rect is None:
            self._dirty_rect = rect
        else:
            ox, oy, ow, oh = self._dirty_rect
            nx, ny, nw, nh = rect
            x1 = min(ox, nx); y1 = min(oy, ny)
            x2 = max(ox + ow, nx + nw); y2 = max(oy + oh, ny + nh)
            self._dirty_rect = (x1, y1, x2 - x1, y2 - y1)


class LayerStack(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.image_layer = None
        self.roi_layers = []
        self._color_index = 0
        self._global_opacity_factor = 1.0
        # Global ROI display settings
        self.fill_mode = 'solid'                     # 'solid', 'boundary', 'both'
        self.boundary_thickness = 1                   # boundary width
        self.boundary_color = (255, 255, 0)           # yellow (all ROIs)
        self.active_boundary_color = (0, 255, 255)    # cyan (selected ROIs)

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
        """Merge multiple ROI layers into one. Keep first, remove rest.

        Rejects mixed 2D+3D selections and 3D ROIs spanning multiple docs
        (returns ``False`` without mutating state so the caller can toast).
        """
        if len(indices) < 2:
            return True
        layers = [self.roi_layers[i] for i in indices]
        vols = [l for l in layers if getattr(l, 'is_volume', False)]
        flats = [l for l in layers if not getattr(l, 'is_volume', False)]
        if vols and flats:
            return False
        if vols:
            doc_ids = {id(l._doc) for l in vols}
            if len(doc_ids) != 1:
                return False
            target = vols[0]
            target_lid = target._label_id
            doc = target._doc
            for roi in vols[1:]:
                other_lid = roi._label_id
                doc.labels_3d[doc.labels_3d == other_lid] = target_lid
                doc.labels_meta.pop(other_lid, None)
            target.invalidate_bbox()
            for idx in sorted(indices[1:], reverse=True):
                self.roi_layers.pop(idx)
            self.changed.emit()
            return True
        target = layers[0]
        target.flatten_offset()
        for roi in layers[1:]:
            roi.flatten_offset()
            target.mask = np.maximum(target.mask, roi.mask)
        target.invalidate_bbox()
        for idx in sorted(indices[1:], reverse=True):
            self.roi_layers.pop(idx)
        self.changed.emit()
        return True

    def duplicate_roi(self, index):
        """Duplicate ROI layer with a new distinct color.

        For 3D ROIs (:class:`VolumeROILayer`), a single labels volume only
        lets one id occupy each voxel — a true voxel-level copy would have
        to overwrite the source. Instead, duplicate reserves an *empty*
        label with the source's display metadata so the user can paint or
        fill into it; the source keeps all its voxels untouched.
        """
        if not (0 <= index < len(self.roi_layers)):
            return
        src = self.roi_layers[index]
        if getattr(src, 'is_volume', False):
            doc = src._doc
            if doc.labels_3d is None:
                return
            new_lid = doc.reserve_label_id(
                name=f"{src.name} (copy)",
                color=self.next_color(),
                opacity=src.opacity,
                visible=True,
                fill_mode=src.fill_mode,
            )
            self.roi_layers.insert(index + 1, VolumeROILayer(doc, new_lid))
            self.changed.emit()
            return
        # Copy compressed data directly if available (avoids full decompression)
        new_roi = ROILayer.__new__(ROILayer)
        new_roi.name = f"{src.name} (copy)"
        if src._rle_data is not None:
            new_roi._mask = None
            new_roi._rle_data = src._rle_data  # bytes are immutable, safe to share
            new_roi._mask_shape = src._mask_shape
        else:
            new_roi._mask = src._mask.copy()
            new_roi._rle_data = None
            new_roi._mask_shape = src._mask_shape
        new_roi.color = self.next_color()
        new_roi.opacity = src.opacity
        new_roi.visible = True
        new_roi.fill_mode = src.fill_mode
        new_roi._dirty_rect = None
        new_roi.offset_x = src.offset_x
        new_roi.offset_y = src.offset_y
        new_roi._cached_bbox = src._cached_bbox
        new_roi._bbox_valid = src._bbox_valid
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
        # VolumeROILayer stores its voxels in the shared labels volume, not
        # in a per-layer RLE buffer, so it has no ``_mask`` attribute and
        # no compression step. Skip it here instead of crashing.
        targets = [
            r for r in self.roi_layers
            if r is not active_layer
            and not getattr(r, 'is_volume', False)
            and getattr(r, '_mask', None) is not None
        ]
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
    original_shape: tuple | None = None
    tint_color: tuple | None = None  # (R, G, B) or None for grayscale
    image_path: str | None = None  # full path to the source image file
    # Raw 3D z-stack (Z, H, W) when imported from a z-stack TIFF; None for 2D images.
    # The image_layer holds the 2D representation used for ROI drawing (max projection or
    # selected slice), while this keeps the full volume for 3D viewing.
    volume_data: "np.ndarray | None" = None
    volume_axes: str | None = None  # e.g. 'ZYX'
    # 3D ROI storage. ``labels_3d`` is a (Z, Y, X) integer array where each
    # positive value is one ROI; 0 is background. Lazy-allocated the first
    # time a 3D ROI is drawn. ``labels_meta`` maps label_id → display dict
    # (name, color, opacity, visible, fill_mode). Matches napari's Labels
    # layer, so the volume can be saved as a single multi-page TIFF.
    labels_3d: "np.ndarray | None" = None
    labels_meta: dict = field(default_factory=dict)
    labels_next_id: int = 1
    # Current Z-slice the user is viewing in 2D. VolumeROILayer reads this
    # to compute its 2D ``mask`` property, so the 2D compositor can render
    # 3D ROIs as per-slice overlays without any extra plumbing.
    active_z: int = 0

    def ensure_labels_3d(self):
        """Lazy-allocate the labels volume the first time a 3D ROI is drawn.

        Shape matches ``volume_data``. Starts as uint8 (up to 255 ROIs); the
        caller must promote to uint16 before assigning IDs ≥ 256.
        """
        if self.labels_3d is not None:
            return self.labels_3d
        if self.volume_data is None:
            raise RuntimeError("cannot allocate labels_3d: document has no volume_data")
        self.labels_3d = np.zeros(self.volume_data.shape, dtype=np.uint8)
        return self.labels_3d

    def promote_labels_dtype(self, target_dtype):
        """Promote ``labels_3d`` to a wider unsigned int dtype in-place."""
        if self.labels_3d is None:
            return
        cur = np.dtype(self.labels_3d.dtype)
        tgt = np.dtype(target_dtype)
        if tgt.itemsize <= cur.itemsize:
            return
        self.labels_3d = self.labels_3d.astype(tgt, copy=False)

    def reserve_label_id(self, name=None, color=None, opacity=128,
                         visible=True, fill_mode="solid"):
        """Allocate a new label ID and register its display metadata.

        Returns the new integer ID. Promotes ``labels_3d`` to uint16 when the
        ID would overflow uint8. Does NOT write into ``labels_3d`` — the
        caller fills voxels after the ID is reserved (fill/paint tools do).
        """
        lid = self.labels_next_id
        self.labels_next_id += 1
        if self.labels_3d is not None and lid > np.iinfo(self.labels_3d.dtype).max:
            self.promote_labels_dtype(np.uint16)
        self.labels_meta[lid] = {
            "name": name or f"3D ROI {lid}",
            "color": color or ROI_COLORS[0],
            "opacity": int(opacity),
            "visible": bool(visible),
            "fill_mode": fill_mode,
        }
        return lid

    def release_label_id(self, lid):
        """Zero out voxels with this label and drop its metadata."""
        if self.labels_3d is not None and lid in self.labels_meta:
            self.labels_3d[self.labels_3d == lid] = 0
        self.labels_meta.pop(lid, None)
