import numpy as np
from dataclasses import dataclass, field
from PySide6.QtCore import QObject, Signal

ROI_COLORS = [
    (255, 50, 50),
    (50, 255, 50),
    (50, 100, 255),
    (255, 255, 50),
    (255, 50, 255),
    (50, 255, 255),
    (255, 150, 50),
    (150, 50, 255),
    (50, 255, 150),
    (255, 150, 150),
    (100, 200, 100),
    (200, 100, 200),
    (100, 200, 255),
    (255, 200, 100),
    (180, 80, 80),
    (80, 180, 80),
    (80, 80, 180),
    (200, 200, 100),
    (100, 150, 200),
    (200, 150, 100),
]


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
    def __init__(self, name, width, height, color=None):
        self.name = name
        self.mask = np.zeros((height, width), dtype=np.uint8)
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
        return self.mask.shape

    def invalidate_bbox(self):
        """Mark the cached bounding box as stale."""
        self._bbox_valid = False

    def get_bbox(self):
        """Return bounding box of non-zero mask pixels (cached)."""
        if self._bbox_valid:
            return self._cached_bbox
        from montaris.core.roi_transform import get_mask_bbox
        self._cached_bbox = get_mask_bbox(self.mask)
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
        h, w = self.mask.shape
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

    def add_roi(self, roi):
        if roi.color == ROI_COLORS[0] and self._color_index < len(ROI_COLORS):
            roi.color = ROI_COLORS[self._color_index]
            self._color_index = (self._color_index + 1) % len(ROI_COLORS)
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
        """Duplicate ROI layer."""
        if 0 <= index < len(self.roi_layers):
            src = self.roi_layers[index]
            h, w = src.mask.shape
            new_roi = ROILayer(f"{src.name} (copy)", w, h, src.color)
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
        if roi.color == ROI_COLORS[0] and self._color_index < len(ROI_COLORS):
            roi.color = ROI_COLORS[self._color_index]
            self._color_index = (self._color_index + 1) % len(ROI_COLORS)
        self.roi_layers.insert(index, roi)
        self.changed.emit()


@dataclass
class MontageDocument:
    """Represents a single montage with its image, ROIs, and settings."""
    name: str
    image_layer: ImageLayer
    roi_layers: list = field(default_factory=list)
    adjustments: dict = field(default_factory=lambda: {
        'brightness': 0.0, 'contrast': 1.0, 'exposure': 0.0, 'gamma': 1.0,
    })
    color_index: int = 0
    downsample_factor: int = 1
    original_shape: tuple = None
    tint_color: tuple = None  # (R, G, B) or None for grayscale
