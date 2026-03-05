import numpy as np
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
]


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

    @property
    def shape(self):
        return self.mask.shape

    def mark_dirty(self, rect):
        """Mark a rectangular region as dirty.

        *rect* is ``(x, y, w, h)``.  Successive calls expand the dirty
        region to the union of all supplied rects.
        """
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


class LayerStack(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.image_layer = None
        self.roi_layers = []
        self._color_index = 0

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
        for idx in indices[1:]:
            roi = self.roi_layers[idx]
            target.mask = np.maximum(target.mask, roi.mask)
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
        self.roi_layers.insert(index, roi)
        self.changed.emit()
