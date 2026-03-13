import numpy as np
from PySide6.QtCore import QObject, Signal


class SelectionModel(QObject):
    """Holds an ordered list of selected ROI layers. Emits changed(list) on mutation."""

    changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layers = []

    @property
    def layers(self):
        return list(self._layers)

    @property
    def primary(self):
        """First selected layer, or None."""
        return self._layers[0] if self._layers else None

    @property
    def count(self):
        return len(self._layers)

    def set(self, layers):
        if layers == self._layers:
            return
        self._layers = list(layers)
        self.changed.emit(self._layers)

    def add(self, layer):
        if layer not in self._layers:
            self._layers.append(layer)
            self.changed.emit(self._layers)

    def remove(self, layer):
        if layer in self._layers:
            self._layers.remove(layer)
            self.changed.emit(self._layers)

    def toggle(self, layer):
        if layer in self._layers:
            self._layers.remove(layer)
        else:
            self._layers.append(layer)
        self.changed.emit(self._layers)

    def clear(self):
        if self._layers:
            self._layers.clear()
            self.changed.emit(self._layers)

    def select_all(self, all_layers):
        roi_layers = [l for l in all_layers if getattr(l, 'is_roi', False)]
        if roi_layers != self._layers:
            self._layers = roi_layers
            self.changed.emit(self._layers)

    def select_all_silent(self, all_layers):
        """Set selection to all ROI layers WITHOUT emitting changed signal.

        Caller is responsible for triggering deferred UI sync.
        """
        roi_layers = [l for l in all_layers if getattr(l, 'is_roi', False)]
        self._layers = roi_layers

    def contains(self, layer):
        return layer in self._layers

    @staticmethod
    def hit_test(x, y, roi_layers):
        """Search roi_layers in reverse z-order, return first whose mask[y,x] > 0.

        Uses cached bbox for fast rejection and avoids full decompression
        of compressed (RLE) masks by reading only the needed pixel.
        """
        ix, iy = int(x), int(y)
        for layer in reversed(roi_layers):
            if not getattr(layer, 'is_roi', False) or not layer.visible:
                continue
            mx = ix - getattr(layer, 'offset_x', 0)
            my = iy - getattr(layer, 'offset_y', 0)
            # Fast bbox rejection — avoids decompression entirely
            bbox = layer.get_bbox()
            if bbox is not None:
                y1, y2, x1, x2 = bbox
                if not (y1 <= my < y2 and x1 <= mx < x2):
                    continue
                # Point is inside bbox — check the actual pixel
                # Use get_mask_crop for a 1x1 region (cheap even for compressed)
                pixel = layer.get_mask_crop((my, my + 1, mx, mx + 1))
                if pixel[0, 0] > 0:
                    return layer
            else:
                # No bbox (empty mask) — skip
                continue
        return None
