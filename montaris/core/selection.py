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
        roi_layers = [l for l in all_layers if hasattr(l, 'mask')]
        if roi_layers != self._layers:
            self._layers = roi_layers
            self.changed.emit(self._layers)

    def contains(self, layer):
        return layer in self._layers

    @staticmethod
    def hit_test(x, y, roi_layers):
        """Search roi_layers in reverse z-order, return first whose mask[y,x] > 0."""
        ix, iy = int(x), int(y)
        for layer in reversed(roi_layers):
            if not hasattr(layer, 'mask'):
                continue
            mask = layer.mask
            if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]:
                if mask[iy, ix] > 0:
                    return layer
        return None
