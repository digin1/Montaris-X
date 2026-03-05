import numpy as np


class UndoCommand:
    def __init__(self, roi_layer, bbox, old_data, new_data):
        self.roi_layer = roi_layer
        self.bbox = bbox  # (y1, y2, x1, x2)
        self.old_data = old_data.copy()
        self.new_data = new_data.copy()

    def undo(self):
        y1, y2, x1, x2 = self.bbox
        self.roi_layer.mask[y1:y2, x1:x2] = self.old_data
        self.roi_layer.invalidate_bbox()

    def redo(self):
        y1, y2, x1, x2 = self.bbox
        self.roi_layer.mask[y1:y2, x1:x2] = self.new_data
        self.roi_layer.invalidate_bbox()


class UndoStack:
    def __init__(self, max_size=100):
        self._stack = []
        self._index = -1
        self._max_size = max_size

    def push(self, command):
        self._stack = self._stack[:self._index + 1]
        self._stack.append(command)
        if len(self._stack) > self._max_size:
            self._stack.pop(0)
        else:
            self._index += 1

    def undo(self):
        if self._index >= 0:
            cmd = self._stack[self._index]
            cmd.undo()
            self._index -= 1
            return cmd
        return None

    def redo(self):
        if self._index < len(self._stack) - 1:
            self._index += 1
            cmd = self._stack[self._index]
            cmd.redo()
            return cmd
        return None

    def clear(self):
        self._stack.clear()
        self._index = -1

    @property
    def can_undo(self):
        return self._index >= 0

    @property
    def can_redo(self):
        return self._index < len(self._stack) - 1
