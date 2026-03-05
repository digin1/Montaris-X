import numpy as np
from PySide6.QtCore import Qt, QPointF
from montaris.tools.base import BaseTool
from montaris.core.undo import UndoCommand


class StampTool(BaseTool):
    name = "Stamp"

    def __init__(self, app):
        super().__init__(app)
        self.size = 20
        self._stamping = False
        self._last_pos = None
        self._snapshot = None

    def on_press(self, pos, layer, canvas):
        if layer is None or not hasattr(layer, 'mask'):
            return
        self._stamping = True
        self._last_pos = pos
        self._snapshot = layer.mask.copy()
        self._stamp(pos, layer)
        canvas.refresh_active_overlay(layer)

    def on_move(self, pos, layer, canvas):
        if not self._stamping or layer is None:
            return
        self._stamp_line(self._last_pos, pos, layer)
        self._last_pos = pos
        canvas.refresh_active_overlay(layer)

    def on_release(self, pos, layer, canvas):
        if not self._stamping or layer is None:
            return
        self._stamping = False
        if self._snapshot is not None:
            diff = self._snapshot != layer.mask
            if diff.any():
                ys, xs = np.where(diff)
                y1, y2 = ys.min(), ys.max() + 1
                x1, x2 = xs.min(), xs.max() + 1
                cmd = UndoCommand(
                    layer, (y1, y2, x1, x2),
                    self._snapshot[y1:y2, x1:x2],
                    layer.mask[y1:y2, x1:x2],
                )
                self.app.undo_stack.push(cmd)
        self._snapshot = None

    def _stamp(self, pos, layer):
        cx, cy = int(pos.x()), int(pos.y())
        half = self.size // 2
        h, w = layer.mask.shape

        y1 = max(0, cy - half)
        y2 = min(h, cy - half + self.size)
        x1 = max(0, cx - half)
        x2 = min(w, cx - half + self.size)

        if y1 < y2 and x1 < x2:
            layer.mask[y1:y2, x1:x2] = 255
            layer.mark_dirty((x1, y1, x2 - x1, y2 - y1))

    def _stamp_line(self, p1, p2, layer):
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        dist = max(abs(x2 - x1), abs(y2 - y1))
        steps = max(1, int(dist / max(1, self.size // 3)))
        for i in range(steps + 1):
            t = i / max(1, steps)
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            self._stamp(QPointF(x, y), layer)

    def cursor(self):
        return Qt.CrossCursor
