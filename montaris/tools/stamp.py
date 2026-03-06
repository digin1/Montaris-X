import numpy as np
from PySide6.QtCore import Qt, QPointF
from montaris.tools.base import BaseTool
from montaris.core.undo import UndoCommand


class StampTool(BaseTool):
    name = "Stamp"

    def __init__(self, app):
        super().__init__(app)
        self.size = 20
        self.width = 20
        self.height = 20
        self._stamping = False
        self._last_pos = None
        self._snapshot = None
        self._stroke_bbox = None

    def on_press(self, pos, layer, canvas):
        if layer is None or not hasattr(layer, 'mask'):
            return
        self._stamping = True
        self._last_pos = pos
        self._snapshot = layer.mask.copy()
        self._stroke_bbox = None
        self._stamp(pos, layer)
        canvas.refresh_active_overlay(layer)

    def on_move(self, pos, layer, canvas):
        if not self._stamping or layer is None:
            return
        self._stamp_line(self._last_pos, pos, layer)
        self._last_pos = pos
        canvas.stamp_on_roi_pixmap(
            layer, int(pos.x()), int(pos.y()),
            self.width // 2, self.height // 2,
        )

    def on_release(self, pos, layer, canvas):
        if not self._stamping or layer is None:
            return
        self._stamping = False
        if self._snapshot is not None and self._stroke_bbox is not None:
            sy1, sy2, sx1, sx2 = self._stroke_bbox
            old_crop = self._snapshot[sy1:sy2, sx1:sx2]
            new_crop = layer.mask[sy1:sy2, sx1:sx2]
            if not np.array_equal(old_crop, new_crop):
                cmd = UndoCommand(
                    layer, (sy1, sy2, sx1, sx2),
                    old_crop, new_crop,
                )
                self.app.undo_stack.push(cmd)
        canvas.refresh_active_overlay(layer)
        self._snapshot = None
        self._stroke_bbox = None

    def _stamp(self, pos, layer):
        cx, cy = int(pos.x()), int(pos.y())
        sw, sh = self.width, self.height
        half_w = sw // 2
        half_h = sh // 2
        mh, mw = layer.mask.shape

        y1 = max(0, cy - half_h)
        y2 = min(mh, cy - half_h + sh)
        x1 = max(0, cx - half_w)
        x2 = min(mw, cx - half_w + sw)

        if y1 < y2 and x1 < x2:
            layer.mask[y1:y2, x1:x2] = 255
            layer.mark_dirty((x1, y1, x2 - x1, y2 - y1))
            if self._stroke_bbox is None:
                self._stroke_bbox = (y1, y2, x1, x2)
            else:
                self._stroke_bbox = (
                    min(self._stroke_bbox[0], y1),
                    max(self._stroke_bbox[1], y2),
                    min(self._stroke_bbox[2], x1),
                    max(self._stroke_bbox[3], x2),
                )

    def _stamp_line(self, p1, p2, layer):
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        dist = max(abs(x2 - x1), abs(y2 - y1))
        step_size = max(1, min(self.width, self.height) // 3)
        steps = max(1, int(dist / step_size))
        for i in range(steps + 1):
            t = i / max(1, steps)
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            self._stamp(QPointF(x, y), layer)

    def cursor(self):
        return Qt.CrossCursor
