import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPen, QColor
from montaris.tools.base import BaseTool
from montaris.core.undo import UndoCommand


class RectangleTool(BaseTool):
    name = "Rectangle"

    def __init__(self, app):
        super().__init__(app)
        self._start = None
        self._snapshot = None
        self._preview_item = None

    def on_press(self, pos, layer, canvas):
        if layer is None or not hasattr(layer, 'mask'):
            return
        self._start = pos
        self._snapshot = layer.mask.copy()

    def on_move(self, pos, layer, canvas):
        if self._start is None:
            return
        self._update_preview(pos, canvas)

    def on_release(self, pos, layer, canvas):
        if self._start is None or layer is None:
            return

        self._clear_preview(canvas)

        x1 = int(min(self._start.x(), pos.x()))
        y1 = int(min(self._start.y(), pos.y()))
        x2 = int(max(self._start.x(), pos.x()))
        y2 = int(max(self._start.y(), pos.y()))

        h, w = layer.mask.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2 + 1)
        y2 = min(h, y2 + 1)

        if x1 < x2 and y1 < y2:
            layer.mask[y1:y2, x1:x2] = 255

            diff = self._snapshot != layer.mask
            if diff.any():
                ys, xs = np.where(diff)
                dy1, dy2 = ys.min(), ys.max() + 1
                dx1, dx2 = xs.min(), xs.max() + 1
                cmd = UndoCommand(
                    layer, (dy1, dy2, dx1, dx2),
                    self._snapshot[dy1:dy2, dx1:dx2],
                    layer.mask[dy1:dy2, dx1:dx2],
                )
                self.app.undo_stack.push(cmd)

        canvas.refresh_overlays()
        self._start = None
        self._snapshot = None

    def _update_preview(self, pos, canvas):
        self._clear_preview(canvas)
        x1 = min(self._start.x(), pos.x())
        y1 = min(self._start.y(), pos.y())
        w = abs(pos.x() - self._start.x())
        h = abs(pos.y() - self._start.y())
        rect = QRectF(x1, y1, w, h)
        pen = QPen(QColor(255, 255, 0), 1.5)
        pen.setCosmetic(True)
        self._preview_item = canvas.scene().addRect(rect, pen)
        self._preview_item.setZValue(1000)

    def _clear_preview(self, canvas):
        if self._preview_item is not None:
            canvas.scene().removeItem(self._preview_item)
            self._preview_item = None

    def cursor(self):
        return Qt.CrossCursor
