import math

import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPen, QColor
from montaris.tools.base import BaseTool
from montaris.core.undo import UndoCommand


class CircleTool(BaseTool):
    name = "Circle"

    def __init__(self, app):
        super().__init__(app)
        self._center = None
        self._snapshot = None
        self._preview_item = None

    def on_press(self, pos, layer, canvas):
        if layer is None or not getattr(layer, 'is_roi', False):
            return
        self._center = pos
        self._snapshot = layer.mask.copy()

    def on_move(self, pos, layer, canvas):
        if self._center is None:
            return
        self._update_preview(pos, canvas)

    def on_release(self, pos, layer, canvas):
        if self._center is None or layer is None:
            return

        self._clear_preview(canvas)

        cx = int(self._center.x())
        cy = int(self._center.y())
        radius = math.sqrt(
            (pos.x() - self._center.x()) ** 2
            + (pos.y() - self._center.y()) ** 2
        )
        radius = int(radius)

        if radius < 1:
            self._center = None
            self._snapshot = None
            return

        h, w = layer.mask.shape
        # Compute bbox of the circle and clamp to mask bounds
        by1 = max(0, cy - radius)
        by2 = min(h, cy + radius + 1)
        bx1 = max(0, cx - radius)
        bx2 = min(w, cx + radius + 1)

        if by1 < by2 and bx1 < bx2:
            y, x = np.ogrid[by1:by2, bx1:bx2]
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            circle_mask = dist_sq <= radius * radius

            old_crop = self._snapshot[by1:by2, bx1:bx2].copy()
            layer.mask[by1:by2, bx1:bx2][circle_mask] = 255
            new_crop = layer.mask[by1:by2, bx1:bx2]

            if not np.array_equal(old_crop, new_crop):
                cmd = UndoCommand(
                    layer, (by1, by2, bx1, bx2),
                    old_crop, new_crop,
                )
                self.app.undo_stack.push(cmd)

        canvas.refresh_active_overlay(layer)
        self._center = None
        self._snapshot = None

    def _update_preview(self, pos, canvas):
        self._clear_preview(canvas)
        radius = math.sqrt(
            (pos.x() - self._center.x()) ** 2
            + (pos.y() - self._center.y()) ** 2
        )
        rect = QRectF(
            self._center.x() - radius,
            self._center.y() - radius,
            radius * 2,
            radius * 2,
        )
        pen = QPen(QColor(255, 255, 0), 1.5)
        pen.setCosmetic(True)
        self._preview_item = canvas.scene().addEllipse(rect, pen)
        self._preview_item.setZValue(1000)

    def _clear_preview(self, canvas):
        if self._preview_item is not None:
            canvas.scene().removeItem(self._preview_item)
            self._preview_item = None

    def cursor(self):
        return Qt.CrossCursor
