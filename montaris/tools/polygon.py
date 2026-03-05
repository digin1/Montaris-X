import numpy as np
from PySide6.QtCore import Qt
from montaris.tools.base import BaseTool
from montaris.core.undo import UndoCommand


class PolygonTool(BaseTool):
    name = "Polygon"

    def __init__(self, app):
        super().__init__(app)
        self._vertices = []
        self._active_layer = None
        self._canvas = None

    def on_press(self, pos, layer, canvas):
        if layer is None or not hasattr(layer, 'mask'):
            return
        self._active_layer = layer
        self._canvas = canvas
        self._vertices.append((int(pos.x()), int(pos.y())))
        canvas.draw_polygon_preview(self._vertices)

    def on_move(self, pos, layer, canvas):
        if self._vertices:
            canvas.draw_polygon_preview(self._vertices, (int(pos.x()), int(pos.y())))

    def on_release(self, pos, layer, canvas):
        pass

    def on_key_press(self, key, canvas):
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.finish()
        elif key == Qt.Key_Escape:
            self._vertices.clear()
            if canvas:
                canvas.clear_polygon_preview()

    def finish(self):
        if len(self._vertices) < 3 or self._active_layer is None:
            self._vertices.clear()
            if self._canvas:
                self._canvas.clear_polygon_preview()
            return

        layer = self._active_layer
        snapshot = layer.mask.copy()

        self._fill_polygon(layer)

        diff = snapshot != layer.mask
        if diff.any():
            ys, xs = np.where(diff)
            y1, y2 = ys.min(), ys.max() + 1
            x1, x2 = xs.min(), xs.max() + 1
            cmd = UndoCommand(
                layer, (y1, y2, x1, x2),
                snapshot[y1:y2, x1:x2],
                layer.mask[y1:y2, x1:x2],
            )
            self.app.undo_stack.push(cmd)

        if self._canvas:
            self._canvas.clear_polygon_preview()
            self._canvas.refresh_overlays()

        self._vertices.clear()

    def _fill_polygon(self, layer):
        h, w = layer.mask.shape
        vertices = np.array(self._vertices)

        min_x = max(0, vertices[:, 0].min())
        max_x = min(w - 1, vertices[:, 0].max())
        min_y = max(0, vertices[:, 1].min())
        max_y = min(h - 1, vertices[:, 1].max())

        if min_x >= max_x or min_y >= max_y:
            return

        yy, xx = np.mgrid[min_y:max_y + 1, min_x:max_x + 1]
        points_x = xx.ravel()
        points_y = yy.ravel()

        n = len(self._vertices)
        inside = np.zeros(len(points_x), dtype=bool)

        j = n - 1
        for i in range(n):
            xi, yi = self._vertices[i]
            xj, yj = self._vertices[j]

            cond1 = (yi > points_y) != (yj > points_y)
            with np.errstate(divide='ignore', invalid='ignore'):
                slope = (xj - xi) * (points_y - yi) / (yj - yi) + xi
            cond2 = points_x < slope

            inside ^= (cond1 & cond2)
            j = i

        mask_y = points_y[inside]
        mask_x = points_x[inside]
        if len(mask_y) > 0:
            layer.mask[mask_y, mask_x] = 255

    def cursor(self):
        return Qt.CrossCursor
