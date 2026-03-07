import math
import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import Qt
from montaris.tools.base import BaseTool
from montaris.core.undo import UndoCommand

CLOSE_DISTANCE = 10  # pixels (screen space) to snap-close polygon


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
        px, py = int(pos.x()), int(pos.y())

        # If 4+ vertices and clicking near the start point, close the polygon
        if len(self._vertices) >= 4:
            sx, sy = self._vertices[0]
            scale = canvas.transform().m11() or 1.0
            screen_dist = math.hypot(px - sx, py - sy) * scale
            if screen_dist <= CLOSE_DISTANCE:
                self.finish()
                return

        self._vertices.append((px, py))
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
        h, w = layer.mask.shape
        vertices = np.array(self._vertices)

        # Known bbox from vertex extremes
        bx1 = max(0, int(vertices[:, 0].min()))
        bx2 = min(w, int(vertices[:, 0].max()) + 1)
        by1 = max(0, int(vertices[:, 1].min()))
        by2 = min(h, int(vertices[:, 1].max()) + 1)

        if bx1 < bx2 and by1 < by2:
            old_crop = layer.mask[by1:by2, bx1:bx2].copy()

            self._fill_polygon(layer, bx1, by1, bx2, by2)
            layer.invalidate_bbox()

            new_crop = layer.mask[by1:by2, bx1:bx2]
            if not np.array_equal(old_crop, new_crop):
                cmd = UndoCommand(
                    layer, (by1, by2, bx1, bx2),
                    old_crop, new_crop,
                )
                self.app.undo_stack.push(cmd)

        if self._canvas:
            self._canvas.clear_polygon_preview()
            self._canvas.refresh_active_overlay(layer)

        self._vertices.clear()

    def _fill_polygon(self, layer, bx1, by1, bx2, by2):
        bw, bh = bx2 - bx1, by2 - by1
        # Shift vertices to local bbox coordinates
        local_verts = [(x - bx1, y - by1) for x, y in self._vertices]
        img = Image.new('L', (bw, bh), 0)
        ImageDraw.Draw(img).polygon(local_verts, fill=255)
        poly_mask = np.asarray(img)
        layer.mask[by1:by2, bx1:bx2][poly_mask > 0] = 255

    def cursor(self):
        return Qt.CrossCursor
