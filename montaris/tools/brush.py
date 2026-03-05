import numpy as np
from PySide6.QtCore import Qt, QPointF
from montaris.tools.base import BaseTool
from montaris.core.undo import UndoCommand
from montaris.core.multi_undo import CompoundUndoCommand


class BrushTool(BaseTool):
    name = "Brush"
    zoom_compensated = True

    def __init__(self, app):
        super().__init__(app)
        self.size = 10
        self._painting = False
        self._last_pos = None
        self._snapshot = None
        self._canvas = None
        self._other_snapshots = {}  # id(layer) -> (layer, snapshot)

    def on_press(self, pos, layer, canvas):
        if layer is None or not hasattr(layer, 'mask'):
            return
        self._painting = True
        self._last_pos = pos
        self._canvas = canvas
        self._snapshot = layer.mask.copy()

        # Snapshot other layers if auto-overlap is on (C.7)
        self._other_snapshots.clear()
        if self.app._auto_overlap:
            for other in self.app.layer_stack.roi_layers:
                if other is not layer and hasattr(other, 'mask'):
                    self._other_snapshots[id(other)] = (other, other.mask.copy())

        self._paint(pos, layer)
        canvas.refresh_active_overlay(layer)

    def on_move(self, pos, layer, canvas):
        if not self._painting or layer is None:
            return
        self._paint_line(self._last_pos, pos, layer)
        self._last_pos = pos
        canvas.refresh_active_overlay(layer)

    def on_release(self, pos, layer, canvas):
        if not self._painting or layer is None:
            return
        self._painting = False

        commands = []
        # Main layer undo
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
                commands.append(cmd)

        # Auto-overlap: zero other layers where we painted (C.7)
        if self.app._auto_overlap and self._snapshot is not None:
            painted = layer.mask > 0
            for lid, (other, snap) in self._other_snapshots.items():
                overlap = painted & (other.mask > 0)
                if overlap.any():
                    other.mask[overlap] = 0
                    diff = snap != other.mask
                    if diff.any():
                        ys, xs = np.where(diff)
                        y1, y2 = ys.min(), ys.max() + 1
                        x1, x2 = xs.min(), xs.max() + 1
                        cmd = UndoCommand(
                            other, (y1, y2, x1, x2),
                            snap[y1:y2, x1:x2],
                            other.mask[y1:y2, x1:x2],
                        )
                        commands.append(cmd)
            if len(commands) > 1:
                canvas.refresh_overlays()

        if commands:
            if len(commands) == 1:
                self.app.undo_stack.push(commands[0])
            else:
                self.app.undo_stack.push(CompoundUndoCommand(commands))

        self._snapshot = None
        self._canvas = None
        self._other_snapshots.clear()

    def _effective_size(self):
        if self._canvas is not None:
            zoom = self._canvas.transform().m11()
            if zoom > 0:
                return max(1, int(self.size / zoom))
        return self.size

    def _paint(self, pos, layer):
        cx, cy = int(pos.x()), int(pos.y())
        effective = self._effective_size()
        r = effective // 2
        h, w = layer.mask.shape

        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        circle = x * x + y * y <= r * r

        y1 = max(0, cy - r)
        y2 = min(h, cy + r + 1)
        x1 = max(0, cx - r)
        x2 = min(w, cx + r + 1)

        cy1 = y1 - (cy - r)
        cy2 = circle.shape[0] - ((cy + r + 1) - y2)
        cx1 = x1 - (cx - r)
        cx2 = circle.shape[1] - ((cx + r + 1) - x2)

        if y1 < y2 and x1 < x2 and cy1 < cy2 and cx1 < cx2:
            layer.mask[y1:y2, x1:x2][circle[cy1:cy2, cx1:cx2]] = 255
            layer.mark_dirty((x1, y1, x2 - x1, y2 - y1))

    def _paint_line(self, p1, p2, layer):
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        dist = max(abs(x2 - x1), abs(y2 - y1))
        effective = self._effective_size()
        steps = max(1, int(dist / max(1, effective // 3)))
        for i in range(steps + 1):
            t = i / max(1, steps)
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            self._paint(QPointF(x, y), layer)

    def cursor(self):
        return Qt.CrossCursor
