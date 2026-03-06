import numpy as np
from PySide6.QtCore import Qt, QPointF
from montaris.tools.base import BaseTool
from montaris.core.undo import UndoCommand
from montaris.core.multi_undo import CompoundUndoCommand


class BrushTool(BaseTool):
    name = "Brush"

    def __init__(self, app):
        super().__init__(app)
        self.size = 100
        self._painting = False
        self._last_pos = None
        self._snapshot = None
        self._other_snapshots = {}  # id(layer) -> (layer, snapshot)
        self._stroke_bbox = None  # (y1, y2, x1, x2) accumulated during stroke

    def on_press(self, pos, layer, canvas):
        if layer is None or not hasattr(layer, 'mask'):
            return
        self._painting = True
        self._last_pos = pos
        self._snapshot = layer.mask.copy()
        self._stroke_bbox = None

        # Snapshot other layers if auto-overlap is on (C.7)
        self._other_snapshots.clear()
        if self.app._auto_overlap:
            for other in self.app.layer_stack.roi_layers:
                if other is not layer and hasattr(other, 'mask'):
                    self._other_snapshots[id(other)] = (other, other.mask.copy())

        self._paint(pos, layer)
        canvas.refresh_active_overlay_partial(layer, self._stroke_bbox)

    def on_move(self, pos, layer, canvas):
        if not self._painting or layer is None:
            return
        h, w = layer.mask.shape
        r = self.size // 2
        lx, ly = int(self._last_pos.x()), int(self._last_pos.y())
        px, py = int(pos.x()), int(pos.y())
        self._paint_line(self._last_pos, pos, layer)
        self._last_pos = pos
        # Dirty bbox for this line segment
        dy1 = max(0, min(ly, py) - r)
        dy2 = min(h, max(ly, py) + r + 1)
        dx1 = max(0, min(lx, px) - r)
        dx2 = min(w, max(lx, px) + r + 1)
        canvas.refresh_active_overlay_partial(layer, (dy1, dy2, dx1, dx2))

    def on_release(self, pos, layer, canvas):
        if not self._painting or layer is None:
            return
        self._painting = False

        commands = []
        # Main layer undo — use accumulated stroke bbox for efficient diff
        if self._snapshot is not None and self._stroke_bbox is not None:
            sy1, sy2, sx1, sx2 = self._stroke_bbox
            old_crop = self._snapshot[sy1:sy2, sx1:sx2]
            new_crop = layer.mask[sy1:sy2, sx1:sx2]
            if not np.array_equal(old_crop, new_crop):
                cmd = UndoCommand(
                    layer, (sy1, sy2, sx1, sx2),
                    old_crop, new_crop,
                )
                commands.append(cmd)

        # Auto-overlap: zero other layers where we painted (C.7)
        sb = self._stroke_bbox
        if self.app._auto_overlap and sb is not None:
            sy1, sy2, sx1, sx2 = sb
            painted = layer.mask[sy1:sy2, sx1:sx2] > 0
            for lid, (other, snap) in self._other_snapshots.items():
                other_crop = other.mask[sy1:sy2, sx1:sx2]
                overlap = painted & (other_crop > 0)
                if overlap.any():
                    other_crop[overlap] = 0
                    snap_crop = snap[sy1:sy2, sx1:sx2]
                    if not np.array_equal(snap_crop, other_crop):
                        cmd = UndoCommand(
                            other, (sy1, sy2, sx1, sx2),
                            snap_crop, other_crop,
                        )
                        commands.append(cmd)
            if len(commands) > 1:
                canvas.refresh_overlays()

        if commands:
            if len(commands) == 1:
                self.app.undo_stack.push(commands[0])
                canvas.refresh_active_overlay(layer)
            else:
                self.app.undo_stack.push(CompoundUndoCommand(commands))
                # refresh_overlays already called above for auto-overlap
        else:
            # No change, but still do a proper refresh for edge rendering
            canvas.refresh_active_overlay(layer)

        self._snapshot = None
        self._stroke_bbox = None
        self._other_snapshots.clear()

    def _paint(self, pos, layer):
        cx, cy = int(pos.x()), int(pos.y())
        r = self.size // 2
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
            if self._stroke_bbox is None:
                self._stroke_bbox = (y1, y2, x1, x2)
            else:
                self._stroke_bbox = (
                    min(self._stroke_bbox[0], y1),
                    max(self._stroke_bbox[1], y2),
                    min(self._stroke_bbox[2], x1),
                    max(self._stroke_bbox[3], x2),
                )

    def _paint_line(self, p1, p2, layer):
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        dist = max(abs(x2 - x1), abs(y2 - y1))
        steps = max(1, int(dist / max(1, self.size // 3)))
        for i in range(steps + 1):
            t = i / max(1, steps)
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            self._paint(QPointF(x, y), layer)

    def cursor(self):
        return Qt.CrossCursor
