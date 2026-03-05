import collections

import numpy as np
from PySide6.QtCore import Qt
from montaris.tools.base import BaseTool
from montaris.core.undo import UndoCommand


class BucketFillTool(BaseTool):
    name = "Bucket Fill"

    def __init__(self, app):
        super().__init__(app)
        self.tolerance = 0  # 0 = exact match, higher = more fill

    def on_press(self, pos, layer, canvas):
        if layer is None or not hasattr(layer, 'mask'):
            return

        x, y = int(pos.x()), int(pos.y())
        h, w = layer.mask.shape

        if x < 0 or x >= w or y < 0 or y >= h:
            return

        snapshot = layer.mask.copy()
        target_value = layer.mask[y, x]

        # If pixel is painted, erase the connected component; otherwise paint it
        if target_value > 0:
            fill_value = 0
        else:
            fill_value = 255

        self._flood_fill(layer.mask, x, y, fill_value, target_value)

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

        canvas.refresh_overlays()

    def _flood_fill(self, mask, start_x, start_y, fill_value, target_value):
        h, w = mask.shape
        if fill_value == target_value:
            return

        visited = np.zeros((h, w), dtype=bool)
        queue = collections.deque()
        queue.append((start_x, start_y))
        visited[start_y, start_x] = True

        while queue:
            cx, cy = queue.popleft()
            mask[cy, cx] = fill_value

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    if abs(int(mask[ny, nx]) - int(target_value)) <= self.tolerance:
                        visited[ny, nx] = True
                        queue.append((nx, ny))

    def cursor(self):
        return Qt.CrossCursor
