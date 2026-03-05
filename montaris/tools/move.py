import numpy as np
from PySide6.QtCore import Qt
from montaris.tools.base import BaseTool
from montaris.core.undo import UndoCommand
from montaris.core.roi_transform import get_mask_bbox, make_translation_matrix, apply_affine_to_mask


class MoveTool(BaseTool):
    name = "Move"

    def __init__(self, app):
        super().__init__(app)
        self.apply_to_all = False
        self._moving = False
        self._start_pos = None
        self._snapshot = None
        self._all_snapshots = None

    def on_press(self, pos, layer, canvas):
        if layer is None or not hasattr(layer, 'mask'):
            return
        bbox = get_mask_bbox(layer.mask)
        if bbox is None:
            return
        self._moving = True
        self._start_pos = pos
        self._snapshot = layer.mask.copy()
        if self.apply_to_all:
            self._all_snapshots = {
                id(roi): roi.mask.copy()
                for roi in canvas.layer_stack.roi_layers
            }

    def on_move(self, pos, layer, canvas):
        if not self._moving or layer is None:
            return
        dx = pos.x() - self._start_pos.x()
        dy = pos.y() - self._start_pos.y()
        M = make_translation_matrix(dx, dy)

        if self.apply_to_all and self._all_snapshots:
            for roi in canvas.layer_stack.roi_layers:
                snap = self._all_snapshots.get(id(roi))
                if snap is not None:
                    roi.mask[:] = apply_affine_to_mask(snap, M, roi.mask.shape)
        else:
            layer.mask[:] = apply_affine_to_mask(self._snapshot, M, layer.mask.shape)

        canvas.refresh_overlays()

    def on_release(self, pos, layer, canvas):
        if not self._moving or layer is None:
            return
        self._moving = False

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
        self._all_snapshots = None
        self._start_pos = None

    def cursor(self):
        return Qt.SizeAllCursor
