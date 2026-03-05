import numpy as np
from PySide6.QtCore import Qt
from montaris.tools.base import BaseTool
from montaris.core.undo import UndoCommand
from montaris.core.multi_undo import CompoundUndoCommand
from montaris.core.roi_transform import get_mask_bbox, make_translation_matrix, apply_affine_to_mask


class MoveTool(BaseTool):
    name = "Move"

    def __init__(self, app):
        super().__init__(app)
        self._moving = False
        self._start_pos = None
        self._snapshots = {}  # id(layer) -> (layer, snapshot)
        self._target_layers = []
        self._component_mask = None  # If moving a single component

    def _get_target_layers(self, layer, canvas):
        """Return selected layers if in selection, else [layer]."""
        sel = canvas._selection.layers
        if sel and layer in sel:
            return sel
        return [layer]

    def on_press(self, pos, layer, canvas):
        if layer is None or not hasattr(layer, 'mask'):
            return

        ix, iy = int(pos.x()), int(pos.y())

        # Check if clicking on a painted pixel for component-aware move (D.13)
        sel = canvas._selection.layers
        multi = sel and layer in sel and len(sel) > 1

        if not multi and hasattr(layer, 'mask'):
            h, w = layer.mask.shape
            if 0 <= iy < h and 0 <= ix < w and layer.mask[iy, ix] > 0:
                from montaris.core.components import get_component_at
                comp = get_component_at(layer.mask, ix, iy)
                # Only use component mode if there are multiple components
                if comp is not None:
                    total_pixels = np.count_nonzero(layer.mask)
                    comp_pixels = np.count_nonzero(comp)
                    if comp_pixels < total_pixels:
                        self._component_mask = comp
                        self._target_layers = [layer]
                        self._moving = True
                        self._start_pos = pos
                        self._snapshots = {id(layer): (layer, layer.mask.copy())}
                        return

        # Whole-layer move
        self._component_mask = None
        self._target_layers = self._get_target_layers(layer, canvas)
        has_content = any(
            get_mask_bbox(l.mask) is not None for l in self._target_layers
        )
        if not has_content:
            return
        self._moving = True
        self._start_pos = pos
        self._snapshots = {
            id(l): (l, l.mask.copy()) for l in self._target_layers
        }

    def on_move(self, pos, layer, canvas):
        if not self._moving:
            return
        dx = pos.x() - self._start_pos.x()
        dy = pos.y() - self._start_pos.y()

        if self._component_mask is not None:
            # Component-aware move: shift only component pixels
            l = self._target_layers[0]
            snap = self._snapshots[id(l)][1]
            l.mask[:] = snap.copy()
            # Zero component at original position
            l.mask[self._component_mask] = 0
            # Place at new position
            ys, xs = np.where(self._component_mask)
            new_ys = ys + int(round(dy))
            new_xs = xs + int(round(dx))
            h, w = l.mask.shape
            valid = (new_ys >= 0) & (new_ys < h) & (new_xs >= 0) & (new_xs < w)
            l.mask[new_ys[valid], new_xs[valid]] = snap[ys[valid], xs[valid]]
        else:
            M = make_translation_matrix(dx, dy)
            for lid, (l, snap) in self._snapshots.items():
                l.mask[:] = apply_affine_to_mask(snap, M, l.mask.shape)

        canvas.refresh_overlays()
        canvas._update_selection_highlights()

    def on_release(self, pos, layer, canvas):
        if not self._moving:
            return
        self._moving = False

        commands = []
        for lid, (l, snap) in self._snapshots.items():
            diff = snap != l.mask
            if diff.any():
                ys, xs = np.where(diff)
                y1, y2 = ys.min(), ys.max() + 1
                x1, x2 = xs.min(), xs.max() + 1
                cmd = UndoCommand(
                    l, (y1, y2, x1, x2),
                    snap[y1:y2, x1:x2],
                    l.mask[y1:y2, x1:x2],
                )
                commands.append(cmd)

        if commands:
            if len(commands) == 1:
                self.app.undo_stack.push(commands[0])
            else:
                self.app.undo_stack.push(CompoundUndoCommand(commands))

        self._snapshots.clear()
        self._target_layers.clear()
        self._start_pos = None
        self._component_mask = None

    def cursor(self):
        return Qt.SizeAllCursor
