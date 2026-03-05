import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem
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
        self._preview_items = []
        self._preview_rgba_buffers = []
        self._hidden_layers = []
        self._preview_offsets = []  # (bx1, by1) per preview item

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
                        self._create_previews(canvas)
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
        self._create_previews(canvas)

    def on_move(self, pos, layer, canvas):
        if not self._moving:
            return
        dx = pos.x() - self._start_pos.x()
        dy = pos.y() - self._start_pos.y()

        # Move preview items — instant, no numpy
        for i, item in enumerate(self._preview_items):
            bx1, by1 = self._preview_offsets[i]
            item.setOffset(bx1 + dx, by1 + dy)

    def on_release(self, pos, layer, canvas):
        if not self._moving:
            return
        self._moving = False

        # Remove preview items (don't re-render yet — mask not updated)
        affected = [l for l, _ in self._hidden_layers]
        self._remove_previews(canvas, re_render=False)

        # Rasterize the final position
        dx = pos.x() - self._start_pos.x()
        dy = pos.y() - self._start_pos.y()

        if self._component_mask is not None:
            l = self._target_layers[0]
            snap = self._snapshots[id(l)][1]
            l.mask[:] = snap.copy()
            l.mask[self._component_mask] = 0
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

        # Re-render only the affected ROI items (mask now updated)
        for l in affected:
            try:
                idx = canvas.layer_stack.roi_layers.index(l)
                canvas._refresh_roi_item(l, idx)
            except ValueError:
                pass

    def _create_previews(self, canvas):
        """Create preview pixmaps for live move preview."""
        self._remove_previews(canvas)
        self._preview_rgba_buffers = []
        self._preview_offsets = []
        self._hidden_layers = []
        scene = canvas.scene()

        for lid, (l, snap) in self._snapshots.items():
            source = self._component_mask if self._component_mask is not None else snap
            bbox = get_mask_bbox(source.astype(np.uint8) if source.dtype == bool else source)
            if bbox is None:
                continue
            by1, by2, bx1, bx2 = bbox

            # Directly remove this ROI's pixmap item from scene
            rid = id(l)
            if rid in canvas._roi_items:
                scene.removeItem(canvas._roi_items.pop(rid))
            self._hidden_layers.append((l, True))

            # Build tight RGBA
            bh, bw = by2 - by1, bx2 - bx1
            r, g, b = l.color
            rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
            if self._component_mask is not None:
                mask_crop = self._component_mask[by1:by2, bx1:bx2]
            else:
                mask_crop = snap[by1:by2, bx1:bx2]
            rgba[mask_crop > 0] = [r, g, b, l.opacity]
            rgba = np.ascontiguousarray(rgba)
            self._preview_rgba_buffers.append(rgba)

            qimg = QImage(rgba.data, bw, bh, bw * 4, QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimg)
            item = QGraphicsPixmapItem(pixmap)
            item.setOffset(bx1, by1)
            item.setZValue(900)
            item.setAcceptedMouseButtons(Qt.NoButton)
            scene.addItem(item)
            self._preview_items.append(item)
            self._preview_offsets.append((bx1, by1))

    def _remove_previews(self, canvas, re_render=True):
        """Remove preview items. If re_render, also rebuild affected ROI items."""
        scene = canvas.scene()
        for item in self._preview_items:
            scene.removeItem(item)
        self._preview_items.clear()
        if re_render:
            for l, _ in self._hidden_layers:
                try:
                    idx = canvas.layer_stack.roi_layers.index(l)
                    canvas._refresh_roi_item(l, idx)
                except ValueError:
                    pass
        self._hidden_layers = []
        self._preview_rgba_buffers = []
        self._preview_offsets = []

    def cursor(self):
        return Qt.SizeAllCursor
