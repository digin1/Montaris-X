import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsPixmapItem
from montaris.tools.base import BaseTool
from montaris.core.undo import UndoCommand
from montaris.core.multi_undo import CompoundUndoCommand
from montaris.core.roi_transform import make_translation_matrix, apply_affine_to_mask
from montaris.core.components import get_component_at


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
        self._old_bboxes = {}  # id(layer) -> bbox at press time

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
                layer_bbox = layer.get_bbox()
                comp = get_component_at(layer.mask, ix, iy, bbox=layer_bbox)
                # Only use component mode if there are multiple components
                if comp is not None:
                    by1, by2, bx1, bx2 = layer_bbox
                    crop = layer.mask[by1:by2, bx1:bx2]
                    comp_crop = comp[by1:by2, bx1:bx2]
                    total_pixels = np.count_nonzero(crop)
                    comp_pixels = np.count_nonzero(comp_crop)
                    if comp_pixels < total_pixels:
                        self._component_mask = comp
                        self._target_layers = [layer]
                        self._moving = True
                        self._start_pos = pos
                        self._snapshots = {id(layer): (layer, layer.mask[by1:by2, bx1:bx2].copy(), layer_bbox)}
                        self._old_bboxes = {id(layer): layer_bbox}
                        self._create_previews(canvas)
                        return

        # Whole-layer move
        self._component_mask = None
        self._target_layers = self._get_target_layers(layer, canvas)
        has_content = any(
            l.get_bbox() is not None for l in self._target_layers
        )
        if not has_content:
            return
        self._moving = True
        self._start_pos = pos
        self._snapshots = {}
        self._old_bboxes = {}
        for l in self._target_layers:
            bb = l.get_bbox()
            self._old_bboxes[id(l)] = bb
            if bb is not None:
                self._snapshots[id(l)] = (l, l.mask[bb[0]:bb[1], bb[2]:bb[3]].copy(), bb)
            else:
                self._snapshots[id(l)] = (l, None, None)
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
        canvas.flash_progress("Applying move…")

        # Rasterize the final position
        dx = pos.x() - self._start_pos.x()
        dy = pos.y() - self._start_pos.y()

        if self._component_mask is not None:
            l = self._target_layers[0]
            lid = id(l)
            snap_data, sb = self._snapshots[lid][1], self._snapshots[lid][2]
            old_bb = self._old_bboxes.get(lid)
            by1, by2, bx1, bx2 = old_bb or sb
            # Restore original state from crop snapshot
            l.mask[:] = 0
            if snap_data is not None and sb is not None:
                l.mask[sb[0]:sb[1], sb[2]:sb[3]] = snap_data
            # Erase component, then paste at new position
            comp_crop = self._component_mask[by1:by2, bx1:bx2]
            l.mask[by1:by2, bx1:bx2][comp_crop] = 0
            cys, cxs = np.where(comp_crop)
            ys, xs = cys + by1, cxs + bx1
            new_ys = ys + int(round(dy))
            new_xs = xs + int(round(dx))
            h, w = l.mask.shape
            valid = (new_ys >= 0) & (new_ys < h) & (new_xs >= 0) & (new_xs < w)
            # Get original pixel values from the restored mask region
            l.mask[new_ys[valid], new_xs[valid]] = snap_data[ys[valid] - sb[0], xs[valid] - sb[2]] if snap_data is not None else 255
            l.invalidate_bbox()
        else:
            # Fast numpy shift for translation (avoids Pillow affine overhead)
            idx_dy, idx_dx = int(round(dy)), int(round(dx))
            for lid, (l, snap_data, sb) in self._snapshots.items():
                old_bb = self._old_bboxes.get(lid)
                if old_bb is not None and (idx_dy != 0 or idx_dx != 0):
                    h, w = l.mask.shape
                    l.mask[:] = 0
                    sy1, sy2, sx1, sx2 = old_bb
                    bh, bw = sy2 - sy1, sx2 - sx1
                    # Destination region
                    d_y1, d_y2 = sy1 + idx_dy, sy2 + idx_dy
                    d_x1, d_x2 = sx1 + idx_dx, sx2 + idx_dx
                    # Clip to bounds
                    cy1 = max(0, -d_y1)
                    cy2 = bh - max(0, d_y2 - h)
                    cx1 = max(0, -d_x1)
                    cx2 = bw - max(0, d_x2 - w)
                    if cy2 > cy1 and cx2 > cx1:
                        src = snap_data[cy1:cy2, cx1:cx2] if snap_data is not None else l.mask[sy1+cy1:sy1+cy2, sx1+cx1:sx1+cx2]
                        l.mask[d_y1+cy1:d_y1+cy2, d_x1+cx1:d_x1+cx2] = src
                else:
                    M = make_translation_matrix(dx, dy)
                    # Reconstruct full mask from crop for affine fallback
                    if snap_data is not None and sb is not None and snap_data.shape != l.mask.shape:
                        full = np.zeros_like(l.mask)
                        full[sb[0]:sb[1], sb[2]:sb[3]] = snap_data
                        l.mask[:] = apply_affine_to_mask(full, M, l.mask.shape)
                    else:
                        l.mask[:] = apply_affine_to_mask(snap_data, M, l.mask.shape)
                l.invalidate_bbox()

        commands = []
        for lid, (l, snap_data, sb) in self._snapshots.items():
            # Diff only the union of old + new bounding boxes
            old_bb = self._old_bboxes.get(lid)
            new_bb = l.get_bbox()
            if old_bb is None and new_bb is None:
                continue
            bbs = [b for b in (old_bb, new_bb) if b is not None]
            y1 = min(b[0] for b in bbs)
            y2 = max(b[1] for b in bbs)
            x1 = min(b[2] for b in bbs)
            x2 = max(b[3] for b in bbs)
            # Reconstruct old region from crop snapshot
            if sb is not None and snap_data is not None and snap_data.shape != l.mask.shape:
                sy1, sy2, sx1, sx2 = sb
                region_old = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                oy1, oy2 = max(y1, sy1), min(y2, sy2)
                ox1, ox2 = max(x1, sx1), min(x2, sx2)
                if oy2 > oy1 and ox2 > ox1:
                    region_old[oy1 - y1:oy2 - y1, ox1 - x1:ox2 - x1] = \
                        snap_data[oy1 - sy1:oy2 - sy1, ox1 - sx1:ox2 - sx1]
            elif snap_data is not None:
                region_old = snap_data[y1:y2, x1:x2].copy()
            else:
                region_old = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            region_new = l.mask[y1:y2, x1:x2]
            if not np.array_equal(region_old, region_new):
                cmd = UndoCommand(
                    l, (y1, y2, x1, x2),
                    region_old.copy(),
                    region_new.copy(),
                )
                commands.append(cmd)

        if commands:
            if len(commands) == 1:
                self.app.undo_stack.push(commands[0])
            else:
                self.app.undo_stack.push(CompoundUndoCommand(commands))

        self._snapshots.clear()
        self._old_bboxes = {}
        self._target_layers.clear()
        self._start_pos = None
        self._component_mask = None

        # Re-render only the affected ROI items (mask now updated)
        lod = canvas._current_lod_level()
        for l in affected:
            try:
                idx = canvas.layer_stack.roi_layers.index(l)
                target_lod = 0 if l == canvas._active_layer else lod
                canvas._refresh_roi_item(l, idx, lod_level=target_lod)
                canvas._roi_lod[id(l)] = target_lod
            except ValueError:
                pass
        canvas._update_selection_highlights()

    def _create_previews(self, canvas):
        """Use existing ROI pixmap items as live preview (zero scene changes)."""
        for item in self._preview_items:
            item.setOffset(self._preview_offsets[self._preview_items.index(item)] if self._preview_offsets else item.offset())
        self._preview_items.clear()
        self._preview_offsets = []
        self._hidden_layers = []

        # Hide selection highlights during drag
        for item in canvas._selection_highlight_items:
            item.setVisible(False)

        for lid, (l, snap_data, sb) in self._snapshots.items():
            rid = id(l)
            if rid in canvas._roi_items:
                item = canvas._roi_items[rid]
                self._preview_items.append(item)
                off = item.offset()
                self._preview_offsets.append((off.x(), off.y()))
            self._hidden_layers.append((l, True))

    def _remove_previews(self, canvas, re_render=True):
        """Reset preview offsets. If re_render, also rebuild affected ROI items."""
        for i, item in enumerate(self._preview_items):
            if i < len(self._preview_offsets):
                bx1, by1 = self._preview_offsets[i]
                item.setOffset(bx1, by1)
        self._preview_items.clear()
        if re_render:
            lod = canvas._current_lod_level()
            for l, _ in self._hidden_layers:
                try:
                    idx = canvas.layer_stack.roi_layers.index(l)
                    target_lod = 0 if l == canvas._active_layer else lod
                    canvas._refresh_roi_item(l, idx, lod_level=target_lod)
                    canvas._roi_lod[id(l)] = target_lod
                except ValueError:
                    pass
            canvas._update_selection_highlights()
        self._hidden_layers = []
        self._preview_offsets = []

    def cursor(self):
        return Qt.SizeAllCursor
