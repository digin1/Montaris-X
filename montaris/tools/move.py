import numpy as np
from PySide6.QtCore import Qt, QRectF, QLineF
from PySide6.QtGui import QPen, QColor, QImage, QPixmap
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
        self._component_mask = None  # Combined mask of selected components
        self._component_bbox = None  # union bbox of selected components
        self._preview_items = []
        self._old_bboxes = {}  # id(layer) -> bbox at press time

        self._hidden_layers = []
        self._preview_offsets = []  # (bx1, by1) per preview item
        self._temp_scene_items = []  # temporary items (remainder pixmap, bbox)
        self._bbox_item = None
        # Persistent multi-component selection (survives between drags)
        self._multi_comp_mask = None  # combined bool mask of Ctrl+clicked components
        self._multi_comp_layer = None  # the layer these components belong to
        self._multi_bbox_items = []  # visual bbox indicators for selection

    def _get_target_layers(self, layer, canvas):
        """Return selected layers if in selection, else [layer]."""
        sel = canvas._selection.layers
        if sel and layer in sel:
            return sel
        return [layer]

    def on_press(self, pos, layer, canvas):
        if layer is None or not hasattr(layer, 'mask'):
            return

        from PySide6.QtWidgets import QApplication
        ctrl = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)

        ix, iy = int(pos.x()), int(pos.y())

        # Check if clicking on a painted pixel for component-aware move
        sel = canvas._selection.layers
        multi_layer = sel and layer in sel and len(sel) > 1

        if not multi_layer and hasattr(layer, 'mask'):
            h, w = layer.mask.shape
            if 0 <= iy < h and 0 <= ix < w and layer.mask[iy, ix] > 0:
                layer_bbox = layer.get_bbox()
                comp = get_component_at(layer.mask, ix, iy, bbox=layer_bbox)
                if comp is not None:
                    by1, by2, bx1, bx2 = layer_bbox
                    crop = layer.mask[by1:by2, bx1:bx2]
                    comp_crop = comp[by1:by2, bx1:bx2]
                    total_pixels = np.count_nonzero(crop)
                    comp_pixels = np.count_nonzero(comp_crop)
                    has_multiple = comp_pixels < total_pixels

                    if has_multiple and ctrl:
                        # Ctrl+click: toggle component in multi-selection
                        self._toggle_component(comp, layer, canvas)
                        return

                    if has_multiple:
                        # Check if clicking on a pixel in the multi-selection
                        if (self._multi_comp_mask is not None
                                and self._multi_comp_layer is layer
                                and self._multi_comp_mask[iy, ix]):
                            # Move all selected components
                            self._component_mask = self._multi_comp_mask
                            self._component_bbox = self._multi_comp_bbox()
                        else:
                            # Single component move — clear multi-selection
                            self._clear_multi_selection(canvas)
                            cys, cxs = np.where(comp_crop)
                            comp_bb = (int(cys.min()) + by1, int(cys.max()) + 1 + by1,
                                       int(cxs.min()) + bx1, int(cxs.max()) + 1 + bx1)
                            self._component_mask = comp
                            self._component_bbox = comp_bb

                        self._target_layers = [layer]
                        self._moving = True
                        self._start_pos = pos
                        self._snapshots = {id(layer): (layer, layer.mask[by1:by2, bx1:bx2].copy(), layer_bbox)}
                        self._old_bboxes = {id(layer): layer_bbox}
                        self._create_previews(canvas)
                        return

            # Clicked on empty space or non-component pixel — clear multi-selection
            if self._multi_comp_mask is not None:
                self._clear_multi_selection(canvas)

        # Whole-layer move
        self._component_mask = None
        self._component_bbox = None
        self._clear_multi_selection(canvas)
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

        # Move bbox indicator
        if self._bbox_item and self._component_bbox:
            cy1, cy2, cx1, cx2 = self._component_bbox
            self._bbox_item.setRect(QRectF(cx1 + dx, cy1 + dy, cx2 - cx1, cy2 - cy1))

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

        # Update component selection mask to new positions and promote
        # single-component drags to a persistent multi-selection
        comp_mask_used = self._component_mask
        moved_layer = self._target_layers[0] if self._target_layers else None
        if comp_mask_used is not None and moved_layer is not None:
            idx_dy, idx_dx = int(round(dy)), int(round(dx))
            old_mask = comp_mask_used
            new_mask = np.zeros_like(old_mask)
            ys, xs = np.where(old_mask)
            nys, nxs = ys + idx_dy, xs + idx_dx
            mh, mw = new_mask.shape
            valid = (nys >= 0) & (nys < mh) & (nxs >= 0) & (nxs < mw)
            new_mask[nys[valid], nxs[valid]] = True
            self._multi_comp_mask = new_mask
            self._multi_comp_layer = moved_layer

        self._snapshots.clear()
        self._old_bboxes = {}
        self._target_layers.clear()
        self._start_pos = None
        self._component_mask = None
        self._component_bbox = None

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

        # Show persistent bbox around moved component(s)
        if self._multi_comp_mask is not None:
            self._show_multi_selection_bbox(canvas)

    def _create_previews(self, canvas):
        """Use existing ROI pixmap items as live preview (zero scene changes)."""
        for item in self._preview_items:
            item.setOffset(self._preview_offsets[self._preview_items.index(item)] if self._preview_offsets else item.offset())
        self._preview_items.clear()
        self._preview_offsets = []
        self._hidden_layers = []

        # Hide selection highlights and multi-selection bbox during drag
        for item in canvas._selection_highlight_items:
            item.setVisible(False)
        for item in self._multi_bbox_items:
            item.setVisible(False)

        if self._component_mask is not None:
            self._create_component_previews(canvas)
            return

        for lid, (l, snap_data, sb) in self._snapshots.items():
            rid = id(l)
            if rid in canvas._roi_items:
                item = canvas._roi_items[rid]
                self._preview_items.append(item)
                off = item.offset()
                self._preview_offsets.append((off.x(), off.y()))
            self._hidden_layers.append((l, True))

    def _create_component_previews(self, canvas):
        """Split ROI into component (draggable) and remainder (static) pixmaps."""
        l = self._target_layers[0]
        lid = id(l)
        rid = id(l)
        comp = self._component_mask
        scene = canvas.scene()

        # Hide the real ROI item
        real_item = canvas._roi_items.get(rid)
        z = real_item.zValue() if real_item else 10
        if real_item:
            real_item.setVisible(False)
        self._hidden_layers.append((l, True))

        r, g, b = l.color
        gof = canvas.layer_stack._global_opacity_factor
        eff = int(l.opacity * gof)

        layer_bb = l.get_bbox()
        if layer_bb is None:
            return
        ly1, ly2, lx1, lx2 = layer_bb
        lh, lw = ly2 - ly1, lx2 - lx1
        mask_crop = l.mask[ly1:ly2, lx1:lx2]
        comp_crop = comp[ly1:ly2, lx1:lx2]

        current_comp = (mask_crop > 0) & comp_crop
        non_comp = (mask_crop > 0) & ~comp_crop

        # Component-only pixmap (moves with drag)
        cys, cxs = np.where(current_comp)
        if len(cys) > 0:
            cy1 = int(cys.min()) + ly1
            cy2 = int(cys.max()) + 1 + ly1
            cx1 = int(cxs.min()) + lx1
            cx2 = int(cxs.max()) + 1 + lx1
            ch, cw = cy2 - cy1, cx2 - cx1
            comp_region = current_comp[cy1 - ly1:cy2 - ly1, cx1 - lx1:cx2 - lx1]
            rgba = np.zeros((ch, cw, 4), dtype=np.uint8)
            rgba[comp_region] = [r, g, b, eff]
            rgba = np.ascontiguousarray(rgba)
            qimg = QImage(rgba.data, cw, ch, cw * 4, QImage.Format_RGBA8888)
            comp_item = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
            comp_item.setOffset(cx1, cy1)
            comp_item.setZValue(z)
            comp_item.setAcceptedMouseButtons(Qt.NoButton)
            scene.addItem(comp_item)
            self._preview_items.append(comp_item)
            self._preview_offsets.append((cx1, cy1))
            self._temp_scene_items.append(comp_item)

        # Remainder pixmap (static)
        if np.any(non_comp):
            rem_rgba = np.zeros((lh, lw, 4), dtype=np.uint8)
            rem_rgba[non_comp] = [r, g, b, eff]
            rem_rgba = np.ascontiguousarray(rem_rgba)
            qimg2 = QImage(rem_rgba.data, lw, lh, lw * 4, QImage.Format_RGBA8888)
            rem_item = QGraphicsPixmapItem(QPixmap.fromImage(qimg2))
            rem_item.setOffset(lx1, ly1)
            rem_item.setZValue(z - 1)
            rem_item.setAcceptedMouseButtons(Qt.NoButton)
            scene.addItem(rem_item)
            self._temp_scene_items.append(rem_item)

        # Bbox indicator around the component
        if self._component_bbox:
            cy1, cy2, cx1, cx2 = self._component_bbox
            pen = QPen(QColor(0, 180, 255), 1.5)
            pen.setCosmetic(True)
            pen.setStyle(Qt.DashLine)
            self._bbox_item = scene.addRect(
                QRectF(cx1, cy1, cx2 - cx1, cy2 - cy1), pen
            )
            self._bbox_item.setZValue(999)
            self._temp_scene_items.append(self._bbox_item)

    def _remove_previews(self, canvas, re_render=True):
        """Reset preview offsets. If re_render, also rebuild affected ROI items."""
        # Remove temporary scene items (component mode)
        scene = canvas.scene()
        for item in self._temp_scene_items:
            scene.removeItem(item)
        self._temp_scene_items.clear()
        self._bbox_item = None
        # Reset non-temp preview items
        for i, item in enumerate(self._preview_items):
            if item.scene() and i < len(self._preview_offsets):
                bx1, by1 = self._preview_offsets[i]
                item.setOffset(bx1, by1)
        self._preview_items.clear()
        # Restore visibility of hidden real items
        for l, _ in self._hidden_layers:
            rid = id(l)
            if rid in canvas._roi_items:
                canvas._roi_items[rid].setVisible(True)
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

    def _toggle_component(self, comp, layer, canvas):
        """Ctrl+click: add or remove a component from multi-selection."""
        if self._multi_comp_layer is not layer:
            # Switching layers — start fresh
            self._clear_multi_selection(canvas)
            self._multi_comp_mask = comp.copy()
            self._multi_comp_layer = layer
        else:
            # Check if this component overlaps existing selection
            overlap = self._multi_comp_mask & comp
            if np.any(overlap):
                # Remove component from selection
                self._multi_comp_mask[comp] = False
                if not np.any(self._multi_comp_mask):
                    self._clear_multi_selection(canvas)
                    return
            else:
                # Add component to selection
                self._multi_comp_mask |= comp
        self._show_multi_selection_bbox(canvas)

    def _multi_comp_bbox(self):
        """Compute union bbox of multi-component selection."""
        if self._multi_comp_mask is None:
            return None
        from montaris.core.roi_transform import get_mask_bbox
        return get_mask_bbox(self._multi_comp_mask.view(np.uint8))

    def _show_multi_selection_bbox(self, canvas):
        """Show/update the dashed bbox around multi-selected components."""
        self._clear_multi_bbox_items(canvas)
        bb = self._multi_comp_bbox()
        if bb is None:
            return
        cy1, cy2, cx1, cx2 = bb
        scene = canvas.scene()
        pen = QPen(QColor(0, 180, 255), 1.5)
        pen.setCosmetic(True)
        pen.setStyle(Qt.DashLine)
        item = scene.addRect(QRectF(cx1, cy1, cx2 - cx1, cy2 - cy1), pen)
        item.setZValue(999)
        self._multi_bbox_items.append(item)

    def _clear_multi_bbox_items(self, canvas):
        """Remove visual multi-selection bbox indicators."""
        scene = canvas.scene()
        for item in self._multi_bbox_items:
            scene.removeItem(item)
        self._multi_bbox_items.clear()

    def _clear_multi_selection(self, canvas):
        """Clear multi-component selection and its visuals."""
        self._clear_multi_bbox_items(canvas)
        self._multi_comp_mask = None
        self._multi_comp_layer = None

    def _clear_handles(self, canvas):
        """Called by canvas on tool/layer switch to clean up state."""
        self._clear_multi_selection(canvas)

    def cursor(self):
        return Qt.SizeAllCursor
