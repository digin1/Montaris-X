import numpy as np
from PySide6.QtCore import Qt, QRectF, QTimer
from PySide6.QtGui import QPen, QColor, QImage, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem
from montaris.tools.base import BaseTool
from montaris.core.undo import UndoCommand, OffsetUndoCommand
from montaris.core.multi_undo import CompoundUndoCommand
from montaris.core.components import get_component_at


class MoveTool(BaseTool):
    name = "Move"

    def __init__(self, app):
        super().__init__(app)
        self._moving = False
        self._start_pos = None
        self._target_layers = []
        self._component_mask = None
        self._component_bbox = None
        # Component move state
        self._comp_snapshots = {}  # id(layer) -> (layer, snapshot, bbox)
        self._comp_preview_items = []
        self._comp_preview_offsets = []
        self._comp_temp_items = []
        self._comp_hidden_layers = []
        self._comp_bbox_item = None
        self._old_bboxes = {}
        # Persistent multi-component selection
        self._multi_comp_mask = None
        self._multi_comp_layer = None
        # Whole-layer offset move state
        self._old_offsets = {}  # id(layer) -> (offset_x, offset_y)
        # Marching ants overlay
        self._ants_entries = []
        self._ants_timer = None
        self._ants_phase = 0

    def on_activate(self, layer, canvas):
        """Called when the move tool is selected."""
        sel = canvas._selection.layers
        layers = sel if sel else ([layer] if layer is not None else [])
        layers = [l for l in layers if hasattr(l, 'mask') and l.get_bbox() is not None]
        if layers:
            self._show_marching_ants_multi(layers, canvas)
        else:
            self._clear_marching_ants(canvas)

    def _get_target_layers(self, layer, canvas):
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

        sel = canvas._selection.layers
        multi_layer = sel and layer in sel and len(sel) > 1

        if not multi_layer and hasattr(layer, 'mask'):
            # Convert canvas coords to mask coords for component detection
            mx = ix - layer.offset_x
            my = iy - layer.offset_y
            h, w = layer.mask.shape
            if 0 <= my < h and 0 <= mx < w and layer.mask[my, mx] > 0:
                layer_bbox = layer.get_bbox()
                comp = get_component_at(layer.mask, mx, my, bbox=layer_bbox)
                if comp is not None:
                    by1, by2, bx1, bx2 = layer_bbox
                    crop = layer.mask[by1:by2, bx1:bx2]
                    comp_crop = comp[by1:by2, bx1:bx2]
                    total_pixels = np.count_nonzero(crop)
                    comp_pixels = np.count_nonzero(comp_crop)
                    has_multiple = comp_pixels < total_pixels

                    if has_multiple and ctrl:
                        self._toggle_component(comp, layer, canvas)
                        return

                    if has_multiple:
                        if (self._multi_comp_mask is not None
                                and self._multi_comp_layer is layer
                                and self._multi_comp_mask[my, mx]):
                            self._component_mask = self._multi_comp_mask
                            self._component_bbox = self._multi_comp_bbox()
                        else:
                            self._clear_multi_selection(canvas)
                            cys, cxs = np.where(comp_crop)
                            comp_bb = (int(cys.min()) + by1, int(cys.max()) + 1 + by1,
                                       int(cxs.min()) + bx1, int(cxs.max()) + 1 + bx1)
                            self._component_mask = comp
                            self._component_bbox = comp_bb

                        # Flatten offset before component move (mask-based)
                        if not layer.flatten_offset():
                            # Layer is fully OOB — can't do component move
                            return
                        canvas._refresh_roi_item(layer, canvas.layer_stack.roi_layers.index(layer))

                        layer_bbox = layer.get_bbox()
                        by1, by2, bx1, bx2 = layer_bbox
                        self._target_layers = [layer]
                        self._moving = True
                        self._start_pos = pos
                        self._comp_snapshots = {id(layer): (layer, layer.mask[by1:by2, bx1:bx2].copy(), layer_bbox)}
                        self._old_bboxes = {id(layer): layer_bbox}
                        self._create_component_previews(canvas)
                        return

            if self._multi_comp_mask is not None:
                self._clear_multi_selection(canvas)

        # Whole-layer offset move
        self._component_mask = None
        self._component_bbox = None
        self._clear_multi_selection(canvas)
        self._target_layers = self._get_target_layers(layer, canvas)
        has_content = any(l.get_bbox() is not None for l in self._target_layers)
        if not has_content:
            return
        self._moving = True
        self._start_pos = pos
        self._old_offsets = {id(l): (l.offset_x, l.offset_y) for l in self._target_layers}

        # Hide selection highlights and ants during drag
        for item in canvas._selection_highlight_items:
            item.setVisible(False)
        self._clear_marching_ants(canvas)

    def on_move(self, pos, layer, canvas):
        if not self._moving:
            return
        dx = pos.x() - self._start_pos.x()
        dy = pos.y() - self._start_pos.y()

        if self._component_mask is not None:
            # Component move: free preview, clamped on release
            for i, item in enumerate(self._comp_preview_items):
                bx1, by1 = self._comp_preview_offsets[i]
                item.setOffset(bx1 + dx, by1 + dy)
            if self._comp_bbox_item and self._component_bbox:
                cy1, cy2, cx1, cx2 = self._component_bbox
                self._comp_bbox_item.setRect(QRectF(cx1 + dx, cy1 + dy, cx2 - cx1, cy2 - cy1))
        else:
            # Whole-layer: update offsets and reposition real scene items
            idx_dx, idx_dy = int(round(dx)), int(round(dy))
            for l in self._target_layers:
                lid = id(l)
                old_ox, old_oy = self._old_offsets[lid]
                l.offset_x = old_ox + idx_dx
                l.offset_y = old_oy + idx_dy
                # Reposition the real scene item directly
                rid = id(l)
                real_item = canvas._roi_items.get(rid)
                if real_item:
                    bbox = l.get_bbox()
                    if bbox is not None:
                        y1, y2, x1, x2 = bbox
                        real_item.setOffset(x1 + l.offset_x, y1 + l.offset_y)

        self._expand_scene_rect_for_pos(pos, canvas)
        self._auto_scroll(pos, canvas)
        if self._target_layers:
            canvas.ensureVisible(QRectF(pos.x() - 10, pos.y() - 10, 20, 20), 50, 50)

    def on_release(self, pos, layer, canvas):
        if not self._moving:
            return
        self._moving = False

        dx = pos.x() - self._start_pos.x()
        dy = pos.y() - self._start_pos.y()

        if self._component_mask is not None:
            self._release_component_move(pos, layer, canvas, dx, dy)
        else:
            self._release_offset_move(canvas)

    def _release_offset_move(self, canvas):
        """Finalize whole-layer offset move: push undo commands."""
        commands = []
        for l in self._target_layers:
            lid = id(l)
            old_off = self._old_offsets.get(lid)
            new_off = (l.offset_x, l.offset_y)
            if old_off != new_off:
                commands.append(OffsetUndoCommand(l, old_off, new_off))
        if commands:
            if len(commands) == 1:
                self.app.undo_stack.push(commands[0])
            else:
                self.app.undo_stack.push(CompoundUndoCommand(commands))

        # Refresh ROI pixmap items so they match the new offset
        lod = canvas._current_lod_level()
        for l in self._target_layers:
            try:
                idx = canvas.layer_stack.roi_layers.index(l)
                target_lod = 0 if l == canvas._active_layer else lod
                canvas._refresh_roi_item(l, idx, lod_level=target_lod)
                canvas._roi_lod[id(l)] = target_lod
            except ValueError:
                pass

        # Restore selection highlights
        canvas._update_selection_highlights()

        # Show marching ants
        visible = [l for l in self._target_layers
                   if hasattr(l, 'mask') and l.get_bbox() is not None]
        if visible:
            self._show_marching_ants_multi(visible, canvas)

        self._old_offsets.clear()
        self._target_layers.clear()
        self._start_pos = None

    def _release_component_move(self, pos, layer, canvas, dx, dy):
        """Finalize component move: modify mask data + undo."""
        # Clamp delta to keep component in-bounds (no pixel loss)
        dx, dy = self._clamp_component_delta(dx, dy)

        affected = [l for l, _ in self._comp_hidden_layers]
        self._remove_component_previews(canvas, re_render=False)
        canvas.flash_progress("Applying move...")

        l = self._target_layers[0]
        lid = id(l)
        snap_data, sb = self._comp_snapshots[lid][1], self._comp_snapshots[lid][2]
        old_bb = self._old_bboxes.get(lid)
        by1, by2, bx1, bx2 = old_bb or sb

        comp_crop = self._component_mask[by1:by2, bx1:bx2]
        cys, cxs = np.where(comp_crop)
        ys, xs = cys + by1, cxs + bx1
        new_ys = ys + int(round(dy))
        new_xs = xs + int(round(dx))

        # Restore original state from crop snapshot
        l.mask[:] = 0
        if snap_data is not None and sb is not None:
            l.mask[sb[0]:sb[1], sb[2]:sb[3]] = snap_data
        # Erase component, then paste at new position
        l.mask[by1:by2, bx1:bx2][comp_crop] = 0
        l.mask[new_ys, new_xs] = snap_data[ys - sb[0], xs - sb[2]] if snap_data is not None else 255
        l.invalidate_bbox()

        # Build undo command
        old_bb_snap = self._old_bboxes.get(lid)
        new_bb = l.get_bbox()
        bbs = [b for b in (old_bb_snap, new_bb) if b is not None]
        if bbs:
            y1 = min(b[0] for b in bbs)
            y2 = max(b[1] for b in bbs)
            x1 = min(b[2] for b in bbs)
            x2 = max(b[3] for b in bbs)
            if sb is not None and snap_data is not None:
                sy1, sy2, sx1, sx2 = sb
                region_old = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                oy1, oy2 = max(y1, sy1), min(y2, sy2)
                ox1, ox2 = max(x1, sx1), min(x2, sx2)
                if oy2 > oy1 and ox2 > ox1:
                    region_old[oy1 - y1:oy2 - y1, ox1 - x1:ox2 - x1] = \
                        snap_data[oy1 - sy1:oy2 - sy1, ox1 - sx1:ox2 - sx1]
            else:
                region_old = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            region_new = l.mask[y1:y2, x1:x2]
            if not np.array_equal(region_old, region_new):
                cmd = UndoCommand(l, (y1, y2, x1, x2), region_old.copy(), region_new.copy())
                self.app.undo_stack.push(cmd)

        # Update component selection mask to new positions
        comp_mask_used = self._component_mask
        moved_layer = self._target_layers[0] if self._target_layers else None
        if comp_mask_used is not None and moved_layer is not None:
            idx_dy, idx_dx = int(round(dy)), int(round(dx))
            old_mask = comp_mask_used
            new_mask = np.zeros_like(old_mask)
            ys, xs = np.where(old_mask)
            nys, nxs = ys + idx_dy, xs + idx_dx
            mh, mw = new_mask.shape
            v = (nys >= 0) & (nys < mh) & (nxs >= 0) & (nxs < mw)
            new_mask[nys[v], nxs[v]] = True
            self._multi_comp_mask = new_mask
            self._multi_comp_layer = moved_layer

        self._comp_snapshots.clear()
        self._old_bboxes = {}
        self._target_layers.clear()
        self._start_pos = None
        self._component_mask = None
        self._component_bbox = None

        # Re-render affected ROI items
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

        if self._multi_comp_mask is not None and moved_layer is not None:
            self._show_marching_ants_for_mask(self._multi_comp_mask, moved_layer, canvas)
        else:
            visible = [l for l in affected if hasattr(l, 'mask') and l.get_bbox() is not None]
            if visible:
                self._show_marching_ants_multi(visible, canvas)

    # ------------------------------------------------------------------
    # Component preview (mask-based moves)
    # ------------------------------------------------------------------

    def _create_component_previews(self, canvas):
        """Split ROI into component (draggable) and remainder (static) pixmaps."""
        self._remove_component_previews(canvas, re_render=False)
        self._comp_hidden_layers = []

        for item in canvas._selection_highlight_items:
            item.setVisible(False)
        self._clear_marching_ants(canvas)

        l = self._target_layers[0]
        lid = id(l)
        rid = id(l)
        comp = self._component_mask
        scene = canvas.scene()

        real_item = canvas._roi_items.get(rid)
        z = real_item.zValue() if real_item else 10
        if real_item:
            real_item.setVisible(False)
        self._comp_hidden_layers.append((l, True))

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
            comp_item = QGraphicsPixmapItem(QPixmap.fromImage(qimg.copy()))
            comp_item.setOffset(cx1, cy1)
            comp_item.setZValue(z)
            comp_item.setAcceptedMouseButtons(Qt.NoButton)
            scene.addItem(comp_item)
            self._comp_preview_items.append(comp_item)
            self._comp_preview_offsets.append((cx1, cy1))
            self._comp_temp_items.append(comp_item)

        if np.any(non_comp):
            rem_rgba = np.zeros((lh, lw, 4), dtype=np.uint8)
            rem_rgba[non_comp] = [r, g, b, eff]
            rem_rgba = np.ascontiguousarray(rem_rgba)
            qimg2 = QImage(rem_rgba.data, lw, lh, lw * 4, QImage.Format_RGBA8888)
            rem_item = QGraphicsPixmapItem(QPixmap.fromImage(qimg2.copy()))
            rem_item.setOffset(lx1, ly1)
            rem_item.setZValue(z - 1)
            rem_item.setAcceptedMouseButtons(Qt.NoButton)
            scene.addItem(rem_item)
            self._comp_temp_items.append(rem_item)

        if self._component_bbox:
            cy1, cy2, cx1, cx2 = self._component_bbox
            pen = QPen(QColor(0, 180, 255), 1.5)
            pen.setCosmetic(True)
            pen.setStyle(Qt.DashLine)
            self._comp_bbox_item = scene.addRect(
                QRectF(cx1, cy1, cx2 - cx1, cy2 - cy1), pen
            )
            self._comp_bbox_item.setZValue(999)
            self._comp_temp_items.append(self._comp_bbox_item)

    def _remove_component_previews(self, canvas, re_render=True):
        scene = canvas.scene()
        for item in self._comp_temp_items:
            scene.removeItem(item)
        self._comp_temp_items.clear()
        self._comp_bbox_item = None
        self._comp_preview_items.clear()
        self._comp_preview_offsets = []
        for l, _ in self._comp_hidden_layers:
            rid = id(l)
            if rid in canvas._roi_items:
                canvas._roi_items[rid].setVisible(True)
        if re_render:
            lod = canvas._current_lod_level()
            for l, _ in self._comp_hidden_layers:
                try:
                    idx = canvas.layer_stack.roi_layers.index(l)
                    target_lod = 0 if l == canvas._active_layer else lod
                    canvas._refresh_roi_item(l, idx, lod_level=target_lod)
                    canvas._roi_lod[id(l)] = target_lod
                except ValueError:
                    pass
            canvas._update_selection_highlights()
        self._comp_hidden_layers = []

    # ------------------------------------------------------------------
    # Component delta clamping
    # ------------------------------------------------------------------

    def _clamp_component_delta(self, dx, dy):
        """Clamp dx/dy so the component bbox stays within mask bounds."""
        if not self._target_layers or self._component_bbox is None:
            return dx, dy
        l = self._target_layers[0]
        h, w = l.mask.shape
        cy1, cy2, cx1, cx2 = self._component_bbox
        idx = int(round(dx))
        idy = int(round(dy))
        idy = max(-cy1, min(idy, h - cy2))
        idx = max(-cx1, min(idx, w - cx2))
        return float(idx), float(idy)

    # ------------------------------------------------------------------
    # Scene rect expansion for OOB moves
    # ------------------------------------------------------------------

    def _expand_scene_rect_for_pos(self, pos, canvas):
        """Grow the scene rect when the drag position nears the boundary."""
        scene = canvas.scene()
        sr = scene.sceneRect()
        margin = 400
        px, py = pos.x(), pos.y()
        if px < sr.left() + margin or px > sr.right() - margin \
                or py < sr.top() + margin or py > sr.bottom() - margin:
            new_left = min(sr.left(), px - margin)
            new_top = min(sr.top(), py - margin)
            new_right = max(sr.right(), px + margin)
            new_bottom = max(sr.bottom(), py + margin)
            scene.setSceneRect(QRectF(new_left, new_top,
                                      new_right - new_left, new_bottom - new_top))

    # ------------------------------------------------------------------
    # Auto-scroll
    # ------------------------------------------------------------------

    def _auto_scroll(self, pos, canvas):
        from PySide6.QtCore import QPointF as _QPointF
        vp = canvas.viewport()
        vp_pos = canvas.mapFromScene(_QPointF(pos.x(), pos.y()))
        margin = 50
        scroll_speed = 20
        hs = canvas.horizontalScrollBar()
        vs = canvas.verticalScrollBar()
        if vp_pos.x() < margin:
            hs.setValue(hs.value() - scroll_speed)
        elif vp_pos.x() > vp.width() - margin:
            hs.setValue(hs.value() + scroll_speed)
        if vp_pos.y() < margin:
            vs.setValue(vs.value() - scroll_speed)
        elif vp_pos.y() > vp.height() - margin:
            vs.setValue(vs.value() + scroll_speed)

    # ------------------------------------------------------------------
    # Marching ants overlay
    # ------------------------------------------------------------------

    def _show_marching_ants_multi(self, layers, canvas):
        self._clear_marching_ants(canvas)
        from scipy.ndimage import binary_erosion

        scene = canvas.scene()
        for layer in layers:
            bbox = layer.get_bbox()
            if bbox is None:
                continue
            y1, y2, x1, x2 = bbox
            crop = layer.mask[y1:y2, x1:x2]
            binary = crop > 0
            eroded = binary_erosion(binary)
            edge = binary & ~eroded
            ey, ex = np.where(edge)
            if len(ey) == 0:
                continue
            h, w = y2 - y1, x2 - x1
            buf = np.zeros((h, w), dtype=np.uint8)
            item = QGraphicsPixmapItem()
            # Apply layer offset to ant positioning
            item.setOffset(x1 + layer.offset_x, y1 + layer.offset_y)
            item.setZValue(999)
            item.setAcceptedMouseButtons(Qt.NoButton)
            scene.addItem(item)
            self._ants_entries.append((item, ey, ex, bbox, buf))

        if not self._ants_entries:
            return

        self._ants_phase = 0
        self._build_ants_frame(canvas)

        self._ants_timer = QTimer()
        self._ants_timer.timeout.connect(lambda: self._tick_ants(canvas))
        self._ants_timer.start(150)

    def _show_marching_ants_for_mask(self, mask, layer, canvas):
        self._clear_marching_ants(canvas)
        from scipy.ndimage import binary_erosion
        from montaris.core.roi_transform import get_mask_bbox

        bbox = get_mask_bbox(mask.view(np.uint8) if mask.dtype == np.bool_ else mask)
        if bbox is None:
            return
        y1, y2, x1, x2 = bbox
        crop = mask[y1:y2, x1:x2]
        binary = crop > 0 if crop.dtype != np.bool_ else crop
        eroded = binary_erosion(binary)
        edge = binary & ~eroded
        ey, ex = np.where(edge)
        if len(ey) == 0:
            return

        h, w = y2 - y1, x2 - x1
        buf = np.zeros((h, w), dtype=np.uint8)
        scene = canvas.scene()
        item = QGraphicsPixmapItem()
        item.setOffset(x1 + layer.offset_x, y1 + layer.offset_y)
        item.setZValue(999)
        item.setAcceptedMouseButtons(Qt.NoButton)
        scene.addItem(item)
        self._ants_entries.append((item, ey, ex, bbox, buf))

        self._ants_phase = 0
        self._build_ants_frame(canvas)

        self._ants_timer = QTimer()
        self._ants_timer.timeout.connect(lambda: self._tick_ants(canvas))
        self._ants_timer.start(150)

    def _build_ants_frame(self, canvas):
        phase = self._ants_phase
        for item, ey, ex, bbox, buf in self._ants_entries:
            buf[:] = 0
            pattern = ((ex.astype(np.int32) + ey.astype(np.int32) + phase) % 8) < 4
            buf[ey[pattern], ex[pattern]] = 1
            buf[ey[~pattern], ex[~pattern]] = 2
            h, w = buf.shape
            qimg = QImage(buf.data, w, h, w, QImage.Format_Indexed8)
            qimg.setColorTable([0x00000000, 0xFF000000, 0xFFFFFFFF])
            item.setPixmap(QPixmap.fromImage(qimg))

    def _tick_ants(self, canvas):
        self._ants_phase = (self._ants_phase + 1) % 8
        if self._ants_entries:
            self._build_ants_frame(canvas)

    def _clear_marching_ants(self, canvas):
        if self._ants_timer is not None:
            self._ants_timer.stop()
            self._ants_timer = None
        scene = canvas.scene()
        for item, _, _, _, _ in self._ants_entries:
            scene.removeItem(item)
        self._ants_entries.clear()

    # ------------------------------------------------------------------
    # Multi-component selection
    # ------------------------------------------------------------------

    def _toggle_component(self, comp, layer, canvas):
        if self._multi_comp_layer is not layer:
            self._clear_multi_selection(canvas)
            self._multi_comp_mask = comp.copy()
            self._multi_comp_layer = layer
        else:
            overlap = self._multi_comp_mask & comp
            if np.any(overlap):
                self._multi_comp_mask[comp] = False
                if not np.any(self._multi_comp_mask):
                    self._clear_multi_selection(canvas)
                    return
            else:
                self._multi_comp_mask |= comp
        self._show_marching_ants_for_mask(self._multi_comp_mask, layer, canvas)

    def _multi_comp_bbox(self):
        if self._multi_comp_mask is None:
            return None
        from montaris.core.roi_transform import get_mask_bbox
        return get_mask_bbox(self._multi_comp_mask.view(np.uint8))

    def _clear_multi_selection(self, canvas):
        self._clear_marching_ants(canvas)
        self._multi_comp_mask = None
        self._multi_comp_layer = None

    def _clear_handles(self, canvas):
        """Called by canvas on tool/layer switch to clean up state."""
        self._clear_multi_selection(canvas)

    def cursor(self):
        return Qt.SizeAllCursor
