import math
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QLineF, QTimer
from PySide6.QtGui import QPen, QColor, QBrush, QTransform, QImage, QPixmap
from PySide6.QtWidgets import (
    QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsLineItem,
    QGraphicsPixmapItem, QGraphicsItem, QProgressDialog, QApplication,
)
from montaris.tools.base import BaseTool
from montaris.core.undo import UndoCommand
from montaris.core.multi_undo import CompoundUndoCommand
from montaris.core.roi_transform import (
    TransformHandle, compute_handles, apply_affine_to_mask,
    apply_affine_inplace, make_scale_matrix, make_rotation_matrix,
)
from montaris.core.components import get_component_at
from montaris.core.rle import rle_encode, rle_decode, rle_decode_crop

HANDLE_HIT_RADIUS = 24

# Cursor map per handle type
_HANDLE_CURSORS = {
    'tl': Qt.SizeFDiagCursor,
    'br': Qt.SizeFDiagCursor,
    'tr': Qt.SizeBDiagCursor,
    'bl': Qt.SizeBDiagCursor,
    'tm': Qt.SizeVerCursor,
    'bm': Qt.SizeVerCursor,
    'ml': Qt.SizeHorCursor,
    'mr': Qt.SizeHorCursor,
    'rotate': Qt.CrossCursor,
}


class TransformTool(BaseTool):
    name = "Transform (selected)"

    def __init__(self, app):
        super().__init__(app)
        self._active_handle = None
        self._hovered_handle = None
        self._start_pos = None
        self._bbox = None
        self._snapshots = {}  # id(layer) -> (layer, snapshot)
        self._target_layers = []
        self._handle_items = []  # (item, orig_x, orig_y) tuples
        self._bbox_item = None
        self._canvas = None
        self._dragging = False
        self._shift_held = False
        self._component_mask = None
        self._component_bbox = None  # bbox of component in original mask
        self._preview_items = []  # references to ROI items for live preview
        self._hidden_layers = []
        self._temp_scene_items = []  # temporary items added to scene (component mode)
        self._rotation = 0.0  # cumulative rotation angle in radians
        self._session_snapshots = {}  # persistent snapshots for entire transform session
        self._session_bboxes = {}  # id(layer) -> bbox at session start
        self._snap_bboxes = {}  # id(layer) -> bbox at operation start
        self._cumulative_matrix = None  # accumulated transform from session start

    def _get_target_layers(self, layer, canvas):
        """Return selected layers if in selection, else [layer]."""
        sel = canvas._selection.layers
        if sel and layer in sel:
            return sel
        return [layer]

    def _compute_union_bbox(self, layers):
        """Compute union bounding box across all target layers."""
        y1_min, y2_max, x1_min, x2_max = None, None, None, None
        for l in layers:
            bbox = l.get_bbox()
            if bbox is None:
                continue
            y1, y2, x1, x2 = bbox
            if y1_min is None:
                y1_min, y2_max, x1_min, x2_max = y1, y2, x1, x2
            else:
                y1_min = min(y1_min, y1)
                y2_max = max(y2_max, y2)
                x1_min = min(x1_min, x1)
                x2_max = max(x2_max, x2)
        if y1_min is None:
            return None
        return (y1_min, y2_max, x1_min, x2_max)

    def on_press(self, pos, layer, canvas):
        if layer is None or not hasattr(layer, 'mask'):
            return
        self._canvas = canvas

        # Check if clicking on a handle
        handle = self._hit_test_handle(pos)
        if handle:
            self._active_handle = handle
            self._start_pos = pos
            self._dragging = True
            # Save bbox_item's current transform so we can compose during drag
            self._bbox_base_transform = (
                self._bbox_item.transform() if self._bbox_item else QTransform()
            )
            self._target_layers = self._get_target_layers(layer, canvas)
            # Use session snapshots (original state) for OOB recovery;
            # current mask state for undo diff (crop-only for speed)
            self._snapshots = {}
            self._snap_bboxes = {}
            # Count how many layers need new session snapshots
            need_snap = [l for l in self._target_layers
                         if id(l) not in self._session_snapshots]
            progress = None
            if len(need_snap) > 5:
                progress = QProgressDialog(
                    "Preparing transform snapshots…", None,
                    0, len(need_snap), canvas.window())
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)
                progress.show()
                QApplication.processEvents()
            snap_count = 0
            for l in self._target_layers:
                lid = id(l)
                sb = l.get_bbox()
                self._snap_bboxes[lid] = sb
                # Session snapshot: taken once on first handle use.
                # Store bbox-crop RLE to avoid full-mask decode on release.
                if lid not in self._session_snapshots:
                    if sb is not None:
                        if l.is_compressed:
                            crop = rle_decode_crop(
                                l._rle_data, l._mask_shape, sb)
                        else:
                            crop = l.mask[sb[0]:sb[1], sb[2]:sb[3]]
                        self._session_snapshots[lid] = rle_encode(crop)
                    else:
                        self._session_snapshots[lid] = (b'', (0, 0))
                    self._session_bboxes[lid] = sb
                    # snap_data=None → undo diff uses decoded session on release
                    self._snapshots[lid] = (l, None, sb)
                    if progress:
                        snap_count += 1
                        progress.setValue(snap_count)
                else:
                    # Subsequent ops: crop-only snapshot (fast)
                    if sb is not None:
                        crop = l.mask[sb[0]:sb[1], sb[2]:sb[3]].copy()
                    else:
                        crop = None
                    self._snapshots[lid] = (l, crop, sb)
            if progress:
                progress.close()
            self._create_previews(canvas)
            return

        # First click: show handles — component-aware (D.14)
        self._rotation = 0.0
        self._session_snapshots.clear()
        self._session_bboxes.clear()
        self._cumulative_matrix = None
        self._target_layers = self._get_target_layers(layer, canvas)
        self._component_mask = None
        self._component_bbox = None

        # If single layer and clicking on painted pixel, detect component
        if (len(self._target_layers) == 1
                and hasattr(layer, 'mask')):
            ix, iy = int(pos.x()), int(pos.y())
            h, w = layer.mask.shape
            if 0 <= iy < h and 0 <= ix < w and layer.mask[iy, ix] > 0:
                layer_bbox = layer.get_bbox()
                comp = get_component_at(layer.mask, ix, iy, bbox=layer_bbox)
                if comp is not None:
                    # Count pixels within the bbox crop to avoid full-mask scans
                    by1, by2, bx1, bx2 = layer_bbox
                    crop = layer.mask[by1:by2, bx1:bx2]
                    comp_crop = comp[by1:by2, bx1:bx2]
                    total = np.count_nonzero(crop)
                    comp_px = np.count_nonzero(comp_crop)
                    if comp_px < total:
                        self._component_mask = comp
                        cys, cxs = np.where(comp_crop)
                        bbox = (cys.min() + by1, cys.max() + 1 + by1,
                                cxs.min() + bx1, cxs.max() + 1 + bx1)
                        self._component_bbox = bbox
                        self._show_handles(bbox, canvas)
                        return

        bbox = self._compute_union_bbox(self._target_layers)
        if bbox is None:
            return
        self._show_handles(bbox, canvas)

    def on_move(self, pos, layer, canvas):
        self._canvas = canvas

        # Update shift state from keyboard modifiers
        # (Qt provides modifiers on mouse events via QApplication)
        from PySide6.QtWidgets import QApplication
        self._shift_held = bool(QApplication.keyboardModifiers() & Qt.ShiftModifier)

        # Hover feedback when not dragging
        if not self._dragging:
            self._update_hover(pos, canvas)
            return

        if self._active_handle is None or not self._snapshots:
            return

        dx = pos.x() - self._start_pos.x()
        dy = pos.y() - self._start_pos.y()

        y1, y2, x1, x2 = self._bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = x2 - x1
        bh = y2 - y1

        handle_type = self._active_handle.handle_type

        if handle_type == 'rotate':
            self._drag_angle = math.atan2(pos.x() - cx, -(pos.y() - cy)) - math.atan2(
                self._start_pos.x() - cx, -(self._start_pos.y() - cy)
            )
            if self._shift_held:
                snap_a = math.radians(15)
                self._drag_angle = round(self._drag_angle / snap_a) * snap_a
            M = make_rotation_matrix(self._drag_angle, cx, cy)
        else:
            sx, sy = 1.0, 1.0
            if 'l' in handle_type:
                sx = max(0.1, (bw - dx) / bw)
            elif 'r' in handle_type:
                sx = max(0.1, (bw + dx) / bw)
            if 't' in handle_type:
                sy = max(0.1, (bh - dy) / bh)
            elif 'b' in handle_type:
                sy = max(0.1, (bh + dy) / bh)
            if handle_type in ('tm', 'bm'):
                sx = 1.0
            if handle_type in ('ml', 'mr'):
                sy = 1.0
            if self._shift_held and handle_type in ('tl', 'tr', 'bl', 'br'):
                if abs(dx) >= abs(dy):
                    sx = sy = sx
                else:
                    sx = sy = sy

            # Anchor at the opposite corner/edge (not the center)
            anchor_x, anchor_y = cx, cy
            if handle_type == 'tl':
                anchor_x, anchor_y = x2, y2
            elif handle_type == 'tr':
                anchor_x, anchor_y = x1, y2
            elif handle_type == 'bl':
                anchor_x, anchor_y = x2, y1
            elif handle_type == 'br':
                anchor_x, anchor_y = x1, y1
            elif handle_type == 'tm':
                anchor_y = y2
            elif handle_type == 'bm':
                anchor_y = y1
            elif handle_type == 'ml':
                anchor_x = x2
            elif handle_type == 'mr':
                anchor_x = x1

            M = make_scale_matrix(sx, sy, anchor_x, anchor_y)

        self._current_matrix = M

        # Scene-level transform (for items at pos 0,0 like bbox_item)
        qt_t = QTransform(
            M[0, 0], M[1, 0],
            M[0, 1], M[1, 1],
            M[0, 2], M[1, 2],
        )

        # Apply per-item transform accounting for item's pos offset
        for item in self._preview_items:
            px, py = item.pos().x(), item.pos().y()
            tdx = M[0, 0] * px + M[0, 1] * py + M[0, 2] - px
            tdy = M[1, 0] * px + M[1, 1] * py + M[1, 2] - py
            item_t = QTransform(
                M[0, 0], M[1, 0],
                M[0, 1], M[1, 1],
                tdx, tdy,
            )
            item.setTransform(item_t)

        # Move bbox rect: compose prior rotation with current drag transform
        if self._bbox_item:
            base = getattr(self, '_bbox_base_transform', QTransform())
            self._bbox_item.setTransform(base * qt_t)
        # Move handles by mapping original positions through matrix
        for item, ox, oy in self._handle_items:
            if isinstance(ox, tuple):
                # Line item: map both endpoints through M
                lx1, ly1, lx2, ly2 = ox
                nx1 = M[0, 0] * lx1 + M[0, 1] * ly1 + M[0, 2]
                ny1 = M[1, 0] * lx1 + M[1, 1] * ly1 + M[1, 2]
                nx2 = M[0, 0] * lx2 + M[0, 1] * ly2 + M[0, 2]
                ny2 = M[1, 0] * lx2 + M[1, 1] * ly2 + M[1, 2]
                item.setLine(QLineF(nx1, ny1, nx2, ny2))
            elif ox is not None:
                nx = M[0, 0] * ox + M[0, 1] * oy + M[0, 2]
                ny = M[1, 0] * ox + M[1, 1] * oy + M[1, 2]
                item.setPos(nx, ny)

    def on_release(self, pos, layer, canvas):
        if not self._dragging or not self._snapshots:
            return
        self._dragging = False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        canvas.flash_progress("Applying transform…")

        affected = [l for l, _ in self._hidden_layers]

        # Accumulate transform matrix for session-level OOB preservation
        M = getattr(self, '_current_matrix', None)
        if M is not None:
            M3 = np.array([
                [M[0, 0], M[0, 1], M[0, 2]],
                [M[1, 0], M[1, 1], M[1, 2]],
                [0, 0, 1],
            ], dtype=np.float64)
            if self._cumulative_matrix is not None:
                self._cumulative_matrix = M3 @ self._cumulative_matrix
            else:
                self._cumulative_matrix = M3

        # Rasterize from session snapshot using cumulative matrix (preserves OOB pixels)
        bbox_pairs = {}  # lid -> (src_bbox, dst_bbox)
        commands = []
        if M is not None and self._component_mask is not None:
            # Component-aware: transform only the clicked component
            l = self._target_layers[0]
            lid = id(l)
            session_rle = self._session_snapshots.get(lid)
            session_bbox = self._session_bboxes.get(lid)
            # Component mode needs full-size mask — reconstruct from crop
            if session_rle is not None and session_rle[0] and session_bbox is not None:
                crop = rle_decode(*session_rle)
                session_snap = np.zeros(l.mask.shape, dtype=np.uint8)
                by1, by2, bx1, bx2 = session_bbox
                session_snap[by1:by2, bx1:bx2] = crop
            else:
                session_snap = None
            snap_bb = self._snap_bboxes.get(lid)  # bbox before this drag
            if session_snap is not None and self._cumulative_matrix is not None:
                cum_2x3 = self._cumulative_matrix[:2]
                comp = self._component_mask
                # Restore non-component pixels from session snapshot
                l.mask[:] = session_snap
                l.mask[comp] = 0
                # Transform only component pixels (don't clear dest — non-component pixels live there)
                comp_snap = np.zeros_like(session_snap)
                comp_snap[comp] = session_snap[comp]
                src_bb, dst_bb = apply_affine_inplace(
                    l.mask, comp_snap, cum_2x3,
                    src_bbox=self._component_bbox,
                    clear_src=False)
                # For undo diff, include snap bbox (previous component position)
                # so the union region covers both old and new component locations
                if snap_bb is not None:
                    src_bb = snap_bb if src_bb is None else (
                        min(src_bb[0], snap_bb[0]), max(src_bb[1], snap_bb[1]),
                        min(src_bb[2], snap_bb[2]), max(src_bb[3], snap_bb[3]))
                bbox_pairs[lid] = (src_bb, dst_bb)
                l.invalidate_bbox()
                # Component undo diff
                snap_data = self._snapshots[lid][1]
                undo_snap = snap_data if snap_data is not None else session_snap
                if undo_snap is not None:
                    bbs = [b for b in (src_bb, dst_bb) if b is not None]
                    if bbs:
                        uy1 = min(b[0] for b in bbs)
                        uy2 = max(b[1] for b in bbs)
                        ux1 = min(b[2] for b in bbs)
                        ux2 = max(b[3] for b in bbs)
                        region_old = undo_snap[uy1:uy2, ux1:ux2].copy()
                        region_new = l.mask[uy1:uy2, ux1:ux2]
                        if not np.array_equal(region_old, region_new):
                            commands.append(UndoCommand(
                                l, (uy1, uy2, ux1, ux2),
                                region_old, region_new.copy()))
        elif M is not None:
            # Merged transform + undo diff loop: decode session snapshot once,
            # use for both affine rasterization and undo diff.
            cum_2x3 = self._cumulative_matrix[:2] if self._cumulative_matrix is not None else None

            def _transform_one(lid, l, snap_data, sb, session_rle, session_bbox):
                """Decode session crop, apply affine, compute undo diff."""
                # Session snapshots are now bbox-crop RLE (not full-mask).
                if session_rle is not None and session_rle[0] and cum_2x3 is not None:
                    session_crop = rle_decode(*session_rle)
                    # Clear only the old content bbox (sb), not the entire mask
                    if sb:
                        l.mask[sb[0]:sb[1], sb[2]:sb[3]] = 0
                    src_bb, dst_bb = apply_affine_inplace(
                        l.mask, session_crop, cum_2x3,
                        src_bbox=session_bbox, clear_src=False,
                        snap_is_crop=True)
                elif snap_data is not None:
                    src_bb, dst_bb = apply_affine_inplace(
                        l.mask, snap_data, M,
                        src_bbox=self._snap_bboxes.get(lid))
                elif session_rle is not None and session_rle[0]:
                    session_crop = rle_decode(*session_rle)
                    src_bb, dst_bb = apply_affine_inplace(
                        l.mask, session_crop, M,
                        src_bbox=session_bbox, snap_is_crop=True)
                else:
                    return None, None, None
                l.invalidate_bbox()
                if src_bb is None and dst_bb is None:
                    return None, None, None
                bbs = [b for b in (src_bb, dst_bb) if b is not None]
                y1 = min(b[0] for b in bbs)
                y2 = max(b[1] for b in bbs)
                x1 = min(b[2] for b in bbs)
                x2 = max(b[3] for b in bbs)
                # Undo diff: use snap_data if available, else decoded session crop
                if snap_data is not None:
                    undo_snap = snap_data
                    undo_is_crop = (sb is not None
                                    and snap_data.shape != l.mask.shape)
                elif session_rle is not None and session_rle[0]:
                    undo_snap = rle_decode(*session_rle)
                    undo_is_crop = True
                else:
                    undo_snap = None
                    undo_is_crop = False
                if sb is not None and undo_snap is not None and undo_is_crop:
                    sy1, sy2, sx1, sx2 = sb
                    region_old = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                    oy1, oy2 = max(y1, sy1), min(y2, sy2)
                    ox1, ox2 = max(x1, sx1), min(x2, sx2)
                    if oy2 > oy1 and ox2 > ox1:
                        region_old[oy1 - y1:oy2 - y1, ox1 - x1:ox2 - x1] = \
                            undo_snap[oy1 - sy1:oy2 - sy1, ox1 - sx1:ox2 - sx1]
                elif undo_snap is not None and not undo_is_crop:
                    region_old = undo_snap[y1:y2, x1:x2].copy()
                else:
                    region_old = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                return (src_bb, dst_bb), (y1, y2, x1, x2), region_old

            # Build work items
            work = []
            for lid, (l, snap_data, sb) in self._snapshots.items():
                if not hasattr(l, 'mask'):
                    continue
                session_rle = self._session_snapshots.get(lid)
                session_bbox = self._session_bboxes.get(lid)
                work.append((lid, l, snap_data, sb, session_rle, session_bbox))

            # Parallel decode+affine for many layers (Transform All)
            if len(work) > 3:
                from montaris.core.workers import get_pool
                futures = [(lid, l, get_pool().submit(
                    _transform_one, lid, l, sd, sb, sr, sbx))
                    for lid, l, sd, sb, sr, sbx in work]
                for lid, l, fut in futures:
                    bb_pair, undo_region, region_old = fut.result()
                    if bb_pair is not None:
                        bbox_pairs[lid] = bb_pair
                        region_new = l.mask[undo_region[0]:undo_region[1],
                                            undo_region[2]:undo_region[3]]
                        if not np.array_equal(region_old, region_new):
                            commands.append(UndoCommand(
                                l, undo_region,
                                region_old if region_old.base is None else region_old.copy(),
                                region_new.copy()))
            else:
                for lid, l, snap_data, sb, session_rle, session_bbox in work:
                    bb_pair, undo_region, region_old = _transform_one(
                        lid, l, snap_data, sb, session_rle, session_bbox)
                    if bb_pair is not None:
                        bbox_pairs[lid] = bb_pair
                        region_new = l.mask[undo_region[0]:undo_region[1],
                                            undo_region[2]:undo_region[3]]
                        if not np.array_equal(region_old, region_new):
                            commands.append(UndoCommand(
                                l, undo_region,
                                region_old if region_old.base is None else region_old.copy(),
                                region_new.copy()))

        if commands:
            if len(commands) == 1:
                self.app.undo_stack.push(commands[0])
            else:
                self.app.undo_stack.push(CompoundUndoCommand(commands))

        # Accumulate rotation and preserve original bbox for visual
        was_rotate = self._active_handle and self._active_handle.handle_type == 'rotate'
        saved_bbox = self._bbox  # preserve before _clear_handles resets it
        if was_rotate:
            self._rotation += getattr(self, '_drag_angle', 0.0)
        self._drag_angle = 0.0

        self._active_handle = None
        self._start_pos = None
        self._current_matrix = None

        # Keep preview visible until deferred rebuild finishes re-rendering
        stale_preview_items = self._preview_items[:]
        self._preview_items.clear()
        stale_temp_items = self._temp_scene_items[:]
        self._temp_scene_items.clear()
        hidden_layers = self._hidden_layers[:]
        self._hidden_layers = []

        # After rotation: keep original bbox (don't use inflated axis-aligned bbox)
        # After scale: use new rasterized bbox, reset rotation
        if was_rotate and saved_bbox is not None:
            self._show_handles(saved_bbox, canvas)
        else:
            self._rotation = 0.0
            new_bboxes = [dst_bb for src_bb, dst_bb in bbox_pairs.values() if dst_bb is not None]
            if new_bboxes:
                y1 = min(b[0] for b in new_bboxes)
                y2 = max(b[1] for b in new_bboxes)
                x1 = min(b[2] for b in new_bboxes)
                x2 = max(b[3] for b in new_bboxes)
                self._show_handles((y1, y2, x1, x2), canvas)
        self._snapshots.clear()

        # Defer expensive visual rebuild to next frame (render + highlights)
        is_batch = isinstance(self, TransformAllTool)

        def _deferred_rebuild():
            lod = canvas._current_lod_level()
            # Force minimum LOD 2 during Transform All for faster rebuilds
            if is_batch:
                lod = max(lod, 2)

            gof = canvas.layer_stack._global_opacity_factor

            if is_batch and len(affected) > 3:
                # Parallel rebuild for Transform All with viewport culling
                from montaris.core.workers import get_pool
                from montaris.canvas import _compute_roi_rgba_from_crop
                from PySide6.QtCore import QRectF
                vp_rect = canvas.mapToScene(canvas.viewport().rect()).boundingRect()
                use_culling = vp_rect.width() > 0 and vp_rect.height() > 0
                futures = []
                for l in affected:
                    try:
                        idx = canvas.layer_stack.roi_layers.index(l)
                    except ValueError:
                        continue
                    rid = id(l)
                    target_lod = 0 if l == canvas._active_layer else lod
                    bbox = l.get_bbox()
                    if bbox is None or not l.visible:
                        futures.append((l, idx, target_lod, None))
                        continue
                    y1, y2, x1, x2 = bbox
                    # Viewport culling: skip off-screen ROIs
                    if use_culling and l != canvas._active_layer:
                        dx, dy = l.offset_x, l.offset_y
                        if not vp_rect.intersects(QRectF(x1 + dx, y1 + dy, x2 - x1, y2 - y1)):
                            canvas._roi_stale.add(rid)
                            continue
                    mask_crop = l.get_mask_crop((y1, y2, x1, x2)).copy()
                    eff = int(l.opacity * gof)
                    fill_mode = getattr(l, 'fill_mode', 'solid')
                    fut = get_pool().submit(
                        _compute_roi_rgba_from_crop, mask_crop, l.color,
                        eff, fill_mode, target_lod,
                        x1 + l.offset_x, y1 + l.offset_y)
                    futures.append((l, idx, target_lod, fut))
                for l, idx, target_lod, fut in futures:
                    rid = id(l)
                    if fut is None:
                        existing = canvas._roi_items.get(rid)
                        if existing is not None:
                            existing.setVisible(False)
                    else:
                        canvas._apply_roi_rgba_result(l, idx, fut.result())
                    canvas._roi_lod[rid] = target_lod
            else:
                for l in affected:
                    try:
                        idx = canvas.layer_stack.roi_layers.index(l)
                        target_lod = 0 if l == canvas._active_layer else lod
                        canvas._refresh_roi_item(l, idx, lod_level=target_lod)
                        canvas._roi_lod[id(l)] = target_lod
                    except ValueError:
                        pass

            # Now that new pixmaps are ready, clean up stale preview
            for item in stale_preview_items:
                item.resetTransform()
            scene = canvas.scene()
            for item in stale_temp_items:
                scene.removeItem(item)
            # Restore real item visibility after re-render
            for l, _ in hidden_layers:
                rid = id(l)
                if rid in canvas._roi_items:
                    canvas._roi_items[rid].setVisible(True)
            canvas._update_selection_highlights()

        QApplication.restoreOverrideCursor()
        QTimer.singleShot(0, _deferred_rebuild)

    def on_key_press(self, key, canvas):
        # Escape: cancel transform, restore snapshots
        if key == Qt.Key_Escape and self._dragging and self._snapshots:
            self._remove_previews(canvas)
            # Masks were never modified during drag, no need to restore
            self._dragging = False
            self._active_handle = None
            self._snapshots.clear()
            self._start_pos = None
            self._current_matrix = None
            canvas._update_selection_highlights()
            # Refresh handles
            if self._target_layers:
                bbox = self._compute_union_bbox(self._target_layers)
                if bbox:
                    self._show_handles(bbox, canvas)
            return True  # consumed

    def _get_rotated_handles(self):
        """Get handle positions from stored items (already rotated)."""
        if not self._bbox:
            return []
        y1, y2, x1, x2 = self._bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        angle = self._rotation
        scale = self._view_scale(self._canvas) if self._canvas else 1.0
        rotate_dist = max(30, 50 / max(scale, 0.01))
        raw = [
            (x1, y1, 'tl'), (x2, y1, 'tr'), (x1, y2, 'bl'), (x2, y2, 'br'),
            (cx, y1, 'tm'), (cx, y2, 'bm'), (x1, cy, 'ml'), (x2, cy, 'mr'),
            (cx, y1 - rotate_dist, 'rotate'),
        ]
        result = []
        for hx, hy, htype in raw:
            rx, ry = self._rotate_point(hx, hy, cx, cy, angle)
            result.append(TransformHandle(rx, ry, htype))
        return result

    def _update_hover(self, pos, canvas):
        """Hit-test handles and change cursor on hover."""
        if not self._bbox:
            return
        scale = self._view_scale(canvas)
        hit_r = HANDLE_HIT_RADIUS / max(scale, 0.01)
        for h in self._get_rotated_handles():
            dist = math.hypot(pos.x() - h.x, pos.y() - h.y)
            if dist <= hit_r:
                if self._hovered_handle != h.handle_type:
                    self._hovered_handle = h.handle_type
                    cursor = _HANDLE_CURSORS.get(h.handle_type, Qt.SizeAllCursor)
                    canvas.setCursor(cursor)
                return
        if self._hovered_handle is not None:
            self._hovered_handle = None
            canvas.setCursor(self.cursor())

    def _hit_test_handle(self, pos):
        if not self._bbox:
            return None
        scale = self._view_scale(self._canvas) if self._canvas else 1.0
        hit_r = HANDLE_HIT_RADIUS / max(scale, 0.01)
        for h in self._get_rotated_handles():
            dist = math.hypot(pos.x() - h.x, pos.y() - h.y)
            if dist <= hit_r:
                return h
        return None

    def _create_previews(self, canvas):
        """Use existing ROI pixmap items as live preview (zero scene changes)."""
        for item in self._preview_items:
            item.resetTransform()
        self._preview_items.clear()
        self._hidden_layers = []

        # Hide selection highlights during drag
        for item in canvas._selection_highlight_items:
            item.setVisible(False)

        if self._component_mask is not None:
            self._create_component_previews(canvas)
            return

        for lid, (l, snap_data, sb) in self._snapshots.items():
            rid = id(l)
            if rid in canvas._roi_items:
                # Reference existing item — no pop, no z-change, no scene modification
                self._preview_items.append(canvas._roi_items[rid])
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

        # Identify current component vs remainder pixels using layer bbox crop
        layer_bb = l.get_bbox()
        if layer_bb is None:
            return
        ly1, ly2, lx1, lx2 = layer_bb
        lh, lw = ly2 - ly1, lx2 - lx1
        mask_crop = l.mask[ly1:ly2, lx1:lx2]

        # Non-component = pixels in session snapshot outside the original component mask
        session_rle = self._session_snapshots.get(lid)
        session_bbox = self._session_bboxes.get(lid)
        if session_rle is not None and session_rle[0] and session_bbox is not None:
            sess_crop = rle_decode(*session_rle)
            sby1, sby2, sbx1, sbx2 = session_bbox
            # Build snap_crop aligned to layer_bb
            snap_crop = np.zeros((lh, lw), dtype=np.uint8)
            oy1 = max(ly1, sby1); oy2 = min(ly2, sby2)
            ox1 = max(lx1, sbx1); ox2 = min(lx2, sbx2)
            if oy2 > oy1 and ox2 > ox1:
                snap_crop[oy1-ly1:oy2-ly1, ox1-lx1:ox2-lx1] = \
                    sess_crop[oy1-sby1:oy2-sby1, ox1-sbx1:ox2-sbx1]
            comp_crop_full = comp[ly1:ly2, lx1:lx2]
            non_comp = (snap_crop > 0) & ~comp_crop_full
        else:
            non_comp = np.zeros((lh, lw), dtype=bool)
        current_comp = (mask_crop > 0) & ~non_comp

        # Component-only pixmap (this gets transformed during drag)
        cys, cxs = np.where(current_comp)
        if len(cys) > 0:
            cy1 = int(cys.min()) + ly1
            cy2 = int(cys.max()) + 1 + ly1
            cx1 = int(cxs.min()) + lx1
            cx2 = int(cxs.max()) + 1 + lx1
            ch, cw = cy2 - cy1, cx2 - cx1
            comp_region = current_comp[cy1 - ly1:cy2 - ly1, cx1 - lx1:cx2 - lx1]
            mask_region = mask_crop[cy1 - ly1:cy2 - ly1, cx1 - lx1:cx2 - lx1]
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
            self._temp_scene_items.append(comp_item)

        # Remainder pixmap (static, not transformed)
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

    def _remove_previews(self, canvas, re_render=True):
        """Reset preview transforms. If re_render, rebuild affected ROI items."""
        for item in self._preview_items:
            item.resetTransform()
        self._preview_items.clear()
        # Remove temporary scene items (component mode)
        scene = canvas.scene()
        for item in self._temp_scene_items:
            scene.removeItem(item)
        self._temp_scene_items.clear()
        if re_render:
            lod = canvas._current_lod_level()
            for l, _ in getattr(self, '_hidden_layers', []):
                try:
                    idx = canvas.layer_stack.roi_layers.index(l)
                    target_lod = 0 if l == canvas._active_layer else lod
                    canvas._refresh_roi_item(l, idx, lod_level=target_lod)
                    canvas._roi_lod[id(l)] = target_lod
                except ValueError:
                    pass
            canvas._update_selection_highlights()
        # Restore visibility after re-render (avoids old-pixmap flash)
        for l, _ in getattr(self, '_hidden_layers', []):
            rid = id(l)
            if rid in canvas._roi_items:
                canvas._roi_items[rid].setVisible(True)
        self._hidden_layers = []

    def _view_scale(self, canvas):
        """Return the current view scale factor (pixels per scene unit)."""
        return canvas.transform().m11() or 1.0

    def _rotate_point(self, x, y, cx, cy, angle):
        """Rotate point (x,y) around (cx,cy) by angle radians."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        dx, dy = x - cx, y - cy
        return cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a

    def _show_handles(self, bbox, canvas):
        self._clear_handle_visuals(canvas)
        self._bbox = bbox
        y1, y2, x1, x2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        angle = self._rotation
        scene = canvas.scene()

        # Rotated bounding box outline
        pen = QPen(QColor(0, 180, 255), 1.5)
        pen.setCosmetic(True)
        pen.setStyle(Qt.DashLine)
        self._bbox_item = scene.addRect(
            QRectF(x1, y1, x2 - x1, y2 - y1), pen
        )
        self._bbox_item.setZValue(999)
        if angle != 0:
            bbox_t = QTransform()
            bbox_t.translate(cx, cy)
            bbox_t.rotateRadians(angle)
            bbox_t.translate(-cx, -cy)
            self._bbox_item.setTransform(bbox_t)

        # Rotate handle distance scales with zoom (constant screen distance)
        scale = self._view_scale(canvas)
        rotate_dist = max(30, 50 / max(scale, 0.01))

        # Compute handle positions (axis-aligned), then rotate
        raw_handles = [
            (x1, y1, 'tl'), (x2, y1, 'tr'), (x1, y2, 'bl'), (x2, y2, 'br'),
            (cx, y1, 'tm'), (cx, y2, 'bm'), (x1, cy, 'ml'), (x2, cy, 'mr'),
            (cx, y1 - rotate_dist, 'rotate'),
        ]
        handle_pen = QPen(QColor(255, 255, 255), 2)
        handle_brush = QBrush(QColor(0, 180, 255))
        rotate_pen = QPen(QColor(0, 0, 0), 2)
        rotate_brush = QBrush(QColor(255, 180, 0))

        for hx, hy, htype in raw_handles:
            # Rotate handle position around center
            rx, ry = self._rotate_point(hx, hy, cx, cy, angle)

            if htype == 'rotate':
                r = 10
                item = scene.addEllipse(
                    QRectF(-r, -r, r * 2, r * 2),
                    rotate_pen, rotate_brush,
                )
                item.setPos(rx, ry)
                item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                item.setZValue(1001)
                self._handle_items.append((item, rx, ry))
                # Line from rotated top-center to rotate handle
                tcx, tcy = self._rotate_point(cx, y1, cx, cy, angle)
                line_pen = QPen(QColor(255, 180, 0), 1.5)
                line_pen.setCosmetic(True)
                line_item = scene.addLine(QLineF(tcx, tcy, rx, ry), line_pen)
                line_item.setZValue(1000)
                # Store line endpoints for direct mapping in on_move
                self._handle_items.append((line_item, (tcx, tcy, rx, ry), None))
            else:
                if htype in ('tl', 'tr', 'bl', 'br'):
                    s = 7
                    item = scene.addRect(
                        QRectF(-s, -s, s * 2, s * 2),
                        handle_pen, handle_brush,
                    )
                elif htype in ('tm', 'bm'):
                    w_h, h_h = 9, 5
                    item = scene.addRect(
                        QRectF(-w_h, -h_h, w_h * 2, h_h * 2),
                        handle_pen, handle_brush,
                    )
                elif htype in ('ml', 'mr'):
                    w_h, h_h = 5, 9
                    item = scene.addRect(
                        QRectF(-w_h, -h_h, w_h * 2, h_h * 2),
                        handle_pen, handle_brush,
                    )
                else:
                    continue
                item.setPos(rx, ry)
                item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                item.setZValue(1000)
                self._handle_items.append((item, rx, ry))

    def _clear_handle_visuals(self, canvas):
        """Remove handle scene items without resetting session/component state."""
        scene = canvas.scene()
        for item, _, _ in self._handle_items:
            scene.removeItem(item)
        self._handle_items.clear()
        if self._bbox_item:
            scene.removeItem(self._bbox_item)
            self._bbox_item = None
        self._bbox = None
        self._hovered_handle = None
        if self._preview_items:
            self._remove_previews(canvas)

    def _clear_handles(self, canvas):
        self._clear_handle_visuals(canvas)
        # Reset session state so it doesn't carry to a different layer
        self._component_mask = None
        self._component_bbox = None
        self._session_snapshots.clear()
        self._session_bboxes.clear()
        self._cumulative_matrix = None

    def cursor(self):
        return Qt.SizeAllCursor


class TransformAllTool(TransformTool):
    """Transform ALL ROI layers, not just selected ones."""
    name = "Transform All"

    def on_activate(self, layer, canvas):
        """Select all ROIs and show transform handles immediately."""
        all_rois = list(canvas.layer_stack.roi_layers)
        if all_rois:
            canvas._selection.select_all(canvas.layer_stack.roi_layers)
            self._canvas = canvas
            self._target_layers = all_rois
            self._rotation = 0.0
            self._session_snapshots.clear()
            self._session_bboxes.clear()
            self._cumulative_matrix = None
            bbox = self._compute_union_bbox(all_rois)
            if bbox:
                self._show_handles(bbox, canvas)

    def _get_target_layers(self, layer, canvas):
        return list(canvas.layer_stack.roi_layers)
