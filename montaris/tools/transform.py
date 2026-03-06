import math
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QLineF, QTimer
from PySide6.QtGui import QPen, QColor, QBrush, QTransform
from PySide6.QtWidgets import (
    QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsLineItem,
    QGraphicsPixmapItem, QGraphicsItem,
)
from montaris.tools.base import BaseTool
from montaris.core.undo import UndoCommand
from montaris.core.multi_undo import CompoundUndoCommand
from montaris.core.roi_transform import (
    TransformHandle, get_mask_bbox, compute_handles, apply_affine_to_mask,
    apply_affine_inplace, make_scale_matrix, make_rotation_matrix,
)

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
    name = "Transform"

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
        self._preview_items = []  # references to ROI items for live preview
        self._hidden_layers = []
        self._rotation = 0.0  # cumulative rotation angle in radians
        self._session_snapshots = {}  # persistent snapshots for entire transform session
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
            bbox = get_mask_bbox(l.mask)
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
            self._target_layers = self._get_target_layers(layer, canvas)
            # Use session snapshots (original state) for OOB recovery;
            # current mask state for undo diff
            self._snapshots = {
                id(l): (l, l.mask.copy()) for l in self._target_layers
            }
            # Session snapshot: taken once on first handle use, preserved across operations
            for l in self._target_layers:
                lid = id(l)
                if lid not in self._session_snapshots:
                    self._session_snapshots[lid] = l.mask.copy()
            self._create_previews(canvas)
            return

        # First click: show handles — component-aware (D.14)
        self._rotation = 0.0
        self._session_snapshots.clear()
        self._cumulative_matrix = None
        self._target_layers = self._get_target_layers(layer, canvas)
        self._component_mask = None

        # If single layer and clicking on painted pixel, detect component
        if (len(self._target_layers) == 1
                and hasattr(layer, 'mask')):
            ix, iy = int(pos.x()), int(pos.y())
            h, w = layer.mask.shape
            if 0 <= iy < h and 0 <= ix < w and layer.mask[iy, ix] > 0:
                from montaris.core.components import get_component_at
                comp = get_component_at(layer.mask, ix, iy)
                if comp is not None:
                    total = np.count_nonzero(layer.mask)
                    comp_px = np.count_nonzero(comp)
                    if comp_px < total:
                        self._component_mask = comp
                        ys, xs = np.where(comp)
                        bbox = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
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
            M = make_scale_matrix(sx, sy, cx, cy)

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

        # Move bbox rect with transform (pos is 0,0)
        if self._bbox_item:
            self._bbox_item.setTransform(qt_t)
        # Move handles by mapping original positions through matrix
        for item, ox, oy in self._handle_items:
            if ox is not None:
                nx = M[0, 0] * ox + M[0, 1] * oy + M[0, 2]
                ny = M[1, 0] * ox + M[1, 1] * oy + M[1, 2]
                item.setPos(nx, ny)
            else:
                # Line items (pos 0,0): apply scene transform
                item.setTransform(qt_t)

    def on_release(self, pos, layer, canvas):
        if not self._dragging or not self._snapshots:
            return
        self._dragging = False

        affected = [l for l, _ in self._hidden_layers]

        # Show progress bar immediately
        from PySide6.QtWidgets import QApplication
        canvas._progress_bar.setRange(0, 0)
        canvas._progress_bar.show()
        QApplication.processEvents()

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
        if M is not None:
            for lid, (l, snap) in self._snapshots.items():
                if not hasattr(l, 'mask'):
                    continue
                session_snap = self._session_snapshots.get(lid)
                if session_snap is not None and self._cumulative_matrix is not None:
                    cum_2x3 = self._cumulative_matrix[:2]
                    l.mask[:] = 0
                    src_bb, dst_bb = apply_affine_inplace(l.mask, session_snap, cum_2x3)
                else:
                    src_bb, dst_bb = apply_affine_inplace(l.mask, snap, M)
                bbox_pairs[lid] = (src_bb, dst_bb)
                l.invalidate_bbox()

        commands = []
        for lid, (l, snap) in self._snapshots.items():
            src_bb, dst_bb = bbox_pairs.get(lid, (None, None))
            if src_bb is None and dst_bb is None:
                continue
            bbs = [b for b in (src_bb, dst_bb) if b is not None]
            y1 = min(b[0] for b in bbs)
            y2 = max(b[1] for b in bbs)
            x1 = min(b[2] for b in bbs)
            x2 = max(b[3] for b in bbs)
            region_old = snap[y1:y2, x1:x2]
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

        # Accumulate rotation and preserve original bbox for visual
        was_rotate = self._active_handle and self._active_handle.handle_type == 'rotate'
        saved_bbox = self._bbox  # preserve before _clear_handles resets it
        if was_rotate:
            self._rotation += getattr(self, '_drag_angle', 0.0)
        self._drag_angle = 0.0

        self._active_handle = None
        self._start_pos = None
        self._current_matrix = None
        for item in self._preview_items:
            item.resetTransform()
        self._preview_items.clear()
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
        def _deferred_rebuild():
            for l in affected:
                try:
                    idx = canvas.layer_stack.roi_layers.index(l)
                    canvas._refresh_roi_item(l, idx)
                except ValueError:
                    pass
            canvas._update_selection_highlights()
            canvas._progress_bar.hide()

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

        for lid, (l, snap) in self._snapshots.items():
            rid = id(l)
            if rid in canvas._roi_items:
                # Reference existing item — no pop, no z-change, no scene modification
                self._preview_items.append(canvas._roi_items[rid])
            self._hidden_layers.append((l, True))

    def _remove_previews(self, canvas, re_render=True):
        """Reset preview transforms. If re_render, rebuild affected ROI items."""
        for item in self._preview_items:
            item.resetTransform()
        self._preview_items.clear()
        if re_render:
            for l, _ in getattr(self, '_hidden_layers', []):
                try:
                    idx = canvas.layer_stack.roi_layers.index(l)
                    canvas._refresh_roi_item(l, idx)
                except ValueError:
                    pass
            canvas._update_selection_highlights()
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
        self._clear_handles(canvas)
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
                self._handle_items.append((line_item, None, None))
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

    def _clear_handles(self, canvas):
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

    def cursor(self):
        return Qt.SizeAllCursor
