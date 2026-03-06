import math
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QLineF, QTimer
from PySide6.QtGui import QPen, QColor, QBrush, QTransform
from PySide6.QtWidgets import (
    QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsLineItem,
    QGraphicsPixmapItem,
)
from montaris.tools.base import BaseTool
from montaris.core.undo import UndoCommand
from montaris.core.multi_undo import CompoundUndoCommand
from montaris.core.roi_transform import (
    get_mask_bbox, compute_handles, apply_affine_to_mask,
    apply_affine_inplace, make_scale_matrix, make_rotation_matrix,
)

HANDLE_HIT_RADIUS = 12

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
        self._handle_items = []
        self._bbox_item = None
        self._canvas = None
        self._dragging = False
        self._shift_held = False
        self._component_mask = None
        self._preview_items = []  # references to ROI items for live preview
        self._hidden_layers = []

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
            # Recompute bbox for current selection (may have changed via Ctrl+click)
            bbox = self._compute_union_bbox(self._target_layers)
            if bbox is not None:
                self._bbox = bbox
            self._snapshots = {
                id(l): (l, l.mask.copy()) for l in self._target_layers
            }
            self._create_previews(canvas)
            return

        # First click: show handles — component-aware (D.14)
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
            angle = math.atan2(pos.x() - cx, -(pos.y() - cy)) - math.atan2(
                self._start_pos.x() - cx, -(self._start_pos.y() - cy)
            )
            if self._shift_held:
                snap_a = math.radians(15)
                angle = round(angle / snap_a) * snap_a
            M = make_rotation_matrix(angle, cx, cy)
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

        # Apply transform to preview items (instant, no numpy)
        # M is in scene coordinates. Preview items are positioned at (bx1, by1).
        # QTransform for item = translate(-bx,-by) → apply M → translate(bx,by)
        # But since item.pos() already places it, we use scene-level transform:
        # T = translate(-pos) * M_scene * translate(pos) — but simpler via
        # QTransform which operates in parent (scene) coords when set on item.
        qt_t = QTransform(
            M[0, 0], M[1, 0],
            M[0, 1], M[1, 1],
            M[0, 2], M[1, 2],
        )
        for item in self._preview_items:
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

        # Rasterize and collect bboxes for undo diff (no redundant get_mask_bbox)
        M = getattr(self, '_current_matrix', None)
        bbox_pairs = {}  # lid -> (src_bbox, dst_bbox)
        if M is not None:
            for lid, (l, snap) in self._snapshots.items():
                if not hasattr(l, 'mask'):
                    continue
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

        self._active_handle = None
        self._start_pos = None
        self._current_matrix = None
        self._preview_items.clear()
        self._hidden_layers = []

        # Show handles using dst_bbox (no extra get_mask_bbox call)
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

    def _update_hover(self, pos, canvas):
        """Hit-test handles and change cursor on hover."""
        if not self._bbox:
            return
        handles = compute_handles(self._bbox)
        for h in handles:
            dist = math.hypot(pos.x() - h.x, pos.y() - h.y)
            if dist <= HANDLE_HIT_RADIUS:
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
        handles = compute_handles(self._bbox)
        for h in handles:
            dist = math.hypot(pos.x() - h.x, pos.y() - h.y)
            if dist <= HANDLE_HIT_RADIUS:
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

    def _show_handles(self, bbox, canvas):
        self._clear_handles(canvas)
        self._bbox = bbox
        y1, y2, x1, x2 = bbox
        scene = canvas.scene()

        # Bounding box outline
        pen = QPen(QColor(0, 180, 255), 1.5)
        pen.setCosmetic(True)
        pen.setStyle(Qt.DashLine)
        self._bbox_item = scene.addRect(
            QRectF(x1, y1, x2 - x1, y2 - y1), pen
        )
        self._bbox_item.setZValue(999)

        handles = compute_handles(bbox)
        handle_pen = QPen(QColor(255, 255, 255), 2)
        handle_pen.setCosmetic(True)
        handle_brush = QBrush(QColor(0, 180, 255))
        rotate_pen = QPen(QColor(0, 0, 0), 2)
        rotate_pen.setCosmetic(True)
        rotate_brush = QBrush(QColor(255, 180, 0))

        for h in handles:
            if h.handle_type == 'rotate':
                # Circle handle for rotation — larger and more visible
                r = 7
                item = scene.addEllipse(
                    QRectF(h.x - r, h.y - r, r * 2, r * 2),
                    rotate_pen, rotate_brush,
                )
                item.setZValue(1001)
                self._handle_items.append(item)
                # Line connecting rotate handle to bbox top center
                cx = (x1 + x2) / 2
                line_pen = QPen(QColor(255, 180, 0), 1.5)
                line_pen.setCosmetic(True)
                line_item = scene.addLine(QLineF(cx, y1, cx, h.y + r), line_pen)
                line_item.setZValue(1000)
                self._handle_items.append(line_item)
                # Expand scene rect to ensure rotate handle is visible
                sr = scene.sceneRect()
                if h.y - r < sr.top():
                    sr.setTop(h.y - r - 5)
                    scene.setSceneRect(sr)
            elif h.handle_type in ('tl', 'tr', 'bl', 'br'):
                # 8x8 square for corner handles
                s = 4
                item = scene.addRect(
                    QRectF(h.x - s, h.y - s, s * 2, s * 2),
                    handle_pen, handle_brush,
                )
                item.setZValue(1000)
                self._handle_items.append(item)
            elif h.handle_type in ('tm', 'bm'):
                # 10x6 rect for top/bottom mid handles
                w_h, h_h = 5, 3
                item = scene.addRect(
                    QRectF(h.x - w_h, h.y - h_h, w_h * 2, h_h * 2),
                    handle_pen, handle_brush,
                )
                item.setZValue(1000)
                self._handle_items.append(item)
            elif h.handle_type in ('ml', 'mr'):
                # 6x10 rect for left/right mid handles
                w_h, h_h = 3, 5
                item = scene.addRect(
                    QRectF(h.x - w_h, h.y - h_h, w_h * 2, h_h * 2),
                    handle_pen, handle_brush,
                )
                item.setZValue(1000)
                self._handle_items.append(item)

    def _clear_handles(self, canvas):
        scene = canvas.scene()
        for item in self._handle_items:
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
