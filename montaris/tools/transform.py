import math
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPen, QColor, QBrush
from montaris.tools.base import BaseTool
from montaris.core.undo import UndoCommand
from montaris.core.roi_transform import (
    get_mask_bbox, compute_handles, apply_affine_to_mask,
    make_scale_matrix, make_rotation_matrix,
)

HANDLE_SIZE = 8
HANDLE_HIT_RADIUS = 12


class TransformTool(BaseTool):
    name = "Transform"

    def __init__(self, app):
        super().__init__(app)
        self.apply_to_all = False
        self._active_handle = None
        self._start_pos = None
        self._bbox = None
        self._snapshot = None
        self._handle_items = []
        self._bbox_item = None
        self._canvas = None
        self._layer = None

    def on_press(self, pos, layer, canvas):
        if layer is None or not hasattr(layer, 'mask'):
            return
        self._canvas = canvas
        self._layer = layer

        # Check if clicking on a handle
        handle = self._hit_test_handle(pos)
        if handle:
            self._active_handle = handle
            self._start_pos = pos
            self._snapshot = layer.mask.copy()
            return

        # First click: show handles around mask bbox
        bbox = get_mask_bbox(layer.mask)
        if bbox is None:
            return
        self._bbox = bbox
        self._show_handles(bbox, canvas)

    def on_move(self, pos, layer, canvas):
        if self._active_handle is None or self._snapshot is None:
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
            # Compute rotation angle
            angle = math.atan2(pos.x() - cx, -(pos.y() - cy)) - math.atan2(
                self._start_pos.x() - cx, -(self._start_pos.y() - cy)
            )
            M = make_rotation_matrix(angle, cx, cy)
        else:
            # Scale based on handle movement
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
            M = make_scale_matrix(sx, sy, cx, cy)

        # Apply transform (preview)
        if self.apply_to_all:
            for roi in self._canvas.layer_stack.roi_layers:
                roi.mask[:] = apply_affine_to_mask(self._snapshot, M, roi.mask.shape)
        else:
            layer.mask[:] = apply_affine_to_mask(self._snapshot, M, layer.mask.shape)

        canvas.refresh_overlays()

    def on_release(self, pos, layer, canvas):
        if self._active_handle is None or self._snapshot is None or layer is None:
            return

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

        self._active_handle = None
        self._snapshot = None
        self._start_pos = None

        # Refresh handles to new position
        bbox = get_mask_bbox(layer.mask)
        if bbox:
            self._bbox = bbox
            self._show_handles(bbox, canvas)

    def _hit_test_handle(self, pos):
        if not self._bbox:
            return None
        handles = compute_handles(self._bbox)
        for h in handles:
            dist = math.hypot(pos.x() - h.x, pos.y() - h.y)
            if dist <= HANDLE_HIT_RADIUS:
                return h
        return None

    def _show_handles(self, bbox, canvas):
        self._clear_handles(canvas)
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

        # Handles
        handles = compute_handles(bbox)
        handle_pen = QPen(QColor(255, 255, 255), 1)
        handle_pen.setCosmetic(True)
        handle_brush = QBrush(QColor(0, 180, 255))
        rotate_brush = QBrush(QColor(255, 180, 0))

        for h in handles:
            brush = rotate_brush if h.handle_type == 'rotate' else handle_brush
            item = scene.addEllipse(
                QRectF(h.x - HANDLE_SIZE / 2, h.y - HANDLE_SIZE / 2,
                       HANDLE_SIZE, HANDLE_SIZE),
                handle_pen, brush
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

    def cursor(self):
        return Qt.SizeAllCursor
