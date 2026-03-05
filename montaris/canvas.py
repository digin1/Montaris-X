import numpy as np
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsEllipseItem, QGraphicsItem,
)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import (
    QPixmap, QImage, QColor, QPainter, QPen, QPolygonF, QBrush,
)


# ------------------------------------------------------------------
# ROI overlay item — paints from a numpy-backed QImage (no QPixmap COW)
# ------------------------------------------------------------------

class ROIOverlayItem(QGraphicsItem):
    """Scene item that renders an RGBA overlay from a numpy array.

    The numpy array is modified in-place by the tools and the QImage
    wrapping it reflects the changes immediately (shared memory).
    Calling ``update_region()`` schedules a repaint of just the dirty
    rectangle, which is very cheap.
    """

    def __init__(self, rgba_array, parent=None):
        super().__init__(parent)
        self._rgba = rgba_array
        h, w = rgba_array.shape[:2]
        self._w = w
        self._h = h
        self._image = QImage(rgba_array.data, w, h, w * 4,
                             QImage.Format_RGBA8888)

    def set_rgba(self, rgba_array):
        h, w = rgba_array.shape[:2]
        self._rgba = rgba_array
        self._w = w
        self._h = h
        self._image = QImage(rgba_array.data, w, h, w * 4,
                             QImage.Format_RGBA8888)
        self.update()

    @property
    def rgba(self):
        return self._rgba

    def boundingRect(self):
        return QRectF(0, 0, self._w, self._h)

    def paint(self, painter, option, widget=None):
        painter.drawImage(0, 0, self._image)

    def update_region(self, x, y, w, h):
        self.update(QRectF(x, y, w, h))


# ------------------------------------------------------------------
# Canvas
# ------------------------------------------------------------------

class ImageCanvas(QGraphicsView):
    cursor_moved = Signal(int, int, str)

    def __init__(self, layer_stack, parent=None):
        super().__init__(parent)
        self.layer_stack = layer_stack
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._image_item = None       # QGraphicsPixmapItem for the image
        self._roi_items = {}          # id(roi) -> ROIOverlayItem
        self._polygon_item = None
        self._brush_preview = None

        self._tool = None
        self._active_layer = None
        self._is_panning = False
        self._last_pan_pos = None
        self._space_held = False

        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QColor(40, 40, 40))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------------
    # Tool / layer management
    # ------------------------------------------------------------------

    def set_tool(self, tool):
        self._tool = tool
        self.hide_brush_preview()
        self._update_cursor()

    def set_active_layer(self, layer):
        self._active_layer = layer

    # ------------------------------------------------------------------
    # Image display — single QGraphicsPixmapItem (no tile pyramid)
    # ------------------------------------------------------------------

    def refresh_image(self):
        if self._image_item:
            self._scene.removeItem(self._image_item)
            self._image_item = None

        img_layer = self.layer_stack.image_layer
        if img_layer is None:
            return

        qimg = numpy_to_qimage(img_layer.data)
        pixmap = QPixmap.fromImage(qimg)
        self._image_item = QGraphicsPixmapItem(pixmap)
        self._image_item.setZValue(0)
        self._scene.addItem(self._image_item)
        # Scene rect larger than image to allow free panning
        h, w = img_layer.data.shape[:2]
        self._scene.setSceneRect(QRectF(-w, -h, w * 3, h * 3))

    # ------------------------------------------------------------------
    # ROI overlay display
    # ------------------------------------------------------------------

    def refresh_overlays(self):
        for item in self._roi_items.values():
            self._scene.removeItem(item)
        self._roi_items.clear()

        for i, roi in enumerate(self.layer_stack.roi_layers):
            if not roi.visible:
                continue
            fill_mode = getattr(roi, 'fill_mode', 'solid')
            rgba = _mask_to_rgba(roi.mask, roi.color, roi.opacity, fill_mode)
            item = ROIOverlayItem(rgba)
            item.setZValue(i + 1)
            self._scene.addItem(item)
            self._roi_items[id(roi)] = item

    def refresh_active_overlay(self, layer):
        """Update only the specified ROI layer's overlay.

        Uses dirty-rect tracking to modify only changed pixels in the
        numpy-backed RGBA array — no full-image conversion needed.
        """
        if layer is None or not hasattr(layer, 'mask'):
            return
        if not layer.visible:
            roi_id = id(layer)
            if roi_id in self._roi_items:
                self._scene.removeItem(self._roi_items.pop(roi_id))
            return

        roi_id = id(layer)
        dirty = layer.dirty_rect
        fill_mode = getattr(layer, 'fill_mode', 'solid')

        # Fast path: update dirty rect directly in the numpy RGBA array
        if (dirty is not None
                and roi_id in self._roi_items
                and fill_mode == "solid"):
            item = self._roi_items[roi_id]
            rgba = item.rgba
            dx, dy, dw, dh = dirty
            h, w = layer.mask.shape
            x1, y1 = max(0, dx), max(0, dy)
            x2 = min(w, dx + dw)
            y2 = min(h, dy + dh)
            if x2 > x1 and y2 > y1:
                r, g, b = layer.color
                region = rgba[y1:y2, x1:x2]
                mask_region = layer.mask[y1:y2, x1:x2]
                region[:] = 0
                region[mask_region > 0] = [r, g, b, layer.opacity]
                item.update_region(x1, y1, x2 - x1, y2 - y1)
            layer.clear_dirty()
            return

        # Full refresh (first time, outline mode, or no dirty rect)
        rgba = _mask_to_rgba(layer.mask, layer.color, layer.opacity, fill_mode)

        if roi_id in self._roi_items:
            self._roi_items[roi_id].set_rgba(rgba)
        else:
            item = ROIOverlayItem(rgba)
            try:
                idx = self.layer_stack.roi_layers.index(layer)
            except ValueError:
                idx = 0
            item.setZValue(idx + 1)
            self._scene.addItem(item)
            self._roi_items[roi_id] = item
        layer.clear_dirty()

    # ------------------------------------------------------------------
    # Brush cursor preview
    # ------------------------------------------------------------------

    def show_brush_preview(self, cx, cy, radius):
        if self._brush_preview is None:
            self._brush_preview = QGraphicsEllipseItem()
            pen = QPen(QColor(255, 255, 255, 180), 1)
            pen.setCosmetic(True)
            self._brush_preview.setPen(pen)
            self._brush_preview.setBrush(QBrush(Qt.NoBrush))
            self._brush_preview.setZValue(2000)
            self._scene.addItem(self._brush_preview)
        self._brush_preview.setRect(cx - radius, cy - radius,
                                    radius * 2, radius * 2)
        self._brush_preview.setVisible(True)

    def hide_brush_preview(self):
        if self._brush_preview is not None:
            self._brush_preview.setVisible(False)

    # ------------------------------------------------------------------
    # Polygon preview
    # ------------------------------------------------------------------

    def draw_polygon_preview(self, vertices, hover_point=None):
        self.clear_polygon_preview()
        if len(vertices) < 1:
            return
        points = [QPointF(x, y) for x, y in vertices]
        if hover_point:
            points.append(QPointF(hover_point[0], hover_point[1]))
        polygon = QPolygonF(points)
        pen = QPen(QColor(255, 255, 0), 1.5)
        pen.setCosmetic(True)
        self._polygon_item = self._scene.addPolygon(polygon, pen)
        self._polygon_item.setZValue(1000)

    def clear_polygon_preview(self):
        if self._polygon_item:
            self._scene.removeItem(self._polygon_item)
            self._polygon_item = None

    # ------------------------------------------------------------------
    # Zoom helpers
    # ------------------------------------------------------------------

    def fit_to_window(self):
        if self._image_item:
            self.fitInView(self._image_item, Qt.KeepAspectRatio)

    def reset_zoom(self):
        self.resetTransform()

    # ------------------------------------------------------------------
    # Event overrides
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._space_held = True
            self.setCursor(Qt.OpenHandCursor)
            return

        if self._tool:
            self._tool.on_key_press(event.key(), self)

        if event.key() == Qt.Key_BracketLeft:
            self._adjust_brush_size(-2)
        elif event.key() == Qt.Key_BracketRight:
            self._adjust_brush_size(2)

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._space_held = False
            self._update_cursor()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event):
        if (event.button() in (Qt.MiddleButton, Qt.RightButton)
                or (self._space_held and event.button() == Qt.LeftButton)
                or (self._tool and getattr(self._tool, 'is_hand', False)
                    and event.button() == Qt.LeftButton)):
            self._is_panning = True
            self._last_pan_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if self._tool and event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._tool.on_press(scene_pos, self._active_layer, self)
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        ix, iy = int(scene_pos.x()), int(scene_pos.y())
        img = self.layer_stack.image_layer
        if img and 0 <= ix < img.data.shape[1] and 0 <= iy < img.data.shape[0]:
            val = img.data[iy, ix]
            self.cursor_moved.emit(ix, iy, str(val))

        self._update_brush_cursor(scene_pos)

        if self._is_panning:
            delta = event.position() - self._last_pan_pos
            self._last_pan_pos = event.position()
            hs = self.horizontalScrollBar()
            vs = self.verticalScrollBar()
            hs.setValue(hs.value() - int(delta.x()))
            vs.setValue(vs.value() - int(delta.y()))
            return

        if self._tool and event.buttons() & Qt.LeftButton:
            self._tool.on_move(scene_pos, self._active_layer, self)
            return

        if self._tool and not (event.buttons() & Qt.LeftButton):
            self._tool.on_move(scene_pos, self._active_layer, self)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._is_panning and event.button() in (
            Qt.MiddleButton, Qt.RightButton, Qt.LeftButton
        ):
            self._is_panning = False
            self._update_cursor()
            return

        if self._tool and event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._tool.on_release(scene_pos, self._active_layer, self)
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._tool and event.button() == Qt.LeftButton:
            if hasattr(self._tool, 'finish'):
                self._tool.finish()
            return
        super().mouseDoubleClickEvent(event)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_brush_cursor(self, scene_pos):
        if self._tool and hasattr(self._tool, 'size') and not self._is_panning:
            zoom = self.transform().m11()
            if zoom > 0 and getattr(self._tool, 'zoom_compensated', False):
                radius = self._tool.size / zoom / 2
            else:
                radius = self._tool.size / 2
            self.show_brush_preview(scene_pos.x(), scene_pos.y(), radius)
        else:
            self.hide_brush_preview()

    def _update_cursor(self):
        if self._space_held:
            self.setCursor(Qt.OpenHandCursor)
        elif self._tool:
            if getattr(self._tool, 'is_hand', False):
                self.setCursor(self._tool.cursor())
            elif hasattr(self._tool, 'size'):
                self.setCursor(Qt.BlankCursor)
            else:
                self.setCursor(self._tool.cursor())
        else:
            self.setCursor(Qt.CrossCursor)

    def _adjust_brush_size(self, delta):
        main_win = self.parent()
        if main_win and hasattr(main_win, 'tool_panel'):
            slider = main_win.tool_panel.size_slider
            slider.setValue(max(1, min(100, slider.value() + delta)))


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def numpy_to_qimage(array):
    if array.ndim == 2:
        h, w = array.shape
        if array.dtype != np.uint8:
            mn, mx = float(array.min()), float(array.max())
            if mx > mn:
                arr = ((array.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                arr = np.zeros_like(array, dtype=np.uint8)
        else:
            arr = np.ascontiguousarray(array)
        img = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
        return img.copy()

    elif array.ndim == 3:
        h, w, c = array.shape
        if array.dtype != np.uint8:
            mn, mx = float(array.min()), float(array.max())
            if mx > mn:
                arr = ((array.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                arr = np.zeros((h, w, c), dtype=np.uint8)
        else:
            arr = np.ascontiguousarray(array)

        if c == 3:
            img = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
            return img.copy()
        elif c == 4:
            img = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888)
            return img.copy()
        elif c == 1:
            arr = arr[:, :, 0]
            img = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
            return img.copy()

    raise ValueError(f"Unsupported array shape: {array.shape}")


def _mask_to_rgba(mask, color, opacity=128, fill_mode="solid"):
    """Return an RGBA numpy array for the given mask and color."""
    if fill_mode == "outline":
        h, w = mask.shape
        filled = mask > 0
        edge = np.zeros_like(filled)
        edge[0, :] |= filled[0, :]
        edge[1:, :] |= filled[1:, :] & ~filled[:-1, :]
        edge[-1, :] |= filled[-1, :]
        edge[:-1, :] |= filled[:-1, :] & ~filled[1:, :]
        edge[:, 0] |= filled[:, 0]
        edge[:, 1:] |= filled[:, 1:] & ~filled[:, :-1]
        edge[:, -1] |= filled[:, -1]
        edge[:, :-1] |= filled[:, :-1] & ~filled[:, 1:]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        r, g, b = color
        overlay[edge] = [r, g, b, opacity]
        return np.ascontiguousarray(overlay)

    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    r, g, b = color
    overlay[mask > 0] = [r, g, b, opacity]
    return np.ascontiguousarray(overlay)


def mask_to_qimage(mask, color, opacity=128):
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    r, g, b = color
    overlay[mask > 0] = [r, g, b, opacity]
    img = QImage(overlay.data, w, h, w * 4, QImage.Format_RGBA8888)
    return img.copy()


def mask_to_outline_qimage(mask, color, opacity=128):
    h, w = mask.shape
    filled = mask > 0
    edge = np.zeros_like(filled)
    edge[0, :] |= filled[0, :]
    edge[1:, :] |= filled[1:, :] & ~filled[:-1, :]
    edge[-1, :] |= filled[-1, :]
    edge[:-1, :] |= filled[:-1, :] & ~filled[1:, :]
    edge[:, 0] |= filled[:, 0]
    edge[:, 1:] |= filled[:, 1:] & ~filled[:, :-1]
    edge[:, -1] |= filled[:, -1]
    edge[:, :-1] |= filled[:, :-1] & ~filled[:, 1:]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    r, g, b = color
    overlay[edge] = [r, g, b, opacity]
    img = QImage(overlay.data, w, h, w * 4, QImage.Format_RGBA8888)
    return img.copy()
