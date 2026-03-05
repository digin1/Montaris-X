import numpy as np
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsEllipseItem, QLabel,
)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QTimer
from PySide6.QtGui import (
    QPixmap, QImage, QColor, QPainter, QPen, QPolygonF, QBrush,
)
from montaris.core.selection import SelectionModel


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
        self._roi_items = {}          # id(roi) -> QGraphicsPixmapItem (tight bbox)
        self._polygon_item = None
        self._brush_preview = None
        self._stamp_preview = None

        self._tool = None
        self._active_layer = None
        self._is_panning = False
        self._last_pan_pos = None
        self._space_held = False

        # Multi-selection
        self._selection = SelectionModel(self)
        self._selection_highlight_items = []
        self._selection.changed.connect(self._on_selection_changed)
        self.layer_stack.changed.connect(self._clean_stale_selection)

        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QColor(40, 40, 40))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # HUD overlay (E.23)
        self._hud_label = QLabel(self)
        self._hud_label.setStyleSheet(
            "QLabel { color: #ddd; background: rgba(0,0,0,150);"
            " padding: 2px 6px; font-size: 11px; }"
        )
        self._hud_label.move(6, 6)
        self._hud_label.setText("I: (-, -)  Z: 100%")
        self._hud_label.show()

        # Selection pulse timer (G.23)
        self._pulse_timer = None

        # Prevent re-entrant refresh
        self._refreshing = False

        # Rasterization progress bar (bottom of canvas)
        from PySide6.QtWidgets import QProgressBar
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setFixedHeight(3)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(
            "QProgressBar { background: transparent; border: none; }"
            "QProgressBar::chunk { background: #00b4ff; }"
        )
        self._progress_bar.hide()
        self._progress_hide_timer = QTimer(self)
        self._progress_hide_timer.setSingleShot(True)
        self._progress_hide_timer.timeout.connect(self._progress_bar.hide)

    # ------------------------------------------------------------------
    # Tool / layer management
    # ------------------------------------------------------------------

    def set_tool(self, tool):
        # Clean up old tool's scene items (e.g. transform handles)
        old = self._tool
        if old is not None and hasattr(old, '_clear_handles'):
            old._clear_handles(self)
        self._tool = tool
        self.hide_brush_preview()
        self._hide_stamp_preview()
        self._update_cursor()

    def set_active_layer(self, layer):
        self._active_layer = layer

    def _on_selection_changed(self, layers):
        """Sync _active_layer to primary selection and update highlights."""
        primary = self._selection.primary
        if primary is not None:
            self._active_layer = primary
        self._update_selection_highlights()
        self._pulse_selection()

    def _clean_stale_selection(self):
        """Remove layers from selection that are no longer in the layer stack."""
        current_rois = set(id(r) for r in self.layer_stack.roi_layers)
        stale = [l for l in self._selection.layers if id(l) not in current_rois]
        if stale:
            for l in stale:
                self._selection.remove(l)

    def _update_selection_highlights(self):
        """Draw actual ROI boundary outline for selected layers."""
        scene = self._scene
        for item in self._selection_highlight_items:
            scene.removeItem(item)
        self._selection_highlight_items.clear()

        for layer in self._selection.layers:
            if not hasattr(layer, 'mask'):
                continue
            bbox = layer.get_bbox()
            if bbox is None:
                continue
            y1, y2, x1, x2 = bbox
            bh, bw = y2 - y1, x2 - x1
            mask_crop = layer.mask[y1:y2, x1:x2]
            edge = _compute_edge(mask_crop)
            rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
            rgba[edge] = [255, 255, 0, 200]
            rgba = np.ascontiguousarray(rgba)
            qimg = QImage(rgba.data, bw, bh, bw * 4, QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimg)
            item = QGraphicsPixmapItem(pixmap)
            item.setOffset(x1, y1)
            item.setZValue(998)
            item.setAcceptedMouseButtons(Qt.NoButton)
            scene.addItem(item)
            self._selection_highlight_items.append(item)

    def _pulse_selection(self):
        """Brief opacity boost on selection change (G.23)."""
        if not self._selection_highlight_items:
            return
        # Flash brighter by replacing pixmaps temporarily
        for item in self._selection_highlight_items:
            item.setOpacity(1.0)

        def _restore():
            for item in self._selection_highlight_items:
                item.setOpacity(0.8)

        if self._pulse_timer is not None:
            self._pulse_timer.stop()
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setSingleShot(True)
        self._pulse_timer.timeout.connect(_restore)
        self._pulse_timer.start(200)

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
    # ROI overlay display — per-ROI QGraphicsPixmapItem (tight bbox)
    # ------------------------------------------------------------------

    def refresh_overlays(self):
        """Rebuild all ROI pixmap items from current mask/color state.

        Shows a progress bar for large batches and processes events
        to keep the UI responsive during rasterization.
        """
        if self._refreshing:
            return  # prevent re-entrant calls from signal cascades
        self._refreshing = True
        try:
            self._do_refresh_overlays()
        finally:
            self._refreshing = False

    def _do_refresh_overlays(self):
        """Internal: actual rebuild of all ROI pixmap items."""
        if self.layer_stack.image_layer is None:
            for item in self._roi_items.values():
                self._scene.removeItem(item)
            self._roi_items.clear()
            return

        gof = self.layer_stack._global_opacity_factor
        rois = self.layer_stack.roi_layers
        current_ids = {id(r) for r in rois}
        n = len(rois)

        # Remove stale items (deleted ROIs)
        for rid in list(self._roi_items.keys()):
            if rid not in current_ids:
                self._scene.removeItem(self._roi_items.pop(rid))

        # Show progress for large batches, process events to stay responsive
        show_progress = n > 5
        if show_progress:
            from PySide6.QtWidgets import QApplication
            self._progress_bar.setRange(0, n)
            self._progress_bar.setValue(0)
            self._progress_bar.show()

        for i, roi in enumerate(rois):
            self._refresh_roi_item(roi, i, gof)
            if show_progress:
                self._progress_bar.setValue(i + 1)
                if (i + 1) % 10 == 0:
                    QApplication.processEvents()

        if show_progress:
            self._progress_bar.hide()

    def flash_progress(self):
        """Show a brief progress flash to indicate rasterization."""
        self._progress_bar.setRange(0, 0)  # indeterminate mode
        self._progress_bar.show()
        self._progress_hide_timer.start(300)

    def refresh_overlays_lut_only(self):
        """Re-render all ROI pixmaps (for color/opacity changes)."""
        self.refresh_overlays()

    def refresh_active_overlay(self, layer):
        """Re-render only the specified ROI layer's pixmap item."""
        if layer is None or not hasattr(layer, 'mask'):
            return
        layer.clear_dirty()
        try:
            index = self.layer_stack.roi_layers.index(layer)
        except ValueError:
            return
        self._refresh_roi_item(layer, index)

    def _refresh_roi_item(self, roi, index, gof=None):
        """Create or update the QGraphicsPixmapItem for a single ROI."""
        if gof is None:
            gof = self.layer_stack._global_opacity_factor

        rid = id(roi)

        # Remove old item
        if rid in self._roi_items:
            self._scene.removeItem(self._roi_items.pop(rid))

        if not roi.visible:
            return

        bbox = roi.get_bbox()  # cached
        if bbox is None:
            return

        y1, y2, x1, x2 = bbox
        bh, bw = y2 - y1, x2 - x1
        r, g, b = roi.color
        effective_opacity = int(roi.opacity * gof)
        fill_mode = getattr(roi, 'fill_mode', 'solid')

        rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
        mask_crop = roi.mask[y1:y2, x1:x2]

        if fill_mode == 'outline':
            edge = _compute_edge(mask_crop)
            rgba[edge] = [r, g, b, effective_opacity]
        elif fill_mode == 'both':
            fill_alpha = max(1, effective_opacity // 2)
            painted = mask_crop > 0
            rgba[painted] = [r, g, b, fill_alpha]
            edge = _compute_edge(mask_crop)
            rgba[edge] = [r, g, b, min(255, effective_opacity)]
        else:
            painted = mask_crop > 0
            rgba[painted] = [r, g, b, effective_opacity]

        rgba = np.ascontiguousarray(rgba)
        qimg = QImage(rgba.data, bw, bh, bw * 4, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)

        item = QGraphicsPixmapItem(pixmap)
        item.setOffset(x1, y1)
        item.setZValue(1 + index * 0.001)
        item.setAcceptedMouseButtons(Qt.NoButton)
        self._scene.addItem(item)
        self._roi_items[rid] = item

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
        # Match brush preview to active ROI color (C.6)
        if self._active_layer and hasattr(self._active_layer, 'color'):
            r, g, b = self._active_layer.color
            pen = QPen(QColor(r, g, b, 200), 1)
        else:
            pen = QPen(QColor(255, 255, 255, 180), 1)
        pen.setCosmetic(True)
        self._brush_preview.setPen(pen)
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
    # Layout
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep progress bar at bottom
        w = self.viewport().width()
        h = self.viewport().height()
        self._progress_bar.setGeometry(0, h - 3, w, 3)

    # ------------------------------------------------------------------
    # Zoom helpers
    # ------------------------------------------------------------------

    def fit_to_window(self):
        if self._image_item:
            self.fitInView(self._image_item, Qt.KeepAspectRatio)

    def reset_zoom(self):
        self.resetTransform()

    def zoom_in(self):
        self.scale(1.25, 1.25)

    def zoom_out(self):
        self.scale(1 / 1.25, 1 / 1.25)

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

        handled = False
        if self._tool:
            handled = self._tool.on_key_press(event.key(), self) or False

        # Escape: clear selection (only if tool didn't consume the event)
        if event.key() == Qt.Key_Escape and not handled:
            self._selection.clear()
            return

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

        # Ctrl+click: toggle ROI selection (empty space clears)
        if (event.button() == Qt.LeftButton
                and event.modifiers() & Qt.ControlModifier):
            scene_pos = self.mapToScene(event.position().toPoint())
            hit = SelectionModel.hit_test(
                scene_pos.x(), scene_pos.y(), self.layer_stack.roi_layers
            )
            if hit is not None:
                self._selection.toggle(hit)
            else:
                self._selection.clear()
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
        # Update HUD (E.23, E.24)
        zoom = self.transform().m11()
        ds = getattr(self.parent(), '_downsample_factor', 1) if self.parent() else 1
        hud_text = f"I: ({ix}, {iy})  Z: {zoom:.0%}"
        if ds > 1:
            hud_text += f"  DS: {ds}x"
        self._hud_label.setText(hud_text)
        self._hud_label.adjustSize()

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
        # Stamp tool: show rect preview (C.20)
        if (self._tool and hasattr(self._tool, 'width')
                and hasattr(self._tool, 'height') and not self._is_panning):
            sw, sh = self._tool.width, self._tool.height
            self._show_stamp_preview(scene_pos.x(), scene_pos.y(), sw, sh)
            self.hide_brush_preview()
            return

        # Hide stamp preview if not stamp tool
        self._hide_stamp_preview()

        if self._tool and hasattr(self._tool, 'size') and not self._is_panning:
            zoom = self.transform().m11()
            if zoom > 0 and getattr(self._tool, 'zoom_compensated', False):
                radius = self._tool.size / zoom / 2
            else:
                radius = self._tool.size / 2
            self.show_brush_preview(scene_pos.x(), scene_pos.y(), radius)
        else:
            self.hide_brush_preview()

    def _show_stamp_preview(self, cx, cy, w, h):
        from PySide6.QtWidgets import QGraphicsRectItem
        if self._stamp_preview is None:
            self._stamp_preview = QGraphicsRectItem()
            pen = QPen(QColor(255, 255, 255, 180), 1)
            pen.setCosmetic(True)
            self._stamp_preview.setPen(pen)
            self._stamp_preview.setBrush(QBrush(Qt.NoBrush))
            self._stamp_preview.setZValue(2000)
            self._scene.addItem(self._stamp_preview)
        self._stamp_preview.setRect(cx - w / 2, cy - h / 2, w, h)
        self._stamp_preview.setVisible(True)

    def _hide_stamp_preview(self):
        if self._stamp_preview is not None:
            self._stamp_preview.setVisible(False)

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
            slider.setValue(max(1, min(500, slider.value() + delta)))


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


def _compute_edge(mask):
    """Return boolean edge array for mask > 0 pixels."""
    from scipy.ndimage import binary_erosion
    filled = mask > 0
    return filled ^ binary_erosion(filled)


def _composite_roi(combined, mask, color, opacity, fill_mode="solid"):
    """Composite a single ROI onto the combined RGBA array (in-place).

    Later ROIs paint over earlier ones where they have pixels.
    """
    r, g, b = color
    painted = mask > 0
    if not np.any(painted):
        return

    if fill_mode == "outline":
        edge = _compute_edge(mask)
        combined[edge] = [r, g, b, opacity]
    elif fill_mode == "both":
        fill_alpha = max(1, opacity // 2)
        combined[painted] = [r, g, b, fill_alpha]
        edge = _compute_edge(mask)
        combined[edge] = [r, g, b, min(255, opacity)]
    else:
        combined[painted] = [r, g, b, opacity]


def _composite_roi_region(region, mask, color, opacity, fill_mode,
                          x1, y1, x2, y2):
    """Composite a single ROI onto a sub-region of the combined array."""
    r, g, b = color
    mask_region = mask[y1:y2, x1:x2]
    painted = mask_region > 0
    if not np.any(painted):
        return

    if fill_mode == "outline":
        edge = _compute_edge(mask_region)
        region[edge] = [r, g, b, opacity]
    elif fill_mode == "both":
        fill_alpha = max(1, opacity // 2)
        region[painted] = [r, g, b, fill_alpha]
        edge = _compute_edge(mask_region)
        region[edge] = [r, g, b, min(255, opacity)]
    else:
        region[painted] = [r, g, b, opacity]


def _mask_to_rgba(mask, color, opacity=128, fill_mode="solid"):
    """Return an RGBA numpy array for the given mask and color."""
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    _composite_roi(overlay, mask, color, opacity, fill_mode)
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
