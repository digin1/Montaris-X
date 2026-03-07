import numpy as np
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsEllipseItem, QGraphicsItem, QLabel,
)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QTimer
from PySide6.QtGui import (
    QPixmap, QImage, QColor, QPainter, QPen, QPolygonF, QBrush, QPainterPath,
)
from montaris.core.selection import SelectionModel


# ------------------------------------------------------------------
# Mutable QImage overlay — avoids QPixmap COW and full-scene updates
# ------------------------------------------------------------------

class _ROIOverlayItem(QGraphicsItem):
    """Lightweight ROI overlay that paints from a mutable QImage.

    Unlike QGraphicsPixmapItem, QPainter on image() modifies in-place
    (no copy-on-write), and updateDirty() invalidates only the changed
    sub-region instead of the entire bounding rect.
    """
    def __init__(self):
        super().__init__()
        self._image = None
        self._rect = QRectF()
        self.setAcceptedMouseButtons(Qt.NoButton)

    def setImage(self, image):
        old = self._rect
        self._rect = QRectF(0, 0, image.width(), image.height())
        if old != self._rect:
            self.prepareGeometryChange()
        self._image = image
        self.update()

    def image(self):
        return self._image  # same object, no copy

    def boundingRect(self):
        return self._rect

    def paint(self, painter, option, widget=None):
        if self._image:
            exposed = option.exposedRect.toAlignedRect() & self._image.rect()
            if not exposed.isEmpty():
                painter.drawImage(exposed, self._image, exposed)

    def updateDirty(self, rect):
        """Invalidate only the dirty sub-region."""
        self.update(rect)


# ------------------------------------------------------------------
# Canvas
# ------------------------------------------------------------------

class ImageCanvas(QGraphicsView):
    cursor_moved = Signal(int, int, str)
    viewport_changed = Signal()

    def __init__(self, layer_stack, parent=None):
        super().__init__(parent)
        self.layer_stack = layer_stack
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._image_item = None       # QGraphicsPixmapItem for the image
        self._roi_items = {}          # id(roi) -> _ROIOverlayItem (tight bbox)
        self._polygon_item = None
        self._brush_preview = None
        self._stamp_preview = None

        # Throttled partial refresh for large brush strokes
        self._pending_dirty = {}  # id(layer) -> (layer, (y1, y2, x1, x2))
        self._dirty_timer = QTimer(self)
        self._dirty_timer.setSingleShot(True)
        self._dirty_timer.setInterval(16)  # ~60fps max
        self._dirty_timer.timeout.connect(self._flush_dirty)

        self._tint_color = None
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

        # LOD / viewport culling state
        self._roi_stale = set()       # ROI ids needing rasterization when visible
        self._roi_lod = {}            # id(roi) -> current LOD level
        self._roi_lod_pending = set() # ROI ids queued for LOD re-rasterization
        self._last_lod_level = 0
        self._lod_timer = None        # debounce timer for viewport changes
        self.viewport_changed.connect(self._schedule_viewport_lod_check)

        # Rasterization progress bar (bottom of canvas)
        from PySide6.QtWidgets import QProgressBar
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setFixedHeight(4)
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

    _PAINT_TOOLS = {'Brush', 'Eraser', 'Polygon', 'Bucket Fill',
                     'Rectangle', 'Circle', 'Stamp', 'Transform'}

    def set_tool(self, tool):
        # Clean up old tool's scene items (e.g. transform handles)
        old = self._tool
        if old is not None and hasattr(old, '_clear_handles'):
            old._clear_handles(self)
        # Flatten offsets when switching to paint/transform tools
        if tool is not None and getattr(tool, 'name', None) in self._PAINT_TOOLS:
            self._flatten_all_offsets()
        self._tool = tool
        self.hide_brush_preview()
        self._hide_stamp_preview()
        self._update_cursor()
        if tool is not None and hasattr(tool, 'on_activate'):
            tool.on_activate(self._active_layer, self)

    def _flatten_all_offsets(self):
        """Bake any non-zero layer offsets into masks and clear undo stack."""
        any_offset = False
        partially_clipped = []
        for roi in self.layer_stack.roi_layers:
            if roi.offset_x != 0 or roi.offset_y != 0:
                if roi.has_oob_content():
                    partially_clipped.append(roi.name)
                result = roi.flatten_offset()
                if result:
                    any_offset = True
                # result=False means fully OOB — offset preserved
        if partially_clipped:
            win = self.window()
            if hasattr(win, 'toast'):
                names = ', '.join(partially_clipped[:5])
                if len(partially_clipped) > 5:
                    names += f' (+{len(partially_clipped) - 5} more)'
                win.toast.show(
                    f"OOB pixels clipped: {names}",
                    level="warning"
                )
        if any_offset:
            # Offset undo commands are now stale — clear the stack
            win = self.window()
            if hasattr(win, 'undo_stack'):
                win.undo_stack.clear()
            self.refresh_overlays()

    def set_active_layer(self, layer):
        if layer is not self._active_layer:
            # Clear transform/move handles when switching layers
            tool = self._tool
            if tool is not None and hasattr(tool, '_clear_handles'):
                tool._clear_handles(self)
            self._active_layer = layer
            # Re-show bbox for new layer if tool supports it
            if tool is not None and hasattr(tool, 'on_activate'):
                tool.on_activate(layer, self)
        else:
            self._active_layer = layer

    def _on_selection_changed(self, layers):
        """Sync _active_layer to primary selection and update highlights."""
        primary = self._selection.primary
        old = self._active_layer
        if primary is not None:
            self._active_layer = primary
        self._update_selection_highlights()
        self._pulse_selection()
        # Notify tool of layer/selection change so it can refresh visuals
        tool = self._tool
        if tool is not None and hasattr(tool, 'on_activate'):
            if primary is not old:
                # Layer changed: clear old state, show for new
                if hasattr(tool, '_clear_handles'):
                    tool._clear_handles(self)
                tool.on_activate(primary, self)
            elif primary is not None:
                # Same layer but selection changed: refresh
                tool.on_activate(primary, self)

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
            disp_x = x1 + layer.offset_x
            disp_y = y1 + layer.offset_y
            item.setOffset(disp_x, disp_y)
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

    def set_tint_color(self, tint_color):
        """Set the display tint for the background image (None for grayscale)."""
        self._tint_color = tint_color

    def refresh_image(self):
        if self._image_item:
            self._scene.removeItem(self._image_item)
            self._image_item = None

        img_layer = self.layer_stack.image_layer
        if img_layer is None:
            return

        data = img_layer.data
        tint = getattr(self, '_tint_color', None)
        if tint is not None and data.ndim == 2:
            data = _apply_tint(data, tint)

        qimg = numpy_to_qimage(data)
        pixmap = QPixmap.fromImage(qimg)
        self._image_item = QGraphicsPixmapItem(pixmap)
        self._image_item.setZValue(0)
        self._scene.addItem(self._image_item)
        # Scene rect with modest margin for comfortable panning
        h, w = img_layer.data.shape[:2]
        m = min(w, h) // 4  # 25% margin
        self._scene.setSceneRect(QRectF(-m, -m, w + 2 * m, h + 2 * m))

    def refresh_image_from_array(self, data):
        """Display an arbitrary numpy array as the background image."""
        if self._image_item:
            self._scene.removeItem(self._image_item)
            self._image_item = None
        qimg = numpy_to_qimage(data)
        pixmap = QPixmap.fromImage(qimg)
        self._image_item = QGraphicsPixmapItem(pixmap)
        self._image_item.setZValue(0)
        self._scene.addItem(self._image_item)
        h, w = data.shape[:2]
        m = min(w, h) // 4
        self._scene.setSceneRect(QRectF(-m, -m, w + 2 * m, h + 2 * m))

    # ------------------------------------------------------------------
    # ROI overlay display — per-ROI _ROIOverlayItem (tight bbox)
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
        # Cancel any pending partial dirty — full rebuild supersedes
        self._pending_dirty.clear()
        if self.layer_stack.image_layer is None:
            for item in self._roi_items.values():
                self._scene.removeItem(item)
            self._roi_items.clear()
            self._roi_stale.clear()
            self._roi_lod.clear()
            self._roi_lod_pending.clear()
            return

        gof = self.layer_stack._global_opacity_factor
        rois = self.layer_stack.roi_layers
        current_ids = {id(r) for r in rois}
        n = len(rois)

        # Remove stale items (deleted ROIs)
        for rid in list(self._roi_items.keys()):
            if rid not in current_ids:
                self._scene.removeItem(self._roi_items.pop(rid))
        # Clean tracking dicts for deleted ROIs
        self._roi_stale &= current_ids
        self._roi_lod_pending.clear()  # full refresh makes pending obsolete
        for rid in list(self._roi_lod):
            if rid not in current_ids:
                del self._roi_lod[rid]

        # Viewport culling: skip off-screen ROIs
        vp_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        use_culling = vp_rect.width() > 0 and vp_rect.height() > 0
        lod = self._current_lod_level() if use_culling else 0
        self._last_lod_level = lod

        # Show progress for large batches, process events to stay responsive
        show_progress = n > 5
        if show_progress:
            from PySide6.QtWidgets import QApplication
            self._progress_bar.setRange(0, n)
            self._progress_bar.setValue(0)
            self._progress_bar.show()

        for i, roi in enumerate(rois):
            rid = id(roi)
            roi.invalidate_bbox()
            # Viewport cull non-active ROIs
            if use_culling and roi != self._active_layer:
                dbbox = roi.get_display_bbox()
                if dbbox is not None:
                    y1, y2, x1, x2 = dbbox
                    if not vp_rect.intersects(QRectF(x1, y1, x2 - x1, y2 - y1)):
                        self._roi_stale.add(rid)
                        if show_progress:
                            self._progress_bar.setValue(i + 1)
                        continue

            self._roi_stale.discard(rid)
            target_lod = 0 if roi == self._active_layer else lod
            self._refresh_roi_item(roi, i, gof, lod_level=target_lod)
            self._roi_lod[rid] = target_lod
            if show_progress:
                self._progress_bar.setValue(i + 1)
                if (i + 1) % 10 == 0:
                    QApplication.processEvents()

        if show_progress:
            self._progress_bar.hide()

        self._update_selection_highlights()

    def flash_progress(self, message=None):
        """Show a brief progress flash to indicate rasterization.

        If *message* is given, also show it in the main window statusbar.
        """
        self._progress_bar.setRange(0, 0)  # indeterminate mode
        self._progress_bar.show()
        self._progress_hide_timer.start(400)
        if message:
            win = self.window()
            if hasattr(win, 'statusbar'):
                win.statusbar.showMessage(message, 1000)

    def refresh_overlays_lut_only(self):
        """Re-render all ROI pixmaps (for color/opacity changes)."""
        self.refresh_overlays()

    def refresh_active_overlay(self, layer):
        """Re-render only the specified ROI layer's pixmap item."""
        if layer is None or not hasattr(layer, 'mask'):
            return
        layer.invalidate_bbox()
        layer.clear_dirty()
        try:
            index = self.layer_stack.roi_layers.index(layer)
        except ValueError:
            return
        rid = id(layer)
        self._roi_stale.discard(rid)
        # Cancel any pending partial dirty for this layer — full rebuild supersedes
        self._pending_dirty.pop(id(layer), None)
        self._refresh_roi_item(layer, index)
        self._roi_lod[rid] = 0
        if layer in self._selection.layers:
            self._update_selection_highlights()

    def refresh_active_overlay_partial(self, layer, dirty_bbox):
        """Throttled partial refresh: accumulate dirty region, render on timer."""
        if layer is None or not hasattr(layer, 'mask'):
            return
        self._roi_stale.discard(id(layer))

        dy1, dy2, dx1, dx2 = dirty_bbox
        if dy2 <= dy1 or dx2 <= dx1:
            return

        lid = id(layer)
        if lid in self._pending_dirty:
            _, old = self._pending_dirty[lid]
            oy1, oy2, ox1, ox2 = old
            self._pending_dirty[lid] = (layer, (min(oy1, dy1), max(oy2, dy2),
                                                 min(ox1, dx1), max(ox2, dx2)))
        else:
            self._pending_dirty[lid] = (layer, dirty_bbox)

        if not self._dirty_timer.isActive():
            self._dirty_timer.start()

    def _flush_dirty(self):
        """Render all accumulated dirty regions."""
        pending = self._pending_dirty
        self._pending_dirty = {}
        lod = self._current_lod_level()
        for lid, (layer, bbox) in pending.items():
            self._render_dirty_region(layer, bbox, lod_level=lod)

    def _render_dirty_region(self, layer, dirty_bbox, lod_level=0):
        """Render a single dirty region onto the layer's pixmap item."""
        dy1, dy2, dx1, dx2 = dirty_bbox
        dh, dw = dy2 - dy1, dx2 - dx1
        if dh <= 0 or dw <= 0:
            return

        rid = id(layer)
        existing = self._roi_items.get(rid)

        r, g, b = layer.color
        gof = self.layer_stack._global_opacity_factor
        alpha = int(layer.opacity * gof)

        # Build RGBA tile for dirty region from actual mask data
        mask_crop = layer.mask[dy1:dy2, dx1:dx2]

        # LOD downsampling during stroke for performance
        scale_factor = 1
        if lod_level > 0:
            factor = 1 << lod_level
            th = (dh // factor) * factor
            tw = (dw // factor) * factor
            if th > 0 and tw > 0:
                mask_crop = mask_crop[:th, :tw].reshape(
                    th // factor, factor, tw // factor, factor
                ).max(axis=(1, 3))
                dh, dw = th // factor, tw // factor
                scale_factor = factor

        rgba = np.zeros((dh, dw, 4), dtype=np.uint8)
        painted = mask_crop > 0
        rgba[painted] = [r, g, b, alpha]
        rgba = np.ascontiguousarray(rgba)
        qimg = QImage(rgba.data, dw, dh, dw * 4, QImage.Format_RGBA8888)
        dirty_img = qimg.copy()  # detach from numpy buffer

        # Scale back up if downsampled
        if scale_factor > 1:
            full_img = QImage(dx2 - dx1, dy2 - dy1, QImage.Format_RGBA8888)
            full_img.fill(QColor(0, 0, 0, 0))
            p = QPainter(full_img)
            target = QRectF(0, 0, full_img.width(), full_img.height())
            p.drawImage(target, dirty_img)
            p.end()
            dirty_img = full_img

        if existing is None:
            # First paint on empty ROI — create item from dirty region
            try:
                index = self.layer_stack.roi_layers.index(layer)
            except ValueError:
                return
            item = _ROIOverlayItem()
            item.setImage(dirty_img)
            item.setPos(dx1, dy1)
            item.setZValue(1 + index * 0.001)
            self._scene.addItem(item)
            self._roi_items[rid] = item
            return

        image = existing.image()
        ox, oy = int(existing.pos().x()), int(existing.pos().y())
        pw, ph = image.width(), image.height()

        # Expand image if dirty region extends beyond current bounds
        px1, py1 = ox, oy
        px2, py2 = ox + pw, oy + ph
        nx1, ny1 = min(px1, dx1), min(py1, dy1)
        nx2, ny2 = max(px2, dx2), max(py2, dy2)

        if nx1 < px1 or ny1 < py1 or nx2 > px2 or ny2 > py2:
            new_img = QImage(nx2 - nx1, ny2 - ny1, QImage.Format_RGBA8888)
            new_img.fill(QColor(0, 0, 0, 0))
            p = QPainter(new_img)
            p.drawImage(px1 - nx1, py1 - ny1, image)
            p.end()
            image = new_img
            existing.setImage(image)
            existing.setPos(nx1, ny1)
            ox, oy = nx1, ny1

        # Composite dirty tile onto image — in-place, no COW
        p = QPainter(image)
        p.setCompositionMode(QPainter.CompositionMode_Source)
        p.drawImage(dx1 - ox, dy1 - oy, dirty_img)
        p.end()
        existing.updateDirty(QRectF(dx1 - ox, dy1 - oy,
                                    dirty_img.width(), dirty_img.height()))

    def _refresh_roi_item(self, roi, index, gof=None, lod_level=0):
        """Create or update the _ROIOverlayItem for a single ROI."""
        if gof is None:
            gof = self.layer_stack._global_opacity_factor

        rid = id(roi)
        existing = self._roi_items.get(rid)

        if not roi.visible:
            if existing is not None:
                self._scene.removeItem(self._roi_items.pop(rid))
            return

        bbox = roi.get_bbox()  # cached
        if bbox is None:
            if existing is not None:
                self._scene.removeItem(self._roi_items.pop(rid))
            return

        y1, y2, x1, x2 = bbox
        bh, bw = y2 - y1, x2 - x1
        r, g, b = roi.color
        effective_opacity = int(roi.opacity * gof)
        fill_mode = getattr(roi, 'fill_mode', 'solid')

        mask_crop = roi.mask[y1:y2, x1:x2]

        # LOD downsampling: max-pool mask then scale up in Qt
        scale_factor = 1
        if lod_level > 0:
            factor = 1 << lod_level
            th = (bh // factor) * factor
            tw = (bw // factor) * factor
            if th > 0 and tw > 0:
                mask_crop = mask_crop[:th, :tw].reshape(
                    th // factor, factor, tw // factor, factor
                ).max(axis=(1, 3))
                bh, bw = th // factor, tw // factor
                scale_factor = factor

        rgba = np.zeros((bh, bw, 4), dtype=np.uint8)

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

        disp_x = x1 + roi.offset_x
        disp_y = y1 + roi.offset_y
        if existing is not None:
            existing.setImage(qimg.copy())
            existing.setPos(disp_x, disp_y)
            existing.setZValue(1 + index * 0.001)
        else:
            item = _ROIOverlayItem()
            item.setImage(qimg.copy())
            item.setPos(disp_x, disp_y)
            item.setZValue(1 + index * 0.001)
            self._scene.addItem(item)
            self._roi_items[rid] = item
            existing = item

        if scale_factor > 1:
            existing.setScale(scale_factor)
        else:
            existing.setScale(1)

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
        pen = QPen(QColor(255, 255, 0), 1.5)
        pen.setCosmetic(True)
        # Draw open polyline (not closed polygon) so it doesn't auto-close
        path = QPainterPath()
        path.moveTo(QPointF(vertices[0][0], vertices[0][1]))
        for x, y in vertices[1:]:
            path.lineTo(QPointF(x, y))
        if hover_point:
            path.lineTo(QPointF(hover_point[0], hover_point[1]))
        self._polygon_item = self._scene.addPath(path, pen)
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
        self._progress_bar.setGeometry(0, h - 4, w, 4)

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
    # LOD / viewport culling
    # ------------------------------------------------------------------

    def _current_lod_level(self):
        """LOD level from zoom: 0=full, 1=half, 2=quarter, 3=eighth."""
        m = abs(self.transform().m11())
        if m >= 0.5:
            return 0
        if m >= 0.25:
            return 1
        if m >= 0.125:
            return 2
        return 3

    def _schedule_viewport_lod_check(self):
        """Debounced check for stale ROIs entering viewport or LOD change."""
        if self._lod_timer is None:
            self._lod_timer = QTimer(self)
            self._lod_timer.setSingleShot(True)
            self._lod_timer.timeout.connect(self._on_viewport_changed_lod)
        self._lod_timer.start(50)

    _LOD_BATCH_SIZE = 6

    def _on_viewport_changed_lod(self):
        """Rasterize stale/pending ROIs in small batches to avoid UI hitches."""
        # Skip LOD updates while a tool is actively dragging (transform/move preview)
        tool = self._tool
        if tool and (getattr(tool, '_dragging', False) or getattr(tool, '_moving', False)):
            return

        lod = self._current_lod_level()
        lod_changed = lod != self._last_lod_level
        self._last_lod_level = lod

        if not self._roi_stale and not lod_changed and not self._roi_lod_pending:
            return

        vp_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        if vp_rect.width() <= 0 or vp_rect.height() <= 0:
            return

        gof = self.layer_stack._global_opacity_factor
        rois = self.layer_stack.roi_layers

        # On LOD change, queue visible ROIs at wrong LOD for update
        if lod_changed:
            for roi in rois:
                rid = id(roi)
                target_lod = 0 if roi == self._active_layer else lod
                if self._roi_lod.get(rid) != target_lod and rid not in self._roi_stale:
                    self._roi_lod_pending.add(rid)

        # Process stale entries + LOD pending in a single batched pass
        budget = self._LOD_BATCH_SIZE
        needs_more = False

        for i, roi in enumerate(rois):
            rid = id(roi)
            target_lod = 0 if roi == self._active_layer else lod

            is_stale = rid in self._roi_stale
            is_pending = rid in self._roi_lod_pending

            if not is_stale and not is_pending:
                continue

            # Stale items need a viewport check first
            if is_stale:
                dbbox = roi.get_display_bbox()
                if dbbox is None:
                    self._roi_stale.discard(rid)
                    continue
                y1, y2, x1, x2 = dbbox
                if not vp_rect.intersects(QRectF(x1, y1, x2 - x1, y2 - y1)):
                    continue  # off-screen, stays in _roi_stale

            if budget <= 0:
                needs_more = True
                continue

            # Rasterize
            self._roi_stale.discard(rid)
            self._roi_lod_pending.discard(rid)
            self._refresh_roi_item(roi, i, gof, lod_level=target_lod)
            self._roi_lod[rid] = target_lod
            budget -= 1

        # Schedule next batch if work remains
        if needs_more:
            self._lod_timer.start(16)

    # ------------------------------------------------------------------
    # Event overrides
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        self.viewport_changed.emit()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self.viewport_changed.emit()

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
            # Auto-create ROI layer if painting with no active layer
            if (self._active_layer is None
                    and getattr(self._tool, 'name', None) in self._PAINT_TOOLS
                    and self.layer_stack.image_layer is not None):
                win = self.window()
                if hasattr(win, '_on_roi_added'):
                    win._on_roi_added()
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
            slider.setValue(max(1, min(2000, slider.value() + delta)))


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


def _apply_tint(array, tint_color):
    """Apply an (R, G, B) tint to a 2D grayscale array, returning an (H, W, 3) uint8 array."""
    if array.dtype != np.uint8:
        mn, mx = float(array.min()), float(array.max())
        if mx > mn:
            gray = ((array.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            gray = np.zeros_like(array, dtype=np.uint8)
    else:
        gray = array
    r, g, b = tint_color
    rgb = np.empty((*gray.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = (gray.astype(np.uint16) * r // 255).astype(np.uint8)
    rgb[:, :, 1] = (gray.astype(np.uint16) * g // 255).astype(np.uint8)
    rgb[:, :, 2] = (gray.astype(np.uint16) * b // 255).astype(np.uint8)
    return rgb


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
