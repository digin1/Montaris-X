import time
import numpy as np
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsEllipseItem, QGraphicsItem, QLabel, QWidget,
    QHBoxLayout, QVBoxLayout, QPushButton,
)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QTimer, QTimeLine
from PySide6.QtGui import (
    QPixmap, QImage, QColor, QPainter, QPen, QPolygonF, QBrush, QPainterPath,
    QTransform,
)
from montaris.core.selection import SelectionModel
from montaris.core.busy import busy_cursor, should_process_events
from montaris import theme as _theme


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
        self._clip_rect = None  # image bounds in scene coords for OOB clipping
        self.setAcceptedMouseButtons(Qt.NoButton)

    def setImage(self, image):
        old = self._rect
        self._rect = QRectF(0, 0, image.width(), image.height())
        if old != self._rect:
            self.prepareGeometryChange()
        self._image = image
        self.update()

    def setClipRect(self, scene_rect):
        """Set the image boundary rect (in scene coords) for clipping."""
        self._clip_rect = scene_rect

    def image(self):
        return self._image  # same object, no copy

    def boundingRect(self):
        return self._rect

    def paint(self, painter, option, widget=None):
        if self._image:
            if self._clip_rect is not None:
                # Convert scene clip rect to item-local coords
                local_clip = self.mapRectFromScene(self._clip_rect).toAlignedRect()
                painter.setClipRect(local_clip)
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
        self._polygon_close_marker = None
        self._brush_preview = None
        self._stamp_preview = None

        # Throttled partial refresh for large brush strokes
        self._pending_dirty = {}  # id(layer) -> (layer, (y1, y2, x1, x2))
        self._dirty_timer = QTimer(self)
        self._dirty_timer.setSingleShot(True)
        self._dirty_timer.setInterval(16)  # ~60fps max
        self._dirty_timer.timeout.connect(self._flush_dirty)

        self._tint_color = None
        self._adjustments = None  # ImageAdjustments applied at display time
        self._display_cache = None  # cached uint8 display copy for fast LUT
        self._tool = None
        self._active_layer = None
        self._is_panning = False
        self._last_pan_pos = None
        self._space_held = False
        self._last_scene_pos = None  # track for brush cursor refresh

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
        self.setBackgroundBrush(_theme.canvas_background())
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # HUD overlay (E.23)
        self._hud_label = QLabel(self)
        self._hud_label.setStyleSheet(_theme.hud_label_style())
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

        # Empty-state hint overlay
        self._empty_hint = QLabel(self)
        self._empty_hint.setText("Open an image to begin\nFile > Open Image(s)")
        self._empty_hint.setAlignment(Qt.AlignCenter)
        self._empty_hint.setStyleSheet(_theme.empty_state_style())
        self._empty_hint.show()

        # Floating control bars (one per corner)
        self._floating_btns = []  # all floating buttons for theme refresh
        self._editing_locked = False
        self._zoom_bar = self._build_zoom_bar()       # bottom-right
        self._view_bar = self._build_view_bar()        # top-left (below HUD)
        self._edit_bar = self._build_edit_bar()        # top-right
        self._roi_bar = self._build_roi_bar()          # bottom-left

    # ------------------------------------------------------------------
    # Tool / layer management
    # ------------------------------------------------------------------

    _PAINT_TOOLS = {'Brush', 'Eraser', 'Polygon', 'Bucket Fill',
                     'Rectangle', 'Circle', 'Stamp',
                     'Transform (selected)', 'Transform All'}

    def refresh_theme(self):
        """Update canvas visuals after a theme switch."""
        self.setBackgroundBrush(_theme.canvas_background())
        self._hud_label.setStyleSheet(_theme.hud_label_style())
        self._empty_hint.setStyleSheet(_theme.empty_state_style())
        # Floating bars
        bar_style = _theme.zoom_bar_style()
        btn_style = _theme.zoom_bar_button_style()
        for bar in (self._zoom_bar, self._view_bar, self._edit_bar, self._roi_bar):
            bar.setStyleSheet(bar_style)
        for b in self._floating_btns:
            b.setStyleSheet(btn_style)
            if hasattr(b, '_qta_name'):
                try:
                    import qtawesome as qta
                    color = '#ffffff' if _theme.is_dark() else '#111'
                    b.setIcon(qta.icon(b._qta_name, color=color))
                except ImportError:
                    pass
        self._zb_pct.setStyleSheet(_theme.zoom_bar_pct_style())

    def set_tool(self, tool):
        t0 = time.perf_counter()
        # Clean up old tool's scene items (e.g. transform handles)
        old = self._tool
        if old is not None and hasattr(old, '_clear_handles'):
            old._clear_handles(self)
        # Set new tool BEFORE flatten so events during slow ops use correct tool
        self._tool = tool
        self.hide_brush_preview()
        self._hide_stamp_preview()
        self._update_cursor()
        _t1 = time.perf_counter()
        # Flatten offsets when switching to paint/transform tools
        if tool is not None and getattr(tool, 'name', None) in self._PAINT_TOOLS:
            self._flatten_all_offsets()
        _t2 = time.perf_counter()
        if tool is not None and hasattr(tool, 'on_activate'):
            tool.on_activate(self._active_layer, self)
        _t3 = time.perf_counter()
        from montaris.core.event_logger import EventLogger
        _log = EventLogger.instance()
        _log.log("tool", "set_tool",
            duration_ms=(time.perf_counter() - t0) * 1000,
            tool=getattr(tool, 'name', 'None'),
            flatten_ms=(_t2 - _t1) * 1000,
            activate_ms=(_t3 - _t2) * 1000,
            setup_ms=(_t1 - t0) * 1000)
        _log.log_mem("set_tool.done", tool=getattr(tool, 'name', 'None'))

    def _flatten_all_offsets(self):
        """Bake any non-zero layer offsets into masks, preserving undo history."""
        # Fast path: skip busy_cursor (and its processEvents overhead)
        # when no ROIs have offsets — common case for fresh imports.
        if not any(r.offset_x != 0 or r.offset_y != 0
                   for r in self.layer_stack.roi_layers):
            return
        # Capture pre-flatten state for undo
        pre_flatten = []
        with busy_cursor("Flattening layer offsets...", self.window()):
            any_offset = False
            partially_clipped = []
            _last_pe = time.monotonic()
            for roi in self.layer_stack.roi_layers:
                if roi.offset_x != 0 or roi.offset_y != 0:
                    if roi.has_oob_content():
                        partially_clipped.append(roi.name)
                    # Snapshot before flatten
                    bbox = roi.get_bbox()
                    crop = (roi.mask[bbox[0]:bbox[1], bbox[2]:bbox[3]].copy()
                            if bbox is not None else None)
                    old_offset = (roi.offset_x, roi.offset_y)
                    result = roi.flatten_offset()
                    if result:
                        any_offset = True
                        pre_flatten.append((roi, crop, bbox, old_offset))
                    # result=False means fully OOB — offset preserved
                _last_pe = should_process_events(_last_pe)
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
        if any_offset and pre_flatten:
            from montaris.core.undo import FlattenUndoCommand, _cmd_byte_size
            from montaris.core.multi_undo import CompoundUndoCommand
            flatten_cmd = FlattenUndoCommand(pre_flatten)
            win = self.window()
            if hasattr(win, 'undo_stack'):
                stack = win.undo_stack
                # Merge with previous command so undo reverses both in one step
                if stack._index >= 0:
                    prev = stack._stack[stack._index]
                    stack._total_bytes -= _cmd_byte_size(prev)
                    merged = CompoundUndoCommand([prev, flatten_cmd])
                    stack._stack[stack._index] = merged
                    stack._total_bytes += _cmd_byte_size(merged)
                else:
                    stack.push(flatten_cmd)
            self.refresh_overlays()

    def set_active_layer(self, layer):
        t0 = time.perf_counter()
        if layer is not self._active_layer:
            old = self._active_layer
            # Clear transform/move handles when switching layers
            tool = self._tool
            if tool is not None and hasattr(tool, '_clear_handles'):
                tool._clear_handles(self)
            self._active_layer = layer
            # Compress the old layer now that it's inactive
            if old is not None and hasattr(old, 'compress'):
                old.compress()
            # Ensure newly active ROI is at LOD 0 so dirty compositing aligns
            self._ensure_active_lod0(layer)
            # Re-show bbox for new layer if tool supports it
            if tool is not None and hasattr(tool, 'on_activate'):
                tool.on_activate(layer, self)
        else:
            self._active_layer = layer
        # Always pulse for visual emphasis on selection
        self._pulse_selection()
        from montaris.core.event_logger import EventLogger
        EventLogger.instance().log("tool", "set_active_layer",
            duration_ms=(time.perf_counter() - t0) * 1000)

    def _ensure_active_lod0(self, layer):
        """Re-render a newly active ROI at LOD 0 if it's currently at a lower resolution.

        When zoomed out, inactive ROIs are rendered at reduced LOD (e.g. LOD 3 = scale 8x).
        Dirty-region compositing assumes item scale=1, so painting on a scaled item
        causes visual misalignment. This ensures the active layer is always at full res.
        """
        if layer is None or not getattr(layer, 'is_roi', False):
            return
        rid = id(layer)
        item = self._roi_items.get(rid)
        if item is None:
            return
        if item.scale() > 1.0:
            try:
                index = self.layer_stack.roi_layers.index(layer)
            except ValueError:
                return
            self._refresh_roi_item(layer, index, lod_level=0)
            self._roi_lod[rid] = 0

    def _on_selection_changed(self, layers):
        """Sync _active_layer to primary selection and update highlights."""
        _t0 = time.perf_counter()
        primary = self._selection.primary
        old = self._active_layer
        if primary is not None:
            self._active_layer = primary
            # Ensure newly active ROI is at LOD 0 so dirty compositing aligns
            if primary is not old:
                self._ensure_active_lod0(primary)
        self._update_selection_highlights()
        _t1 = time.perf_counter()
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
        _t2 = time.perf_counter()
        from montaris.core.event_logger import EventLogger
        EventLogger.instance().log("tool", "_on_selection_changed",
            duration_ms=(_t2 - _t0) * 1000,
            highlights_ms=(_t1 - _t0) * 1000,
            on_activate_ms=(_t2 - _t1) * 1000,
            n_selected=len(layers))

    def _clean_stale_selection(self):
        """Remove layers from selection that are no longer in the layer stack."""
        current_rois = set(id(r) for r in self.layer_stack.roi_layers)
        stale = [l for l in self._selection.layers if id(l) not in current_rois]
        if stale:
            for l in stale:
                self._selection.remove(l)

    def _update_selection_highlights(self):
        """Draw actual ROI boundary outline for selected layers.

        Skips per-ROI edge highlights when many ROIs are selected
        (e.g. Transform All) — the transform bbox outline is sufficient
        and avoids O(n) edge detection passes.
        """
        scene = self._scene
        for item in self._selection_highlight_items:
            scene.removeItem(item)
        self._selection_highlight_items.clear()

        selected = self._selection.layers
        # Skip expensive per-ROI edge highlights for bulk selections
        if len(selected) > 10:
            return

        for layer in selected:
            if not getattr(layer, 'is_roi', False):
                continue
            bbox = layer.get_bbox()
            if bbox is None:
                continue
            y1, y2, x1, x2 = bbox
            bh, bw = y2 - y1, x2 - x1
            mask_crop = layer.get_mask_crop((y1, y2, x1, x2))
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
        """Animated grow-then-shrink pulse on selection for visual emphasis."""
        # Stop any running pulse
        if self._pulse_timer is not None:
            self._pulse_timer.stop()
            self._pulse_timer = None

        layer = self._active_layer
        if layer is None:
            return
        rid = id(layer)
        pop_item = self._roi_items.get(rid)
        if pop_item is None:
            return
        rect = pop_item._rect
        iw, ih = rect.width(), rect.height()
        if iw <= 0 or ih <= 0:
            return

        # Compute max grow in item-local pixels (fixed 3 screen px per side)
        item_scale = pop_item.scale() or 1.0
        zoom = self.transform().m11()
        scene_per_px = 1.0 / (zoom * item_scale) if zoom * item_scale else 1.0
        max_grow = 3.0 * scene_per_px

        cx, cy = iw / 2.0, ih / 2.0

        # 200ms ease-out-in: ramp up first half, ramp down second half
        timeline = QTimeLine(200, self)
        timeline.setFrameRange(0, 100)

        def _on_frame(frame):
            # 0→50: grow, 50→100: shrink  (triangle envelope)
            t_norm = frame / 100.0
            frac = 1.0 - abs(2.0 * t_norm - 1.0)  # 0→1→0
            grow = max_grow * frac
            sx = (iw + 2 * grow) / iw
            sy = (ih + 2 * grow) / ih
            t = QTransform(sx, 0, 0, sy, cx * (1 - sx), cy * (1 - sy))
            pop_item.setTransform(t)

        def _on_finished():
            pop_item.setTransform(QTransform())

        timeline.frameChanged.connect(_on_frame)
        timeline.finished.connect(_on_finished)
        timeline.start()
        self._pulse_timer = timeline

    # ------------------------------------------------------------------
    # Image display — single QGraphicsPixmapItem (no tile pyramid)
    # ------------------------------------------------------------------

    def set_tint_color(self, tint_color):
        """Set the display tint for the background image (None for grayscale)."""
        self._tint_color = tint_color

    def refresh_image(self):
        t0 = time.perf_counter()
        if self._image_item:
            self._scene.removeItem(self._image_item)
            self._image_item = None
        self._display_cache = None  # invalidate cached uint8

        img_layer = self.layer_stack.image_layer
        if img_layer is None:
            self._empty_hint.setVisible(True)
            return

        self._empty_hint.setVisible(False)
        data = self._get_display_uint8()
        if data is None:
            return
        # Apply brightness/contrast/exposure/gamma adjustments (display-time)
        if self._adjustments is not None and not self._adjustments.is_identity():
            data = self._adjustments.apply(data)
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
        self._report_render((time.perf_counter() - t0) * 1000, "refresh_image")

    def _get_display_uint8(self):
        """Get a cached uint8 display copy of the image data.

        For uint8 images, returns the data directly (no copy).
        For uint16/float images, normalizes to uint8 once and caches.
        Cache is invalidated when refresh_image() is called (new image).
        """
        img_layer = self.layer_stack.image_layer
        if img_layer is None:
            return None
        data = img_layer.data
        if data.dtype == np.uint8:
            return data
        if self._display_cache is not None:
            return self._display_cache
        # Normalize to uint8 — expensive but done once per image load
        img = data.astype(np.float32)
        if data.dtype == np.uint16:
            mn, mx = float(data.min()), float(data.max())
            if mx > mn:
                img = (img - mn) * (255.0 / (mx - mn))
            else:
                img = np.zeros_like(data, dtype=np.float32)
        else:
            mn, mx = img.min(), img.max()
            if mx > mn:
                img = (img - mn) * (255.0 / (mx - mn))
        self._display_cache = np.clip(img, 0, 255).astype(np.uint8)
        return self._display_cache

    def refresh_adjustments(self):
        """Fast path: re-apply adjustments without rebuilding scene structure.

        For large images (>4MP), applies LUT only to the visible viewport
        and displays that as a viewport-sized pixmap. The full image is
        updated on a deferred timer so panning shows correct data.
        """
        if self._image_item is None:
            return
        data = self._get_display_uint8()
        if data is None:
            return

        adj = self._adjustments
        has_adj = adj is not None and not adj.is_identity()
        tint = getattr(self, '_tint_color', None)
        has_tint = tint is not None and data.ndim == 2

        if not has_adj and not has_tint:
            qimg = numpy_to_qimage(data)
            self._image_item.setPixmap(QPixmap.fromImage(qimg))
            return

        lut = adj._build_lut() if has_adj else None

        h, w = data.shape[:2]
        if h * w > 4_000_000:
            # Large image: process only visible viewport for responsive sliders
            vr = self.mapToScene(self.viewport().rect()).boundingRect()
            x0 = max(0, int(vr.x()))
            y0 = max(0, int(vr.y()))
            x1 = min(w, int(vr.x() + vr.width()) + 1)
            y1 = min(h, int(vr.y() + vr.height()) + 1)
            if x1 <= x0 or y1 <= y0:
                return

            region = data[y0:y1, x0:x1]
            if lut is not None:
                region = lut[region]
            if has_tint and region.ndim == 2:
                region = _apply_tint(region, tint)

            qimg = numpy_to_qimage(region)
            pixmap = QPixmap.fromImage(qimg)
            self._image_item.setPixmap(pixmap)
            self._image_item.setOffset(x0, y0)

            # Schedule deferred full-image update for when user stops adjusting
            if not hasattr(self, '_full_adj_timer'):
                self._full_adj_timer = QTimer(self)
                self._full_adj_timer.setSingleShot(True)
                self._full_adj_timer.setInterval(300)
                self._full_adj_timer.timeout.connect(self._deferred_full_adjustment)
            self._full_adj_timer.start()
        else:
            # Small image: process everything directly
            result = lut[data] if lut is not None else data
            if has_tint and result.ndim == 2:
                result = _apply_tint(result, tint)
            qimg = numpy_to_qimage(result)
            self._image_item.setPixmap(QPixmap.fromImage(qimg))
            self._image_item.setOffset(0, 0)

    def _deferred_full_adjustment(self):
        """Apply adjustments to the full image (called after slider stops)."""
        if self._image_item is None:
            return
        data = self._get_display_uint8()
        if data is None:
            return
        adj = self._adjustments
        if adj is not None and not adj.is_identity():
            lut = adj._build_lut()
            data = lut[data]
        tint = getattr(self, '_tint_color', None)
        if tint is not None and data.ndim == 2:
            data = _apply_tint(data, tint)
        qimg = numpy_to_qimage(data)
        self._image_item.setPixmap(QPixmap.fromImage(qimg))
        self._image_item.setOffset(0, 0)

    def refresh_image_from_array(self, data):
        """Display an arbitrary numpy array as the background image."""
        if self._image_item:
            self._scene.removeItem(self._image_item)
            self._image_item = None
        # Apply brightness/contrast/exposure/gamma adjustments (display-time)
        if self._adjustments is not None and not self._adjustments.is_identity():
            data = self._adjustments.apply(data)
        qimg = numpy_to_qimage(data)
        pixmap = QPixmap.fromImage(qimg)
        self._image_item = QGraphicsPixmapItem(pixmap)
        self._image_item.setZValue(0)
        self._scene.addItem(self._image_item)
        h, w = data.shape[:2]
        m = min(w, h) // 4
        self._scene.setSceneRect(QRectF(-m, -m, w + 2 * m, h + 2 * m))

    def _image_clip_rect(self):
        """Return QRectF of image bounds for clipping ROI overlays."""
        img = self.layer_stack.image_layer
        if img is not None:
            h, w = img.data.shape[:2]
            return QRectF(0, 0, w, h)
        return None

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

    def _report_render(self, ms, render_type="overlay"):
        """Report a frame to the perf monitor if available."""
        win = self.parent()
        if win and hasattr(win, 'perf_monitor'):
            win.perf_monitor.record_frame()
            win.perf_monitor.record_render_time(ms)
            # Update tile cache info
            img = self.layer_stack.image_layer
            if img and img._tile_pyramid:
                cache = img._tile_pyramid._cache
                win.perf_monitor.set_tile_cache_info(f"{cache.size}/{cache._max_size}")
            else:
                win.perf_monitor.set_tile_cache_info("0")
        from montaris.core.event_logger import EventLogger
        EventLogger.instance().log("render", render_type, duration_ms=ms)

    def _do_refresh_overlays(self):
        """Internal: actual rebuild of all ROI pixmap items."""
        t0 = time.perf_counter()
        # Cancel any pending partial dirty — full rebuild supersedes
        self._pending_dirty.clear()
        if self.layer_stack.image_layer is None:
            for item in self._roi_items.values():
                item.setVisible(False)
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

        # Collect visible ROIs and cull off-screen ones
        # Note: bbox is invalidated by mask setter / mark_dirty, not here.
        # Keeping cached bboxes avoids decompressing compressed ROI masks.
        visible_jobs = []  # (index, roi, target_lod)
        for i, roi in enumerate(rois):
            rid = id(roi)
            if use_culling and roi != self._active_layer:
                dbbox = roi.get_display_bbox()
                if dbbox is not None:
                    y1, y2, x1, x2 = dbbox
                    if not vp_rect.intersects(QRectF(x1, y1, x2 - x1, y2 - y1)):
                        self._roi_stale.add(rid)
                        continue
            self._roi_stale.discard(rid)
            target_lod = 0 if roi == self._active_layer else lod
            visible_jobs.append((i, roi, target_lod))

        nv = len(visible_jobs)
        if show_progress and nv > 0:
            self._progress_bar.setRange(0, nv)
            self._progress_bar.setValue(0)
            self._progress_bar.show()

        # Parallel path: compute RGBA arrays in threads, apply on main thread.
        # Process in batches of pool_size to cap peak memory — each batch's
        # crops/RGBAs are freed before the next batch is submitted.
        if nv > 3:
            from montaris.core.workers import get_pool, worker_count
            from montaris.core.busy import should_process_events
            pool = get_pool()
            batch_size = worker_count() * 2
            _last_pe = time.monotonic()
            job_i = 0

            # Get perf monitor for memory sampling during render
            _perf = getattr(self.parent(), 'perf_monitor', None) if self.parent() else None

            for batch_start in range(0, nv, batch_size):
                batch = visible_jobs[batch_start:batch_start + batch_size]
                futures = []
                for idx, roi, target_lod in batch:
                    if not roi.visible:
                        futures.append((idx, roi, target_lod, None))
                        continue
                    bbox = roi.get_bbox()
                    if bbox is None:
                        futures.append((idx, roi, target_lod, None))
                        continue
                    fill_mode = getattr(roi, 'fill_mode', 'solid')
                    eff_opacity = roi.opacity * gof
                    # Snapshot mask crop on main thread — workers must not
                    # read the shared roi.mask while the main thread may
                    # modify it (e.g. during processEvents).
                    y1, y2, x1, x2 = bbox
                    mask_crop = roi.get_mask_crop((y1, y2, x1, x2)).copy()
                    fut = pool.submit(
                        _compute_roi_rgba_from_crop, mask_crop, roi.color,
                        eff_opacity, fill_mode, target_lod,
                        x1 + roi.offset_x, y1 + roi.offset_y,
                    )
                    futures.append((idx, roi, target_lod, fut))

                # Sample memory after submitting batch (catches peak allocation)
                if _perf is not None:
                    _perf._sample_mem()

                # Collect batch results before submitting next batch
                for idx, roi, target_lod, fut in futures:
                    rid = id(roi)
                    if fut is None:
                        existing = self._roi_items.get(rid)
                        if existing is not None:
                            existing.setVisible(False)
                    else:
                        result = fut.result()
                        self._apply_roi_rgba_result(roi, idx, result)
                    self._roi_lod[rid] = target_lod
                    job_i += 1
                    if show_progress:
                        self._progress_bar.setValue(job_i)
                        _last_pe = should_process_events(_last_pe)
        else:
            # Sequential path for small counts
            for job_i, (idx, roi, target_lod) in enumerate(visible_jobs):
                self._refresh_roi_item(roi, idx, gof, lod_level=target_lod)
                self._roi_lod[id(roi)] = target_lod
                if show_progress:
                    self._progress_bar.setValue(job_i + 1)

        if show_progress:
            self._progress_bar.hide()

        self._report_render((time.perf_counter() - t0) * 1000, "refresh_overlays")
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
        if layer is None or not getattr(layer, 'is_roi', False):
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
        self.flash_progress()
        if layer in self._selection.layers:
            self._update_selection_highlights()

    def refresh_active_overlay_partial(self, layer, dirty_bbox):
        """Throttled partial refresh: accumulate dirty region, render on timer."""
        if layer is None or not getattr(layer, 'is_roi', False):
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
        if not pending:
            return
        lod = self._current_lod_level()
        t0 = time.perf_counter()
        for lid, (layer, bbox) in pending.items():
            self._render_dirty_region(layer, bbox, lod_level=lod)
        self._report_render((time.perf_counter() - t0) * 1000, "flush_dirty")

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
            item.setClipRect(self._image_clip_rect())
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
                existing.setVisible(False)
            return

        bbox = roi.get_bbox()  # cached
        if bbox is None:
            if existing is not None:
                existing.setVisible(False)
            return

        y1, y2, x1, x2 = bbox
        bh, bw = y2 - y1, x2 - x1
        r, g, b = roi.color
        effective_opacity = int(roi.opacity * gof)
        fill_mode = getattr(roi, 'fill_mode', 'solid')

        mask_crop = roi.get_mask_crop((y1, y2, x1, x2))

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
        clip = self._image_clip_rect()
        if existing is not None:
            existing.setImage(qimg.copy())
            existing.setPos(disp_x, disp_y)
            existing.setZValue(1 + index * 0.001)
            existing.setClipRect(clip)
            existing.setVisible(True)
        else:
            item = _ROIOverlayItem()
            item.setImage(qimg.copy())
            item.setPos(disp_x, disp_y)
            item.setZValue(1 + index * 0.001)
            item.setClipRect(clip)
            self._scene.addItem(item)
            self._roi_items[rid] = item
            existing = item

        if scale_factor > 1:
            existing.setScale(scale_factor)
        else:
            existing.setScale(1)

    def _apply_roi_rgba_result(self, roi, index, result):
        """Apply precomputed RGBA result to the scene (main thread only)."""
        rid = id(roi)
        existing = self._roi_items.get(rid)
        rgba, bw, bh, disp_x, disp_y, scale_factor = result

        qimg = QImage(rgba.data, bw, bh, bw * 4, QImage.Format_RGBA8888)

        clip = self._image_clip_rect()
        if existing is not None:
            existing.setImage(qimg.copy())
            existing.setPos(disp_x, disp_y)
            existing.setZValue(1 + index * 0.001)
            existing.setClipRect(clip)
            existing.setVisible(True)
        else:
            item = _ROIOverlayItem()
            item.setImage(qimg.copy())
            item.setPos(disp_x, disp_y)
            item.setZValue(1 + index * 0.001)
            item.setClipRect(clip)
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
            pen = QPen(QColor(255, 255, 255, 180), 5)
            pen.setCosmetic(True)
            self._brush_preview.setPen(pen)
            self._brush_preview.setBrush(QBrush(Qt.NoBrush))
            self._brush_preview.setZValue(2000)
            self._scene.addItem(self._brush_preview)
        # Match brush preview to active ROI color (C.6)
        if self._active_layer and hasattr(self._active_layer, 'color'):
            r, g, b = self._active_layer.color
            pen = QPen(QColor(r, g, b, 200), 5)
        else:
            pen = QPen(QColor(255, 255, 255, 180), 5)
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
        # Large close-marker at first vertex (appears after 2+ vertices)
        if len(vertices) >= 2:
            sx, sy = vertices[0]
            scale = self.transform().m11() or 1.0
            r = max(6, 12 / max(scale, 0.01))
            marker_pen = QPen(QColor(255, 255, 0), 1.5)
            marker_pen.setCosmetic(True)
            self._polygon_close_marker = self._scene.addEllipse(
                sx - r, sy - r, r * 2, r * 2,
                marker_pen, QBrush(QColor(255, 255, 0, 100)),
            )
            self._polygon_close_marker.setZValue(1001)

    def clear_polygon_preview(self):
        if self._polygon_item:
            self._scene.removeItem(self._polygon_item)
            self._polygon_item = None
        if self._polygon_close_marker:
            self._scene.removeItem(self._polygon_close_marker)
            self._polygon_close_marker = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep progress bar at bottom
        w = self.viewport().width()
        h = self.viewport().height()
        self._progress_bar.setGeometry(0, h - 4, w, 4)
        # Center empty-state hint
        hint_w, hint_h = 380, 140
        self._empty_hint.setGeometry(
            (w - hint_w) // 2, (h - hint_h) // 2, hint_w, hint_h
        )
        # Position all floating bars
        self._position_floating_bars()

    # ------------------------------------------------------------------
    # Floating control bars
    # ------------------------------------------------------------------

    def _make_float_btn(self, lay, icon_name, fallback, tooltip, callback,
                        checkable=False):
        """Create a themed icon button and add it to *lay*."""
        b = QPushButton()
        try:
            import qtawesome as qta
            color = '#ffffff' if _theme.is_dark() else '#111'
            b.setIcon(qta.icon(icon_name, color=color))
        except ImportError:
            b.setText(fallback)
        b.setFixedSize(26, 26)
        b.setToolTip(tooltip)
        b.setStyleSheet(_theme.zoom_bar_button_style())
        b.clicked.connect(callback)
        b._qta_name = icon_name
        if checkable:
            b.setCheckable(True)
        lay.addWidget(b)
        self._floating_btns.append(b)
        return b

    def _make_float_bar(self):
        """Return a new transparent floating bar widget."""
        bar = QWidget(self)
        bar.setObjectName("FloatingBar")
        bar.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        bar.setStyleSheet(_theme.zoom_bar_style())
        return bar

    # -- Bottom-right: Zoom ------------------------------------------------

    def _build_zoom_bar(self):
        bar = self._make_float_bar()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(2)

        self._zb_out = self._make_float_btn(lay, 'fa6s.minus', '-',
                                             'Zoom Out (Ctrl+-)', self.zoom_out)
        self._zb_pct = QPushButton('100%')
        self._zb_pct.setFixedHeight(26)
        self._zb_pct.setMinimumWidth(48)
        self._zb_pct.setToolTip('Reset Zoom 1:1 (Ctrl+1)')
        self._zb_pct.setStyleSheet(_theme.zoom_bar_pct_style())
        self._zb_pct.clicked.connect(self.reset_zoom)
        lay.addWidget(self._zb_pct)
        self._zb_in = self._make_float_btn(lay, 'fa6s.plus', '+',
                                            'Zoom In (Ctrl+=)', self.zoom_in)
        bar.adjustSize()
        bar.show()
        return bar

    # -- Top-left: View (below HUD) ----------------------------------------

    def _build_view_bar(self):
        bar = self._make_float_bar()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(2)

        self._make_float_btn(lay, 'fa6s.left-right', '\u2194',
                             'Flip Horizontal (Ctrl+H)', self._cb_flip_h)
        self._make_float_btn(lay, 'fa6s.rotate', '\u21BB',
                             'Rotate 90\u00b0 CW (Ctrl+R)', self._cb_rotate_90)
        bar.adjustSize()
        bar.show()
        return bar

    # -- Top-right: Edit ----------------------------------------------------

    def _build_edit_bar(self):
        bar = self._make_float_bar()
        bar.hide()
        return bar

    # -- Bottom-left: Screenshot / Lock / Fullscreen ------------------------

    def _build_roi_bar(self):
        bar = self._make_float_bar()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(2)

        self._make_float_btn(lay, 'fa6s.camera', '\U0001F4F7',
                             'Screenshot', self._cb_screenshot)
        self._lock_btn = self._make_float_btn(
            lay, 'fa6s.lock-open', '\U0001F513',
            'Lock / Unlock Editing', self._cb_toggle_lock, checkable=True)
        self._make_float_btn(lay, 'fa6s.maximize', '\u26F6',
                             'Fullscreen (F11)', self._cb_fullscreen)
        bar.adjustSize()
        bar.show()
        return bar

    # -- Positioning --------------------------------------------------------

    def _position_floating_bars(self):
        """Pin each bar to its corner."""
        vw = self.viewport().width()
        vh = self.viewport().height()
        margin = 10

        # Bottom-right: zoom
        bw, bh = self._zoom_bar.sizeHint().width(), self._zoom_bar.sizeHint().height()
        self._zoom_bar.move(vw - bw - margin, vh - bh - 14)

        # Top-left: view (below HUD)
        hud_bottom = self._hud_label.y() + self._hud_label.sizeHint().height() + 4
        self._view_bar.move(margin - 4, hud_bottom)

        # Top-right: edit
        ew = self._edit_bar.sizeHint().width()
        self._edit_bar.move(vw - ew - margin, margin)

        # Bottom-left: roi nav
        rh = self._roi_bar.sizeHint().height()
        self._roi_bar.move(margin - 4, vh - rh - 14)

    # -- Callbacks (delegate to parent MontarisApp) -------------------------

    def _app(self):
        """Return parent MontarisApp instance."""
        return self.parent()

    def _cb_flip_h(self):
        app = self._app()
        if app and hasattr(app, 'flip_horizontal'):
            app.flip_horizontal()

    def _cb_rotate_90(self):
        app = self._app()
        if app and hasattr(app, 'rotate_90'):
            app.rotate_90()

    def _cb_toggle_lock(self):
        self._editing_locked = self._lock_btn.isChecked()
        try:
            import qtawesome as qta
            color = '#dcdcdc' if _theme.is_dark() else '#333'
            icon_name = 'fa6s.lock' if self._editing_locked else 'fa6s.lock-open'
            self._lock_btn.setIcon(qta.icon(icon_name, color=color))
            self._lock_btn._qta_name = icon_name
        except ImportError:
            self._lock_btn.setText('\U0001F512' if self._editing_locked else '\U0001F513')

    def _cb_screenshot(self):
        """Save a screenshot of the current canvas view."""
        app = self._app()
        if not app:
            return
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            app, "Save Screenshot", "",
            "PNG (*.png);;JPEG (*.jpg);;All Files (*)",
        )
        if path:
            pixmap = self.grab()
            pixmap.save(path)
            if hasattr(app, 'toast'):
                app.toast.show(f"Screenshot saved", "success")

    def _cb_fullscreen(self):
        app = self._app()
        if not app:
            return
        if app.isFullScreen():
            app.showNormal()
        else:
            app.showFullScreen()

    # -- Zoom helpers -------------------------------------------------------

    def _update_zoom_pct(self):
        """Sync zoom percentage label to current view transform."""
        zoom = self.transform().m11()
        self._zb_pct.setText(f'{zoom:.0%}')

    def fit_to_window(self):
        if self._image_item:
            self.fitInView(self._image_item, Qt.KeepAspectRatio)
        self._update_zoom_pct()

    def reset_zoom(self):
        self.resetTransform()
        self._update_zoom_pct()

    def zoom_in(self):
        self.scale(1.25, 1.25)
        self.viewport_changed.emit()
        self._update_zoom_pct()

    def zoom_out(self):
        self.scale(1 / 1.25, 1 / 1.25)
        self.viewport_changed.emit()
        self._update_zoom_pct()

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
        # 250ms debounce — avoids re-rasterization during continuous zoom/scroll
        self._lod_timer.start(250)

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

        # Show progress flash during LOD re-render
        self.flash_progress()

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
        self._update_zoom_pct()

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
            # Block drawing tools when editing is locked
            if (self._editing_locked
                    and getattr(self._tool, 'name', None) in self._PAINT_TOOLS):
                return
            # Block paint tools when no ROI is selected
            if (self._active_layer is None
                    and getattr(self._tool, 'name', None) in self._PAINT_TOOLS
                    and self.layer_stack.image_layer is not None):
                win = self.window()
                if hasattr(win, 'toast'):
                    win.toast.show(
                        "No ROI selected — create or select an ROI in Layers & Properties",
                        "warning",
                    )
                return
            scene_pos = self.mapToScene(event.position().toPoint())
            self._tool.on_press(scene_pos, self._active_layer, self)
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        self._last_scene_pos = scene_pos
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
            zoom = self.transform().m11() or 1.0
            radius = self._tool.size / max(zoom, 0.01) / 2
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

    def refresh_brush_cursor(self):
        """Redraw the brush cursor at the last known mouse position."""
        if self._last_scene_pos is not None:
            self._update_brush_cursor(self._last_scene_pos)


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

        if c == 1:
            arr = np.ascontiguousarray(arr[:, :, 0])
            img = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
            return img.copy()

        arr = np.ascontiguousarray(arr)
        if c == 3:
            img = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
            return img.copy()
        elif c == 4:
            img = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888)
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
    try:
        from montaris.core.accel import compute_edge
        return compute_edge(mask)
    except ImportError:
        pass
    from scipy.ndimage import binary_erosion
    filled = mask > 0
    return filled ^ binary_erosion(filled)


def _compute_roi_rgba_from_crop(mask_crop, color, opacity, fill_mode, lod_level, disp_x, disp_y):
    """Pure numpy/scipy computation of ROI RGBA from a pre-cropped mask.

    Thread-safe: operates only on the owned mask_crop copy.
    Returns (rgba_array, width, height, disp_x, disp_y, scale_factor).
    Dispatches to Numba JIT kernels when acceleration is enabled.
    """
    try:
        from montaris.core.accel import compute_roi_rgba
        return compute_roi_rgba(mask_crop, color, opacity, fill_mode,
                                lod_level, disp_x, disp_y)
    except ImportError:
        pass

    # Numpy fallback
    bh, bw = mask_crop.shape
    r, g, b = color
    effective_opacity = int(opacity)

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
    return (rgba, bw, bh, disp_x, disp_y, scale_factor)


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
