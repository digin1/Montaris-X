"""Generate an animated GIF showcasing Montaris-X with a natural user workflow."""
import sys
import os
import glob as globmod
import time
import math
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "windows"

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, Qt, QPointF, QRectF
from PySide6.QtGui import QColor
from PIL import Image

app = QApplication(sys.argv)
from montaris.app import MontarisApp, apply_light_theme, apply_dark_theme

apply_dark_theme(app)
win = MontarisApp()
win._auto_detect_roi_zip = lambda f: None
win._auto_detect_session = lambda f: None
win._auto_load_instructions = lambda f: None
win.showMaximized()
app.processEvents()
time.sleep(0.8)
app.processEvents()

frames = []
durations = []

FRAME_DIR = os.path.join("docs", "frames")
os.makedirs(FRAME_DIR, exist_ok=True)

# Clean old frame files before generating new ones
for _old in globmod.glob(os.path.join(FRAME_DIR, "*.png")):
    os.remove(_old)


def grab(duration_ms=1200, label=None):
    """Capture current window as a PNG frame file."""
    app.processEvents()
    time.sleep(0.15)
    app.processEvents()
    from PySide6.QtGui import QImage
    qpx = win.grab()
    buf = qpx.toImage().convertToFormat(QImage.Format.Format_ARGB32)
    w, h = buf.width(), buf.height()
    ptr = buf.bits()
    if hasattr(ptr, 'setsize'):
        ptr.setsize(w * h * 4)
        raw = bytes(ptr)
    else:
        raw = bytes(ptr)
    img = Image.frombytes("RGBA", (w, h), raw, "raw", "BGRA")
    img = img.convert("RGB")
    frames.append(img)
    durations.append(duration_ms)
    idx = len(frames)
    safe_label = (label or "frame").replace(" ", "_").replace("/", "-")
    fname = f"{idx:02d}_{safe_label}.png"
    fpath = os.path.join(FRAME_DIR, fname)
    img.save(fpath)
    tag = f" [{label}]" if label else ""
    print(f"  Frame {idx:>2}{tag} -> {fname}")


def wait(ms=300):
    app.processEvents()
    time.sleep(ms / 1000)
    app.processEvents()


def select_tool(name):
    win.tool_panel._select_tool(name)
    wait(150)


def select_roi(idx):
    layers = win.layer_stack.roi_layers
    if idx >= len(layers):
        return
    roi = layers[idx]
    win._on_layer_selected(roi)
    row = idx + (1 if win.layer_stack.image_layer else 0)
    win.layer_panel.list_widget.setCurrentRow(row)
    wait(150)


def zoom_to_roi(idx, padding=300):
    layers = win.layer_stack.roi_layers
    if idx >= len(layers):
        return
    roi = layers[idx]
    bbox = roi.get_bbox()
    if bbox:
        y1, y2, x1, x2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half_w = (x2 - x1) // 2 + padding
        half_h = (y2 - y1) // 2 + padding
        zoom_to_rect(cx, cy, half_w, half_h)


def get_roi_center(idx):
    """Return (cx, cy) center of a ROI's bounding box."""
    layers = win.layer_stack.roi_layers
    if idx >= len(layers):
        return None
    roi = layers[idx]
    bbox = roi.get_bbox()
    if not bbox:
        return None
    y1, y2, x1, x2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def get_roi_size(idx):
    """Return (width, height) of a ROI's bounding box."""
    layers = win.layer_stack.roi_layers
    if idx >= len(layers):
        return None
    roi = layers[idx]
    bbox = roi.get_bbox()
    if not bbox:
        return None
    y1, y2, x1, x2 = bbox
    return (x2 - x1, y2 - y1)


def zoom_to_rect(cx, cy, half_w, half_h):
    """Zoom canvas to a rectangle, clamped within the image bounds."""
    img = win.layer_stack.image_layer
    if img is None:
        return
    ih, iw = img.shape[:2]
    x1 = max(0, cx - half_w)
    y1 = max(0, cy - half_h)
    x2 = min(iw, cx + half_w)
    y2 = min(ih, cy + half_h)
    rect = QRectF(x1, y1, x2 - x1, y2 - y1)
    win.canvas.fitInView(rect, Qt.KeepAspectRatio)
    wait(200)


def sim_stroke(points, tool_name=None):
    """Simulate a mouse stroke: press at first point, move through rest, release."""
    if tool_name:
        select_tool(tool_name)
    canvas = win.canvas
    layer = canvas._active_layer
    tool = canvas._tool
    if not tool or not layer:
        return
    pts = [QPointF(x, y) for x, y in points]
    tool.on_press(pts[0], layer, canvas)
    for p in pts[1:]:
        tool.on_move(p, layer, canvas)
        app.processEvents()
    tool.on_release(pts[-1], layer, canvas)
    canvas.viewport().update()
    wait(100)


def do_undo():
    """Trigger undo."""
    win.undo()
    wait(200)


def do_redo():
    """Trigger redo."""
    win.redo()
    wait(200)


# ── Pick a target ROI to work with ──────────────────────────
# ROI 62 (PO): center ~(6065,2907), size 839x811 — mid-brain, good visibility
TARGET_ROI = 62


def run():
    print("=== Generating Montaris-X Demo GIF ===\n")

    # ── 1. Fresh app ─────────────────────────────────────────
    print("1. Empty state")
    grab(1500, "Empty state")

    # ── 2. User opens an image ───────────────────────────────
    print("2. Load image")
    win.open_image(["test.tif"])
    wait(600)
    win.canvas.fit_to_window()
    wait(300)
    grab(1500, "Image loaded")

    # ── 3. User imports ROIs ─────────────────────────────────
    print("3. Import ROIs")
    win.import_roi_zip("test.zip")
    wait(1000)
    win.canvas.fit_to_window()
    wait(400)
    grab(2000, "ROIs imported")

    # ── 4. User browses ROIs — clicks through a few ─────────
    print("4. Browsing ROIs")
    for i in [5, 20, 45, 70]:
        select_roi(i)
        wait(80)
    grab(1200, "Browsing ROIs")

    # ── 5. User selects a ROI and zooms in to inspect ───────
    print("5. Select and zoom in")
    select_roi(TARGET_ROI)
    wait(200)
    grab(1000, "ROI selected")

    zoom_to_roi(TARGET_ROI, padding=200)
    grab(1500, "Zoomed into ROI")

    # ── 6. Zoom in closer ───────────────────────────────────
    print("6. Zoom closer")
    zoom_to_roi(TARGET_ROI, padding=80)
    grab(1200, "Close inspection")

    # ── 7. User zooms back out ──────────────────────────────
    print("7. Zoom back out")
    win.canvas.fit_to_window()
    wait(300)
    grab(1000, "Fit to window")

    # ── 8. User decides to delete this ROI ──────────────────
    print("8. Delete ROI")
    select_roi(TARGET_ROI)
    zoom_to_roi(TARGET_ROI, padding=200)
    wait(200)
    grab(1200, "About to delete")

    # Remember where it was before deleting
    roi_center = get_roi_center(TARGET_ROI)
    roi_size = get_roi_size(TARGET_ROI)

    win.clear_active_roi()
    wait(400)
    grab(1500, "ROI deleted")

    # ── 9. User adds a new ROI to draw in that area ─────────
    print("9. Add new ROI")
    win._on_roi_added()
    wait(200)
    new_idx = len(win.layer_stack.roi_layers) - 1
    select_roi(new_idx)
    wait(100)

    # Zoom to the area where the deleted ROI was
    # Match the zoom rectangle to the viewport aspect ratio so the image
    # always fills the view and we never show canvas outside the montage.
    cx, cy = roi_center
    rw, rh = roi_size
    vp = win.canvas.viewport()
    vp_aspect = vp.width() / max(vp.height(), 1)
    base_half = max(rw, rh) * 0.8
    draw_half_w = base_half * max(vp_aspect, 1.0)
    draw_half_h = base_half / min(vp_aspect, 1.0) if vp_aspect < 1.0 else base_half
    zoom_to_rect(cx, cy, draw_half_w, draw_half_h)
    wait(100)
    grab(1200, "New ROI ready")

    # Use a drawing radius that looks good at this zoom (~15% of ROI size)
    dr = min(rw, rh) * 0.15

    # Helper to restore zoom to the drawing area after tool operations
    def rezoom():
        zoom_to_rect(cx, cy, draw_half_w, draw_half_h)

    # ── 10. User draws a circle freehand with Brush ─────────
    print("10. Brush — freehand circle")
    bsz = max(5, int(dr / 8))
    win.tool_panel.size_slider.setValue(bsz)
    circle_pts = []
    for i in range(40):
        angle = (i / 40.0) * 2 * math.pi
        x = cx + dr * math.cos(angle)
        y = cy + dr * math.sin(angle)
        circle_pts.append((x, y))
    circle_pts.append(circle_pts[0])
    sim_stroke(circle_pts, 'Brush')
    rezoom()
    grab(1500, "Brush circle")

    # ── 11. User uses bucket fill to fill inside the circle ─
    print("11. Bucket fill inside circle")
    select_tool('Bucket Fill')
    canvas = win.canvas
    layer = canvas._active_layer
    tool = canvas._tool
    if tool and layer:
        tool.on_press(QPointF(cx, cy), layer, canvas)
        canvas.viewport().update()
        wait(200)
    rezoom()
    grab(1500, "Bucket filled")

    # ── 12. Undo the fill ───────────────────────────────────
    print("12. Undo bucket fill")
    do_undo()
    rezoom()
    grab(1200, "Undo fill")

    # ── 13. Undo the circle too ─────────────────────────────
    print("13. Undo circle")
    do_undo()
    rezoom()
    grab(1000, "Undo circle")

    # ── 14. Redo — bring circle back ────────────────────────
    print("14. Redo circle")
    do_redo()
    rezoom()
    grab(1200, "Redo circle")

    # ── 15. Draw a rectangle shape tool ─────────────────────
    print("15. Rectangle tool")
    sim_stroke([
        (cx - dr * 1.2, cy + dr * 0.5),
        (cx + dr * 1.2, cy + dr * 1.5),
    ], 'Rectangle')
    rezoom()
    grab(1200, "Rectangle drawn")

    # ── 16. Draw freehand brush stroke across the area ──────
    print("16. Brush — freehand stroke")
    win.tool_panel.size_slider.setValue(bsz)
    stroke_pts = []
    for i in range(25):
        t = i / 24.0
        x = cx - dr * 1.5 + t * dr * 3.0
        y = cy - dr * 0.8 + dr * 0.4 * math.sin(t * math.pi * 2)
        stroke_pts.append((x, y))
    sim_stroke(stroke_pts, 'Brush')
    rezoom()
    grab(1200, "Brush stroke")

    # ── 17. Eraser to clean up part of the stroke ───────────
    print("17. Eraser cleanup")
    win.tool_panel.size_slider.setValue(max(4, bsz + 2))
    erase_pts = []
    for i in range(12):
        t = i / 11.0
        x = cx + dr * 0.3 + t * dr * 0.8
        y = cy - dr * 0.5
        erase_pts.append((x, y))
    sim_stroke(erase_pts, 'Eraser')
    rezoom()
    grab(1200, "Eraser cleanup")

    # ── 18. Polygon tool — draw a polygon ───────────────────
    print("18. Polygon tool")
    select_tool('Polygon')
    canvas = win.canvas
    layer = canvas._active_layer
    tool = canvas._tool
    if tool and layer:
        poly_pts = [
            (cx - dr * 1.3, cy - dr * 0.3),
            (cx - dr * 0.8, cy - dr * 1.2),
            (cx - dr * 0.2, cy - dr * 0.8),
            (cx - dr * 0.3, cy - dr * 0.1),
        ]
        for x, y in poly_pts:
            tool.on_press(QPointF(x, y), layer, canvas)
            app.processEvents()
        if hasattr(tool, 'finish'):
            tool.finish()
        canvas.viewport().update()
        wait(200)
    rezoom()
    grab(1500, "Polygon drawn")

    # ── 19. Stamp tool ──────────────────────────────────────
    print("19. Stamp tool")
    sr = max(5, int(dr * 0.4))
    win.tool_panel.stamp_w_spin.setValue(sr)
    win.tool_panel.stamp_h_spin.setValue(sr)
    sim_stroke([
        (cx + dr * 0.8, cy - dr * 0.6),
        (cx + dr * 0.8, cy - dr * 0.6),
    ], 'Stamp')
    rezoom()
    grab(1200, "Stamp placed")

    # ── 20. Zoom out — overview ─────────────────────────────
    print("20. Zoom out — overview")
    select_tool('Hand')
    win.canvas.fit_to_window()
    wait(300)
    grab(1500, "Full overview")

    # ── 21. Transform tool — visible scaling ─────────────────
    print("21. Transform tool")
    TRANSFORM_ROI = 50
    select_roi(TRANSFORM_ROI)
    zoom_to_roi(TRANSFORM_ROI, padding=200)

    roi_t = win.layer_stack.roi_layers[TRANSFORM_ROI]
    t_bbox = roi_t.get_bbox()
    if t_bbox:
        ty1, ty2, tx1, tx2 = t_bbox
        t_cx = (tx1 + tx2) // 2
        t_cy = (ty1 + ty2) // 2

        select_tool('Transform (selected)')
        zoom_to_roi(TRANSFORM_ROI, padding=200)

        # First click on ROI center to show handles
        canvas = win.canvas
        layer = canvas._active_layer
        tool = canvas._tool
        if tool and layer:
            tool.on_press(QPointF(t_cx, t_cy), layer, canvas)
            canvas.viewport().update()
            wait(400)

        zoom_to_roi(TRANSFORM_ROI, padding=200)
        grab(1500, "Transform handles")

        # Drag bottom-right handle outward to scale up ~25%
        if tool and layer and tool._bbox:
            br = QPointF(tx2, ty2)
            scale_pct = 0.25
            target = QPointF(
                tx2 + (tx2 - tx1) * scale_pct,
                ty2 + (ty2 - ty1) * scale_pct,
            )
            tool.on_press(br, layer, canvas)
            # Animate the drag
            for s in range(1, 11):
                frac = s / 10.0
                pt = QPointF(
                    tx2 + (target.x() - tx2) * frac,
                    ty2 + (target.y() - ty2) * frac,
                )
                tool.on_move(pt, layer, canvas)
                app.processEvents()
            tool.on_release(target, layer, canvas)
            canvas.viewport().update()
            wait(400)

        zoom_to_roi(TRANSFORM_ROI, padding=300)
        grab(1500, "Transform scaled")

        # Undo to restore original
        do_undo()

    # ── 22. Move tool — visible movement ─────────────────────
    print("22. Move tool")
    select_roi(TRANSFORM_ROI)
    zoom_to_roi(TRANSFORM_ROI, padding=250)

    roi_m = win.layer_stack.roi_layers[TRANSFORM_ROI]
    m_bbox = roi_m.get_bbox()
    if m_bbox:
        my1, my2, mx1, mx2 = m_bbox
        m_cx = (mx1 + mx2) // 2
        m_cy = (my1 + my2) // 2
        move_dx = int((mx2 - mx1) * 0.4)  # move by 40% of width
        move_dy = int((my2 - my1) * 0.3)   # move by 30% of height

        select_tool('Move (selected)')
        zoom_to_roi(TRANSFORM_ROI, padding=250)

        canvas = win.canvas
        layer = canvas._active_layer
        tool = canvas._tool
        if tool and layer:
            start = QPointF(m_cx, m_cy)
            end = QPointF(m_cx + move_dx, m_cy + move_dy)
            tool.on_press(start, layer, canvas)
            # Animate the move
            for s in range(1, 11):
                frac = s / 10.0
                pt = QPointF(
                    m_cx + move_dx * frac,
                    m_cy + move_dy * frac,
                )
                tool.on_move(pt, layer, canvas)
                app.processEvents()
            tool.on_release(end, layer, canvas)
            canvas.viewport().update()
            wait(400)

        zoom_to_roi(TRANSFORM_ROI, padding=300)
        grab(1500, "ROI moved")

        # Undo to restore original position
        do_undo()

    # ── 23. Hide ROIs for image adjustment demo ────────────
    print("23. Hide ROIs for adjustments")
    select_tool('Hand')
    win.canvas.fit_to_window()
    wait(300)
    win.canvas._overlay_btn.setChecked(True)
    win.canvas._cb_toggle_overlay()
    wait(300)

    # ── 24. Image Adjustments — Smart Auto ──────────────────
    print("24. Smart Auto adjustment")
    win.adjustments_panel._on_auto()
    wait(500)
    grab(1500, "Smart Auto")

    # ── 25. Image Adjustments — Quick Boost ─────────────────
    print("25. Quick Boost")
    win.adjustments_panel._on_boost()
    wait(500)
    grab(1500, "Quick Boost")

    # ── 26. Reset adjustments and show ROIs ─────────────────
    print("26. Reset adjustments")
    win.adjustments_panel._on_reset()
    wait(300)
    win.canvas._overlay_btn.setChecked(False)
    win.canvas._cb_toggle_overlay()
    wait(300)
    grab(1000, "Adjustments reset")

    # ── 27. Global opacity — lower to 50% ──────────────────
    print("27. Global opacity 50%")
    win._global_opacity_slider.setValue(50)
    wait(500)
    grab(1500, "Global opacity 50pct")

    # Restore opacity
    win._global_opacity_slider.setValue(100)
    wait(300)

    # ── 27. Toggle ROI overlay on/off ───────────────────────
    print("27. Toggle overlay off")
    win.canvas._overlay_btn.setChecked(True)
    win.canvas._cb_toggle_overlay()
    wait(400)
    grab(1500, "Overlay hidden")

    print("    Toggle overlay on")
    win.canvas._overlay_btn.setChecked(False)
    win.canvas._cb_toggle_overlay()
    wait(400)
    grab(1200, "Overlay restored")

    # ── 28. Collapse both sidebars — minimal view ────────────
    print("28. Minimal view")
    win._toggle_left_sidebar()
    win._toggle_right_sidebar()
    wait(300)
    grab(1500, "Minimal view")

    # ── 29. Restore sidebars ────────────────────────────────
    print("29. Restore sidebars")
    win._toggle_left_sidebar()
    win._toggle_right_sidebar()
    wait(300)
    grab(1000, "Sidebars restored")

    # ── 30. Fullscreen mode ─────────────────────────────────
    print("30. Fullscreen mode")
    win.showFullScreen()
    wait(600)
    grab(1500, "Fullscreen mode")

    # Restore from fullscreen
    win.showMaximized()
    wait(600)

    # ── 31. Switch to light theme ───────────────────────────
    print("31. Light theme")
    apply_light_theme(app)
    win._refresh_themed_styles()
    win.tool_panel.refresh_theme()
    win.canvas.refresh_theme()
    if hasattr(win, 'perf_monitor'):
        win.perf_monitor.refresh_theme()
    if hasattr(win, 'minimap'):
        win.minimap.refresh_theme()
    wait(500)
    grab(2000, "Light theme")

    # ── 32. Zoom into a ROI in light theme ──────────────────
    print("32. Zoomed in light theme")
    select_roi(60)
    zoom_to_roi(60, padding=200)
    grab(1500, "Light theme zoomed")

    # ── 33. Back to dark theme ──────────────────────────────
    print("33. Back to dark theme")
    apply_dark_theme(app)
    win._refresh_themed_styles()
    win.tool_panel.refresh_theme()
    win.canvas.refresh_theme()
    if hasattr(win, 'perf_monitor'):
        win.perf_monitor.refresh_theme()
    if hasattr(win, 'minimap'):
        win.minimap.refresh_theme()
    wait(500)

    # ── 34. Final overview ──────────────────────────────────
    print("34. Final overview")
    win.canvas.fit_to_window()
    wait(300)
    grab(3000, "Final overview")

    # ── Save durations manifest ─────────────────────────────
    manifest = os.path.join(FRAME_DIR, "durations.txt")
    with open(manifest, "w") as f:
        for dur in durations:
            f.write(f"{dur}\n")

    print(f"\n=== Done: {len(frames)} frames saved to {FRAME_DIR}/ ===")
    print(f"  Total duration: {sum(durations)/1000:.1f}s")
    print(f"  Run assemble_gif.py to create the final GIF.")

    app.quit()


QTimer.singleShot(800, run)
app.exec()
