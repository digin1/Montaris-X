"""Quick peak memory measurement for Transform All operations."""
import os, sys, time, threading, numpy as np, gc
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import psutil
from unittest.mock import patch
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF, Qt
from montaris.app import MontarisApp, apply_dark_theme
from montaris.layers import ImageLayer
from montaris.tools.transform import TransformAllTool
from montaris.core.roi_transform import compute_handles
from montaris.core.rle import rle_encode
from montaris.io.image_io import load_image

proc = psutil.Process()

# Peak tracker thread (samples every 10ms)
peak_rss = [0]
stop_flag = threading.Event()
def track_peak():
    while not stop_flag.is_set():
        rss = proc.memory_info().rss
        if rss > peak_rss[0]:
            peak_rss[0] = rss
        time.sleep(0.01)

t = threading.Thread(target=track_peak, daemon=True)
t.start()

app = QApplication.instance() or QApplication(sys.argv)
apply_dark_theme(app)
win = MontarisApp()
win.show()
QApplication.processEvents()

img = load_image("test.tif")
win.layer_stack.set_image(ImageLayer("test", img))
win.canvas.refresh_image()
win.canvas.fit_to_window()
QApplication.processEvents()
with patch.object(win, "_ask_replace_or_keep", return_value="keep"):
    win.import_roi_zip("test.zip")
QApplication.processEvents()

n = len(win.layer_stack.roi_layers)
first = win.layer_stack.roi_layers[0]
win.canvas.set_active_layer(first)
QApplication.processEvents()

baseline = proc.memory_info().rss / (1024**2)
peak_rss[0] = proc.memory_info().rss
print(f"Baseline after load: {baseline:.0f} MB  ({n} ROIs)")
print()

overall_peak = baseline

def do_transform(handle_type, dx, dy):
    global overall_peak
    peak_rss[0] = proc.memory_info().rss
    tool = TransformAllTool(win)
    canvas = win.canvas
    layer = canvas._active_layer
    rois = list(win.layer_stack.roi_layers)
    tool.on_press(QPointF(50, 50), layer, canvas)
    QApplication.processEvents()
    handles = compute_handles(tool._bbox)
    h = next(x for x in handles if x.handle_type == handle_type)
    tool._active_handle = h
    tool._start_pos = QPointF(h.x, h.y)
    tool._dragging = True
    tool._target_layers = rois
    tool._shift_held = False
    tool._snapshots = {}
    tool._snap_bboxes = {}
    for l in rois:
        lid = id(l)
        sb = l.get_bbox()
        tool._snap_bboxes[lid] = sb
        if lid not in tool._session_snapshots:
            if sb is not None:
                if l.is_compressed:
                    from montaris.core.rle import rle_decode_crop
                    crop_arr = rle_decode_crop(l._rle_data, l._mask_shape, sb)
                else:
                    crop_arr = l.mask[sb[0]:sb[1], sb[2]:sb[3]]
                tool._session_snapshots[lid] = rle_encode(crop_arr)
            else:
                tool._session_snapshots[lid] = (b"", (0, 0))
            tool._session_bboxes[lid] = sb
            tool._snapshots[lid] = (l, None, sb)
        else:
            crop = l.mask[sb[0]:sb[1], sb[2]:sb[3]].copy() if sb else None
            tool._snapshots[lid] = (l, crop, sb)
    tool._create_previews(canvas)
    QApplication.processEvents()
    end = QPointF(h.x + dx, h.y + dy)
    tool.on_move(end, layer, canvas)
    QApplication.processEvents()
    tool.on_release(end, layer, canvas)
    QApplication.processEvents()
    deadline = time.perf_counter() + 60
    while getattr(tool, "_applying", False) and time.perf_counter() < deadline:
        QApplication.processEvents()
        time.sleep(0.05)
    after = proc.memory_info().rss / (1024**2)
    pk = peak_rss[0] / (1024**2)
    if pk > overall_peak:
        overall_peak = pk
    return after, pk

# Test each operation type
for label, ht, dx, dy in [
    ("Scale (br +10,+10)", "br", 10, 10),
    ("Scale (tl -10,-10)", "tl", -10, -10),
    ("Rotate (15px)", "rotate", 15, 0),
]:
    after, pk = do_transform(ht, dx, dy)
    print(f"  {label:30s}  after={after:,.0f} MB  peak={pk:,.0f} MB  delta={pk - baseline:+,.0f} MB")

# Undo peak
peak_rss[0] = proc.memory_info().rss
win.undo_stack.undo()
QApplication.processEvents()
undo_after = proc.memory_info().rss / (1024**2)
undo_peak = peak_rss[0] / (1024**2)
if undo_peak > overall_peak:
    overall_peak = undo_peak
label = "Undo"
print(f"  {label:30s}  after={undo_after:,.0f} MB  peak={undo_peak:,.0f} MB  delta={undo_peak - baseline:+,.0f} MB")

# Multi-transform peak
gc.collect()
peak_rss[0] = proc.memory_info().rss
for i in range(5):
    do_transform("br", 3, 3)
multi_peak = peak_rss[0] / (1024**2)
multi_after = proc.memory_info().rss / (1024**2)
if multi_peak > overall_peak:
    overall_peak = multi_peak
label = "5x sequential scale"
print(f"  {label:30s}  after={multi_after:,.0f} MB  peak={multi_peak:,.0f} MB  delta={multi_peak - baseline:+,.0f} MB")

stop_flag.set()
print()
print(f"OVERALL PEAK RSS: {overall_peak:,.0f} MB  (baseline: {baseline:,.0f} MB, delta: {overall_peak - baseline:+,.0f} MB)")

from montaris.core.workers import shutdown_pool
shutdown_pool()
