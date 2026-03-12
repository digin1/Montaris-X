"""Headed test: verify brush painting memory stays bounded.

Loads test.tif + test.zip, selects an ROI, simulates brush strokes,
and checks that memory doesn't explode (stays under 2GB above baseline).

Run directly:
    python tests/test_brush_memory.py
"""

import sys
import os
import gc
import traceback

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_project_root)
sys.path.insert(0, _project_root)

import psutil
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QPointF

from montaris.app import apply_dark_theme, MontarisApp
from montaris.layers import ROILayer

TIF_PATH = os.path.join(_project_root, "test.tif")
ZIP_PATH = os.path.join(_project_root, "test.zip")

_results = []
_app_window = None


def get_rss_mb():
    """Current process RSS in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def log(msg):
    print(f"  {msg}", flush=True)


def run_test():
    global _app_window
    try:
        _run_test_inner()
    except Exception:
        traceback.print_exc()
        _results.append(("EXCEPTION", False))
    finally:
        if _app_window:
            _app_window.close()
        QApplication.instance().quit()


def _run_test_inner():
    global _app_window

    if not os.path.exists(TIF_PATH) or not os.path.exists(ZIP_PATH):
        log("SKIP: test.tif / test.zip not found")
        _results.append(("test data missing", True))
        return

    app_qt = QApplication.instance()
    apply_dark_theme(app_qt)
    win = MontarisApp()
    _app_window = win
    win.show()
    app_qt.processEvents()

    # Load image
    from montaris.io.image_io import load_image
    data = load_image(TIF_PATH)
    from montaris.layers import ImageLayer
    win.layer_stack.set_image(ImageLayer("test", data))
    win.canvas.fit_to_window()
    app_qt.processEvents()

    gc.collect()
    baseline_mb = get_rss_mb()
    log(f"After image load: {baseline_mb:.0f} MB")

    # Import ROI ZIP
    win.import_roi_zip(ZIP_PATH)
    app_qt.processEvents()
    gc.collect()
    after_import_mb = get_rss_mb()
    n_rois = len(win.layer_stack.roi_layers)
    log(f"After importing {n_rois} ROIs: {after_import_mb:.0f} MB")

    # Check all ROIs are compressed
    compressed_count = sum(1 for r in win.layer_stack.roi_layers if r.is_compressed)
    log(f"Compressed ROIs: {compressed_count}/{n_rois}")
    _results.append((f"ROIs compressed after import: {compressed_count}/{n_rois}",
                      compressed_count == n_rois))

    # Select first ROI
    first_roi = win.layer_stack.roi_layers[0]
    win.canvas.set_active_layer(first_roi)
    app_qt.processEvents()

    # Switch to brush tool
    win.tool_panel._select_tool('Brush')
    app_qt.processEvents()
    gc.collect()
    after_select_mb = get_rss_mb()
    log(f"After selecting ROI + brush tool: {after_select_mb:.0f} MB")

    # Simulate 20 brush strokes
    brush = win.active_tool
    h, w = first_roi.shape
    for stroke in range(20):
        cx = 100 + stroke * 50
        cy = 100 + stroke * 30
        pos_start = QPointF(cx, cy)
        pos_end = QPointF(cx + 40, cy + 20)
        brush.on_press(pos_start, first_roi, win.canvas)
        for step in range(5):
            t = (step + 1) / 5
            pos = QPointF(
                cx + (pos_end.x() - cx) * t,
                cy + (pos_end.y() - cy) * t,
            )
            brush.on_move(pos, first_roi, win.canvas)
        brush.on_release(pos_end, first_roi, win.canvas)
        app_qt.processEvents()

    gc.collect()
    after_paint_mb = get_rss_mb()
    log(f"After 20 brush strokes: {after_paint_mb:.0f} MB")

    # Check compressed state of non-active ROIs
    active = win.canvas._active_layer
    decompressed = [r for r in win.layer_stack.roi_layers
                    if r is not active and r._mask is not None]
    log(f"Non-active decompressed ROIs: {len(decompressed)}")
    _results.append((f"Non-active decompressed after painting: {len(decompressed)}",
                      len(decompressed) == 0))

    # Memory check: should not exceed baseline + 1.5GB
    paint_delta = after_paint_mb - baseline_mb
    log(f"Memory delta from baseline: {paint_delta:.0f} MB")
    _results.append((f"Memory delta {paint_delta:.0f} MB < 1500 MB",
                      paint_delta < 1500))

    # Test with auto-overlap enabled
    win._auto_overlap = True
    for stroke in range(5):
        cx = 200 + stroke * 50
        cy = 200 + stroke * 30
        pos_start = QPointF(cx, cy)
        pos_end = QPointF(cx + 40, cy + 20)
        brush.on_press(pos_start, first_roi, win.canvas)
        brush.on_move(pos_end, first_roi, win.canvas)
        brush.on_release(pos_end, first_roi, win.canvas)
        app_qt.processEvents()

    gc.collect()
    after_overlap_mb = get_rss_mb()
    decompressed_after_overlap = [r for r in win.layer_stack.roi_layers
                                   if r is not active and r._mask is not None]
    log(f"After auto-overlap strokes: {after_overlap_mb:.0f} MB")
    log(f"Non-active decompressed after overlap: {len(decompressed_after_overlap)}")
    _results.append((f"Non-active decompressed after overlap: {len(decompressed_after_overlap)}",
                      len(decompressed_after_overlap) == 0))

    overlap_delta = after_overlap_mb - baseline_mb
    log(f"Memory delta after overlap: {overlap_delta:.0f} MB")
    _results.append((f"Memory delta after overlap {overlap_delta:.0f} MB < 1500 MB",
                      overlap_delta < 1500))


def main():
    app = QApplication(sys.argv)
    QTimer.singleShot(100, run_test)
    app.exec()

    print("\n=== RESULTS ===")
    all_pass = True
    for desc, passed in _results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")
        if not passed:
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
