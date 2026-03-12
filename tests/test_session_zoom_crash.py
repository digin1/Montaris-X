#!/usr/bin/env python3
"""Reproduce crash: zoom in/out after ROI load triggers auto-save.

Run:  LD_LIBRARY_PATH="" python3 tests/test_session_zoom_crash.py
"""

import sys, os, threading, traceback, faulthandler, shutil

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_project_root)
sys.path.insert(0, _project_root)

# Cross-platform wall-clock timeout (45s)
def _timeout_kill():
    print("\n[TIMEOUT] 45s exceeded — likely hung", file=sys.stderr)
    os._exit(99)

_timeout_timer = threading.Timer(45, _timeout_kill)
_timeout_timer.daemon = True
_timeout_timer.start()
faulthandler.enable(all_threads=True)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from montaris.app import apply_dark_theme, MontarisApp
from montaris.io.image_io import load_image_stack

TIF = os.path.join(_project_root, "test.tif")
ZIP = os.path.join(_project_root, "test.zip")
FAILURES = []


def _fail(msg):
    FAILURES.append(msg)
    print(f"  FAIL: {msg}", file=sys.stderr)

def _ok(msg):
    print(f"  OK: {msg}", file=sys.stderr)


def simulate_zoom(canvas, n=20):
    """Zoom in/out by calling canvas.scale() + processEvents (triggers repaint)."""
    for i in range(n):
        factor = 1.15 if i % 2 == 0 else 1 / 1.15
        try:
            canvas.scale(factor, factor)
            canvas.viewport_changed.emit()
            QApplication.processEvents()
        except Exception:
            _fail(f"Zoom step {i} crashed:\n{traceback.format_exc()}")
            return False
    return True


def run(w, app):
    folder = os.path.dirname(TIF)

    def cleanup():
        for d in os.listdir(folder):
            if d.startswith("session_"):
                shutil.rmtree(os.path.join(folder, d), ignore_errors=True)

    def step1_load():
        print("[1] Loading image at ds=4...", file=sys.stderr)
        cleanup()
        try:
            channels = load_image_stack(TIF)
            w._initial_session_saved = False
            for name, data in channels:
                w._load_single_channel(name, data, 4, [], image_path=TIF)
            _ok(f"Loaded, shape={w.layer_stack.image_layer.data.shape}")
        except Exception:
            _fail(f"Load:\n{traceback.format_exc()}")
        QTimer.singleShot(100, step2_import_zip)

    def step2_import_zip():
        print("[2] Importing ROI ZIP...", file=sys.stderr)
        if not os.path.exists(ZIP):
            _fail("test.zip not found — skipping ZIP import, adding manual ROIs")
            from montaris.layers import ROILayer
            img = w.layer_stack.image_layer
            h, wd = img.data.shape[:2]
            for i in range(20):
                roi = ROILayer(f"ROI {i+1}", wd, h)
                y0 = (i * 30) % (h - 30)
                x0 = (i * 40) % (wd - 30)
                roi.mask[y0:y0+30, x0:x0+30] = 255
                roi.compress()
                w.layer_stack.insert_roi(i, roi)
            w.canvas.refresh_overlays()
            w.layer_panel.refresh()
            QTimer.singleShot(100, step3_zoom_immediately)
            return
        try:
            w.import_roi_zip(ZIP)
            n = len(w.layer_stack.roi_layers)
            _ok(f"Imported {n} ROIs")
        except Exception:
            _fail(f"Import ZIP:\n{traceback.format_exc()}")
        # Zoom immediately — before auto-save fires (it's at 500ms)
        QTimer.singleShot(50, step3_zoom_immediately)

    def step3_zoom_immediately():
        print("[3] Zooming BEFORE auto-save fires...", file=sys.stderr)
        try:
            ok = simulate_zoom(w.canvas, 10)
            if ok:
                _ok("Pre-autosave zoom OK")
        except Exception:
            _fail(f"Pre-autosave zoom:\n{traceback.format_exc()}")
        # Now wait for auto-save to fire (500ms timer) and zoom during it
        QTimer.singleShot(400, step4_zoom_during_autosave)

    def step4_zoom_during_autosave():
        print("[4] Zooming DURING auto-save window...", file=sys.stderr)
        try:
            ok = simulate_zoom(w.canvas, 30)
            if ok:
                _ok("During-autosave zoom OK")
        except Exception:
            _fail(f"During-autosave zoom:\n{traceback.format_exc()}")
        QTimer.singleShot(2000, step5_zoom_after_autosave)

    def step5_zoom_after_autosave():
        print("[5] Zooming AFTER auto-save...", file=sys.stderr)
        saved = getattr(w, '_initial_session_saved', False)
        print(f"     _initial_session_saved = {saved}", file=sys.stderr)
        try:
            ok = simulate_zoom(w.canvas, 30)
            if ok:
                _ok("Post-autosave zoom OK")
        except Exception:
            _fail(f"Post-autosave zoom:\n{traceback.format_exc()}")
        QTimer.singleShot(100, step6_manual_save_then_zoom)

    def step6_manual_save_then_zoom():
        print("[6] Manual save + immediate zoom...", file=sys.stderr)
        try:
            w.save_session_progress()
            _ok("save_session_progress returned")
            # Zoom right after save starts (background thread running)
            ok = simulate_zoom(w.canvas, 20)
            if ok:
                _ok("Post-manual-save zoom OK")
        except Exception:
            _fail(f"Manual save+zoom:\n{traceback.format_exc()}")
        QTimer.singleShot(3000, step7_done)

    def step7_done():
        cleanup()
        print("\n" + "=" * 50, file=sys.stderr)
        if FAILURES:
            print(f"FAILED — {len(FAILURES)} failure(s):", file=sys.stderr)
            for f in FAILURES:
                print(f"  - {f}", file=sys.stderr)
        else:
            print("ALL STEPS PASSED — no crash on zoom", file=sys.stderr)
        print("=" * 50, file=sys.stderr)
        app.quit()

    QTimer.singleShot(300, step1_load)


if __name__ == "__main__":
    print("=" * 50, file=sys.stderr)
    print("Zoom-after-ROI-load crash test (headed)", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    app = QApplication(sys.argv)
    apply_dark_theme(app)
    w = MontarisApp()
    w.show()
    run(w, app)
    rc = app.exec()
    _timeout_timer.cancel()
    sys.exit(1 if FAILURES else rc)
