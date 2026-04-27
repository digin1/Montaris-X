"""Headed lag benchmark for stamp / rectangle / circle / polygon /
bucket-fill — the same per-tool measurement done for brush/eraser, run on
the real ``test.tif`` + ``test.zip`` data with all dialogs auto-bypassed.

Usage:
    LD_LIBRARY_PATH="" .venv/bin/python tests/bench_all_tools_lag.py

Optional flags:
    --commits 5      number of commit cycles per tool (default 5)
    --hold 4         seconds to hold the window after the run (default 4)
    --no-show        skip win.show()
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import traceback
from contextlib import ExitStack
from unittest.mock import patch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)


def _bootstrap_gpu_and_platform():
    """Match what `montaris.app.main` does so the window actually shows
    on Wayland+NVIDIA. Must run before any PySide6 import.
    """
    if "MONTARIS_GPU_PREFERENCE" in os.environ:
        return
    try:
        from argparse import Namespace
        from montaris.app import _apply_gpu_selection  # noqa: PLC0415
        _apply_gpu_selection(Namespace(
            gpu=None, gpu_index=None, list_gpus=False, platform=None,
        ))
    except Exception as e:
        print(f"  (gpu/platform bootstrap skipped: {e})", flush=True)


_bootstrap_gpu_and_platform()


import numpy as np  # noqa: E402
from PySide6.QtCore import QPointF, Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QDialog, QFileDialog, QInputDialog, QMessageBox,
)

from montaris.app import MontarisApp, apply_dark_theme  # noqa: E402
from montaris.layers import ImageLayer  # noqa: E402

TIF_PATH = os.path.join(_PROJECT_ROOT, "test.tif")
ZIP_PATH = os.path.join(_PROJECT_ROOT, "test.zip")


def _install_dialog_patches(stack: ExitStack):
    stack.enter_context(patch.object(QMessageBox, "warning",
                                     return_value=QMessageBox.Ok))
    stack.enter_context(patch.object(QMessageBox, "information",
                                     return_value=QMessageBox.Ok))
    stack.enter_context(patch.object(QMessageBox, "critical",
                                     return_value=QMessageBox.Ok))
    stack.enter_context(patch.object(QMessageBox, "question",
                                     return_value=QMessageBox.Yes))
    stack.enter_context(patch.object(QFileDialog, "getOpenFileName",
                                     return_value=("", "")))
    stack.enter_context(patch.object(QFileDialog, "getOpenFileNames",
                                     return_value=([], "")))
    stack.enter_context(patch.object(QFileDialog, "getSaveFileName",
                                     return_value=("", "")))
    stack.enter_context(patch.object(QInputDialog, "getItem",
                                     return_value=("", False)))
    stack.enter_context(patch.object(QInputDialog, "getText",
                                     return_value=("", False)))
    stack.enter_context(patch.object(QDialog, "exec",
                                     return_value=QDialog.Accepted))


def _pump(app_qt, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app_qt.processEvents()
        time.sleep(0.02)


def percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _summarise(label, totals):
    print(
        f"\n  {label}\n"
        f"    commits: {len(totals)}\n"
        f"    commit ms — "
        f"p50={percentile(totals, 50):.1f}  "
        f"p90={percentile(totals, 90):.1f}  "
        f"max={max(totals):.1f}  "
        f"mean={statistics.mean(totals):.1f}",
        flush=True,
    )


_app_window = None


def run_test(opts):
    global _app_window
    try:
        _run_test_inner(opts)
    except Exception:
        traceback.print_exc()
    finally:
        if _app_window is not None:
            _app_window.close()
        QApplication.instance().quit()


def _run_test_inner(opts):
    global _app_window

    if not os.path.exists(TIF_PATH) or not os.path.exists(ZIP_PATH):
        print(f"SKIP: missing {TIF_PATH} or {ZIP_PATH}")
        return

    # Refuse to run unattended when the platform is offscreen — the
    # whole point of this bench is to watch the tools commit headed.
    qt_platform = os.environ.get("QT_QPA_PLATFORM", "").lower()
    if not opts.no_show and qt_platform == "offscreen":
        print(
            "ERROR: QT_QPA_PLATFORM=offscreen is set; the window won't "
            "be visible. Unset it (or pass --no-show) and rerun."
        )
        return

    app_qt = QApplication.instance()
    apply_dark_theme(app_qt)

    with ExitStack() as stack:
        _install_dialog_patches(stack)

        win = MontarisApp()
        _app_window = win
        if not opts.no_show:
            win.resize(1400, 900)
            win.move(80, 80)
            win.show()
            win.raise_()
            _pump(app_qt, 1.0)

        from montaris.io.image_io import load_image
        data = load_image(TIF_PATH)
        win.layer_stack.set_image(ImageLayer("test", data))
        win.canvas.refresh_image()
        win.canvas.fit_to_window()
        _pump(app_qt, 0.5)

        with patch.object(win, "_ask_replace_or_keep", return_value="keep"):
            win.import_roi_zip(ZIP_PATH)
        _pump(app_qt, 0.5)

        # Pick a real ROI from the imports.
        target = win.layer_stack.roi_layers[
            len(win.layer_stack.roi_layers) // 2]
        win.canvas.set_active_layer(target)
        win.layer_panel.refresh()
        _pump(app_qt, 0.3)
        h, w = target.shape
        print(f"  loaded {data.shape}  active ROI {target.name} "
              f"({h}×{w}, bbox={target.get_bbox()})", flush=True)

        bbox = target.get_bbox()
        if bbox is None:
            cy, cx = h // 2, w // 2
        else:
            y1, y2, x1, x2 = bbox
            cy, cx = (y1 + y2) // 2, (x1 + x2) // 2

        rng = np.random.default_rng(0)

        # ── Stamp ────────────────────────────────────────────────────
        win.tool_panel._select_tool("Stamp")
        _pump(app_qt, 0.2)
        stamp = win.active_tool
        stamp.width = stamp.height = stamp.size = 200
        totals = []
        print(f"\n  >>> stamp size=200 ({opts.commits} commits)", flush=True)
        for i in range(opts.commits):
            sx = cx + int(rng.integers(-200, 200))
            sy = cy + int(rng.integers(-200, 200))
            t0 = time.perf_counter()
            stamp.on_press(QPointF(sx, sy), target, win.canvas)
            stamp.on_release(QPointF(sx, sy), target, win.canvas)
            app_qt.processEvents()
            ms = (time.perf_counter() - t0) * 1000
            totals.append(ms)
            print(f"      commit {i + 1}/{opts.commits}  total={ms:6.0f} ms",
                  flush=True)
            _pump(app_qt, 0.05)
        _summarise("stamp size=200", totals)

        # ── Rectangle ────────────────────────────────────────────────
        win.tool_panel._select_tool("Rectangle")
        _pump(app_qt, 0.2)
        rect = win.active_tool
        totals = []
        print(f"\n  >>> rectangle ({opts.commits} commits)", flush=True)
        for i in range(opts.commits):
            sx = cx + int(rng.integers(-200, 200))
            sy = cy + int(rng.integers(-200, 200))
            t0 = time.perf_counter()
            rect.on_press(QPointF(sx, sy), target, win.canvas)
            rect.on_release(QPointF(sx + 100, sy + 80), target, win.canvas)
            app_qt.processEvents()
            ms = (time.perf_counter() - t0) * 1000
            totals.append(ms)
            print(f"      commit {i + 1}/{opts.commits}  total={ms:6.0f} ms",
                  flush=True)
            _pump(app_qt, 0.05)
        _summarise("rectangle 100×80", totals)

        # ── Circle ───────────────────────────────────────────────────
        win.tool_panel._select_tool("Circle")
        _pump(app_qt, 0.2)
        circ = win.active_tool
        totals = []
        print(f"\n  >>> circle ({opts.commits} commits)", flush=True)
        for i in range(opts.commits):
            sx = cx + int(rng.integers(-200, 200))
            sy = cy + int(rng.integers(-200, 200))
            t0 = time.perf_counter()
            circ.on_press(QPointF(sx, sy), target, win.canvas)
            circ.on_release(QPointF(sx + 60, sy + 60), target, win.canvas)
            app_qt.processEvents()
            ms = (time.perf_counter() - t0) * 1000
            totals.append(ms)
            print(f"      commit {i + 1}/{opts.commits}  total={ms:6.0f} ms",
                  flush=True)
            _pump(app_qt, 0.05)
        _summarise("circle r≈60", totals)

        # ── Polygon ──────────────────────────────────────────────────
        win.tool_panel._select_tool("Polygon")
        _pump(app_qt, 0.2)
        poly = win.active_tool
        totals = []
        # Times only ``finish()`` — the on_press cycles building the
        # polygon are O(verts) hash-table appends and not the user-felt
        # cost. The label reflects that.
        print(f"\n  >>> polygon finish ({opts.commits} commits)", flush=True)
        for i in range(opts.commits):
            sx = cx + int(rng.integers(-200, 200))
            sy = cy + int(rng.integers(-200, 200))
            # 5 vertices spaced wide enough that the polygon's
            # close-marker check (CLOSE_DISTANCE=14 screen px) won't
            # auto-finish on the third press at fit-to-window zoom
            # (fresh-eyes L4 — was brittle with 4 vertices at 80×80).
            poly.on_press(QPointF(sx, sy), target, win.canvas)
            poly.on_press(QPointF(sx + 200, sy), target, win.canvas)
            poly.on_press(QPointF(sx + 200, sy + 200), target, win.canvas)
            poly.on_press(QPointF(sx + 100, sy + 280), target, win.canvas)
            poly.on_press(QPointF(sx, sy + 200), target, win.canvas)
            t0 = time.perf_counter()
            poly.finish()
            app_qt.processEvents()
            ms = (time.perf_counter() - t0) * 1000
            totals.append(ms)
            print(f"      commit {i + 1}/{opts.commits}  total={ms:6.0f} ms",
                  flush=True)
            _pump(app_qt, 0.05)
        _summarise("polygon finish (5-vertex)", totals)

        # ── Bucket Fill ──────────────────────────────────────────────
        win.tool_panel._select_tool("Bucket Fill")
        _pump(app_qt, 0.2)
        bucket = win.active_tool
        totals = []
        # Seed the bucket-fill from a known unpainted background pixel.
        # We use the canvas mid-point and skip the press if the pixel is
        # already painted (the fill would be a no-op).
        print(f"\n  >>> bucket-fill ({opts.commits} commits)", flush=True)
        for i in range(opts.commits):
            sx = int(cx + rng.integers(-300, 300))
            sy = int(cy + rng.integers(-300, 300))
            sx = max(1, min(w - 2, sx))
            sy = max(1, min(h - 2, sy))
            t0 = time.perf_counter()
            bucket.on_press(QPointF(sx, sy), target, win.canvas)
            app_qt.processEvents()
            ms = (time.perf_counter() - t0) * 1000
            totals.append(ms)
            print(f"      commit {i + 1}/{opts.commits}  total={ms:6.0f} ms",
                  flush=True)
            _pump(app_qt, 0.05)
        _summarise("bucket-fill", totals)

        if not opts.no_show:
            _pump(app_qt, opts.hold)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--commits", type=int, default=5,
                   help="commits per tool (default 5)")
    p.add_argument("--hold", type=float, default=4.0,
                   help="seconds to hold window after the run (default 4)")
    p.add_argument("--no-show", action="store_true")
    opts = p.parse_args()

    app = QApplication(sys.argv[:1])
    QTimer.singleShot(100, lambda: run_test(opts))
    app.exec()


if __name__ == "__main__":
    main()
