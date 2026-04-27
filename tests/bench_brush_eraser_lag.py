"""Headed lag test: brush + eraser stroke latency on test.tif + test.zip.

Reproduces the user reports of lag while brushing/erasing. Loads the real
test image, picks an existing ROI, and drives a sequence of synthetic
strokes through ``BrushTool`` / ``EraserTool`` while timing every press,
move, and release. Reports per-stroke percentiles so regressions are
visible at a glance.

All dialogs that could pop up during load (replace-or-keep, shape
warnings, etc.) are patched away so the test runs unattended.

Usage (headed — recommended so the user can watch):

    LD_LIBRARY_PATH="" .venv/bin/python tests/test_brush_eraser_lag.py

Optional CLI flags:
    --sizes 50,200,500     brush radii to sweep (default: 50,200,500)
    --strokes 30           strokes per size+tool combo (default: 30)
    --steps 12             move events per stroke (default: 12)
    --csv path.csv         dump raw timings to a CSV
    --no-show              skip window.show() (won't be visible but works)
"""
from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time
import traceback
from contextlib import ExitStack
from unittest.mock import patch

# Project root so the test runs from a checkout without installing.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)


# Mirror montaris.app.main()'s pre-Qt setup BEFORE QApplication is
# constructed. We literally call the real launcher's selection helper
# with the same default args, so the test environment matches exactly
# what ``python main.py`` produces (NVIDIA offload + xcb on Wayland).
# Has to happen before any PySide6 import that creates a Qt context.
def _bootstrap_gpu_and_platform():
    if "MONTARIS_GPU_PREFERENCE" in os.environ:
        return  # already bootstrapped (e.g. by the real launcher)
    try:
        from argparse import Namespace
        from montaris.app import _apply_gpu_selection  # noqa: PLC0415

        # Default-args equivalent of ``python main.py`` with no flags:
        # gpu=None (means enable=True downstream), no index, no platform
        # override, no list-gpus.
        _apply_gpu_selection(Namespace(
            gpu=None, gpu_index=None, list_gpus=False, platform=None,
        ))
    except SystemExit:
        # _apply_gpu_selection's --list-gpus path calls sys.exit; we
        # never request it, but defend just in case.
        raise
    except Exception as e:
        print(f"  (gpu/platform bootstrap skipped: {e})", flush=True)
    print(
        f"  [env] DISPLAY={os.environ.get('DISPLAY', '')!r}  "
        f"XDG_SESSION_TYPE={os.environ.get('XDG_SESSION_TYPE', '')!r}  "
        f"QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM', '')!r}",
        flush=True,
    )


_bootstrap_gpu_and_platform()


import numpy as np  # noqa: E402
from PySide6.QtCore import QPointF, QTimer  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QDialog, QFileDialog, QInputDialog, QMessageBox,
)

from montaris.app import MontarisApp, apply_dark_theme  # noqa: E402
from montaris.layers import ImageLayer  # noqa: E402

TIF_PATH = os.path.join(_PROJECT_ROOT, "test.tif")
ZIP_PATH = os.path.join(_PROJECT_ROOT, "test.zip")


# ── Dialog suppression ────────────────────────────────────────────────
# Anything modal that the user said they don't want to click is mocked
# here. We patch the static methods globally so even code paths the test
# doesn't directly invoke (e.g. an internal call deep in import_roi_zip)
# can't trigger a real dialog.

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
    # Any QDialog.exec() in unfamiliar paths defaults to Accepted.
    stack.enter_context(patch.object(QDialog, "exec",
                                     return_value=QDialog.Accepted))


# ── Stroke timing ─────────────────────────────────────────────────────

def time_stroke(tool, layer, canvas, app_qt, start, end, steps):
    """Drive one press → N moves → release through ``tool`` and time it.

    Mirrors real interactive Qt behaviour: events are NOT pumped between
    individual move calls (Qt buffers move events and the canvas's dirty
    timer coalesces them into batched flushes), only after the press and
    after the release. ``move_ms`` therefore reflects the brush's own
    per-stamp work; ``release_ms`` captures the final flush + edge
    rebuild, which is what the user feels as the end-of-stroke pause.
    """
    move_ms = []
    t0_total = time.perf_counter()

    t0 = time.perf_counter()
    tool.on_press(start, layer, canvas)
    press_ms = (time.perf_counter() - t0) * 1000

    for i in range(steps):
        t = (i + 1) / steps
        pt = QPointF(start.x() + (end.x() - start.x()) * t,
                     start.y() + (end.y() - start.y()) * t)
        t0 = time.perf_counter()
        tool.on_move(pt, layer, canvas)
        move_ms.append((time.perf_counter() - t0) * 1000)

    # Flush any throttled-paint timer that the moves scheduled, so the
    # release-time rebuild has a clean queue to work on. This replaces
    # the per-move processEvents calls that artificially fired the timer
    # 4× per stroke and skewed the previous numbers.
    app_qt.processEvents()

    t0 = time.perf_counter()
    tool.on_release(end, layer, canvas)
    app_qt.processEvents()
    release_ms = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - t0_total) * 1000
    return press_ms, move_ms, release_ms, total_ms


def percentile(values, p):
    """p in [0, 100]."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarise(label, totals, all_moves):
    print(f"\n  {label}")
    print(f"    strokes: {len(totals)}    moves: {len(all_moves)}")
    print(
        f"    stroke total ms — "
        f"p50={percentile(totals, 50):.1f}  "
        f"p90={percentile(totals, 90):.1f}  "
        f"p99={percentile(totals, 99):.1f}  "
        f"max={max(totals):.1f}  "
        f"mean={statistics.mean(totals):.1f}"
    )
    print(
        f"    per-move ms   — "
        f"p50={percentile(all_moves, 50):.1f}  "
        f"p90={percentile(all_moves, 90):.1f}  "
        f"p99={percentile(all_moves, 99):.1f}  "
        f"max={max(all_moves):.1f}  "
        f"mean={statistics.mean(all_moves):.1f}"
    )


# ── Test driver ───────────────────────────────────────────────────────

_results: list[tuple[str, dict]] = []
_app_window: MontarisApp | None = None


def run_test(opts):
    global _app_window
    try:
        _run_test_inner(opts)
    except Exception:
        traceback.print_exc()
        _results.append(("EXCEPTION", {}))
    finally:
        if _app_window is not None:
            _app_window.close()
        QApplication.instance().quit()


def _pump(app_qt, seconds):
    """Pump Qt events for ``seconds`` so the compositor can paint between
    blocking numpy work. Without this the window never gets mapped before
    the strokes finish and the user sees nothing.
    """
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app_qt.processEvents()
        time.sleep(0.02)


def _run_test_inner(opts):
    global _app_window

    if not os.path.exists(TIF_PATH) or not os.path.exists(ZIP_PATH):
        print(f"SKIP: missing {TIF_PATH} or {ZIP_PATH}")
        return

    # Refuse to run unattended when the platform is offscreen — the user
    # explicitly asked for a headed run so they can watch the strokes.
    qt_platform = os.environ.get("QT_QPA_PLATFORM", "").lower()
    if not opts.no_show and qt_platform == "offscreen":
        print(
            "ERROR: QT_QPA_PLATFORM=offscreen is set; the window won't be "
            "visible. Unset it (or pass --no-show to run anyway) and rerun."
        )
        _results.append(("offscreen-blocked", {"failed": True}))
        return

    app_qt = QApplication.instance()
    apply_dark_theme(app_qt)

    with ExitStack() as stack:
        _install_dialog_patches(stack)

        win = MontarisApp()
        _app_window = win
        if not opts.no_show:
            win.resize(1400, 900)
            # Position at a known on-screen location so an off-screen
            # geometry from a stale QSettings can't hide the window.
            win.move(80, 80)
            win.show()
            win.raise_()
            win.activateWindow()
            # Give the window manager time to actually map + paint the
            # window before any blocking numpy work starts.
            _pump(app_qt, 1.5)
            print(
                f"  [show] visible={win.isVisible()}  "
                f"active={win.isActiveWindow()}  "
                f"geometry={win.geometry().getRect()}  "
                f"qt_platform="
                f"{QApplication.platformName()!r}",
                flush=True,
            )

        # Load image directly (skip File > Open dialog).
        from montaris.io.image_io import load_image
        data = load_image(TIF_PATH)
        win.layer_stack.set_image(ImageLayer("test", data))
        win.canvas.refresh_image()
        win.canvas.fit_to_window()
        _pump(app_qt, 0.5)
        print(f"  loaded image {data.shape} {data.dtype}", flush=True)

        # Import ROIs without prompting for replace-or-keep.
        with patch.object(win, "_ask_replace_or_keep", return_value="keep"):
            win.import_roi_zip(ZIP_PATH)
        _pump(app_qt, 0.5)
        n_rois = len(win.layer_stack.roi_layers)
        print(f"  imported {n_rois} ROIs", flush=True)
        if n_rois == 0:
            _results.append(("no rois", {"failed": True}))
            return

        # Pick an ROI somewhere in the middle of the stack.
        target = win.layer_stack.roi_layers[len(win.layer_stack.roi_layers) // 2]
        win.canvas.set_active_layer(target)
        # Also drive the LayerPanel selection so the rest of the app sees it.
        win.layer_panel.refresh()
        _pump(app_qt, 0.3)
        h, w = target.shape
        print(f"  active ROI: {target.name}  shape={h}x{w}", flush=True)

        # Strokes are placed inside the ROI's bounding box so the brush
        # actually has work to do (bbox-based dirty regions are the
        # interesting case for lag).
        bbox = target.get_bbox()
        if bbox is None:
            cy, cx = h // 2, w // 2
        else:
            y1, y2, x1, x2 = bbox
            cy, cx = (y1 + y2) // 2, (x1 + x2) // 2

        sizes = [int(s) for s in opts.sizes.split(",") if s.strip()]
        rng = np.random.default_rng(0)

        all_rows = []  # (tool, size, stroke_idx, total_ms, max_move_ms)

        def run_tool(label, tool_name, has_size):
            print(f"\n  >>> {label}", flush=True)
            win.tool_panel._select_tool(tool_name)
            _pump(app_qt, 0.3)
            tool = win.active_tool
            for size in sizes:
                if has_size and hasattr(tool, "size"):
                    tool.size = size
                totals = []
                all_moves = []
                print(f"    [{label} size={size}] running "
                      f"{opts.strokes} strokes…", flush=True)
                for i in range(opts.strokes):
                    start = QPointF(cx + rng.integers(-200, 200),
                                    cy + rng.integers(-200, 200))
                    end = QPointF(start.x() + rng.integers(-150, 150),
                                  start.y() + rng.integers(-150, 150))
                    press, moves, release, total = time_stroke(
                        tool, target, win.canvas, app_qt,
                        start, end, opts.steps,
                    )
                    totals.append(total)
                    all_moves.extend(moves)
                    all_rows.append((label, size, i, total, max(moves)))
                    # Heartbeat so the user sees progress without waiting
                    # for the whole sweep to finish.
                    print(
                        f"      stroke {i + 1}/{opts.strokes}  "
                        f"total={total:6.0f} ms  "
                        f"max_move={max(moves):6.0f} ms",
                        flush=True,
                    )
                    # Yield to the compositor so each stroke is visible
                    # rather than the whole sweep running back-to-back.
                    _pump(app_qt, opts.between)
                summarise(f"{label} size={size}", totals, all_moves)

        run_tool("brush", "Brush", has_size=True)
        run_tool("eraser", "Eraser", has_size=True)

        if opts.csv:
            with open(opts.csv, "w") as f:
                f.write("tool,size,stroke_idx,total_ms,max_move_ms\n")
                for row in all_rows:
                    f.write(",".join(str(x) for x in row) + "\n")
            print(f"\n  raw timings → {opts.csv}")

        # Hold the window briefly so the user can see the result before
        # we tear down the QApplication.
        if not opts.no_show:
            t_end = time.perf_counter() + opts.hold
            while time.perf_counter() < t_end:
                app_qt.processEvents()
                time.sleep(0.05)

        gc.collect()
        _results.append(("ran", {"failed": False}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="50,200,500",
                   help="comma-separated brush radii (default 50,200,500)")
    p.add_argument("--strokes", type=int, default=30,
                   help="strokes per size+tool combo (default 30)")
    p.add_argument("--steps", type=int, default=12,
                   help="move events per stroke (default 12)")
    p.add_argument("--csv", default=None,
                   help="dump raw per-stroke timings to this CSV")
    p.add_argument("--hold", type=float, default=4.0,
                   help="seconds to hold the window after the run "
                        "(default 4.0)")
    p.add_argument("--between", type=float, default=0.15,
                   help="seconds to pump the event loop between strokes "
                        "so each stroke renders (default 0.15)")
    p.add_argument("--no-show", action="store_true",
                   help="don't show the window (still runs the timing)")
    opts = p.parse_args()

    app = QApplication(sys.argv[:1])  # don't pass argparse args to Qt
    QTimer.singleShot(100, lambda: run_test(opts))
    app.exec()
    sys.exit(0 if all(not r[1].get("failed", True) for r in _results) else 1)


if __name__ == "__main__":
    main()
