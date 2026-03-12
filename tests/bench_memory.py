"""Memory benchmarks for Montaris-X.

Run:  QT_QPA_PLATFORM=offscreen python tests/bench_memory.py
"""
import os
import sys
import gc
import time
import tracemalloc

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF

from montaris.core.rle import rle_encode, rle_decode
from montaris.layers import ImageLayer, ROILayer
from montaris.app import MontarisApp, apply_dark_theme

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIF_PATH = os.path.join(BASE, "test.tif")
ZIP_PATH = os.path.join(BASE, "test.zip")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def get_rss_mb():
    """Current RSS in MB (cross-platform)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0


class FakeCanvas:
    def __init__(self):
        self._selection_highlight_items = []
    def refresh_active_overlay(self, layer): pass
    def refresh_active_overlay_partial(self, layer, bbox): pass
    def _update_selection_highlights(self): pass
    def scene(self): return self
    def addRect(self, *a, **kw): return _FI()
    def addEllipse(self, *a, **kw): return _FI()
    def removeItem(self, item): pass

class _FI:
    def setZValue(self, z): pass
    def setVisible(self, v): pass


def main():
    if not os.path.exists(TIF_PATH) or not os.path.exists(ZIP_PATH):
        print("ERROR: Need test.tif and test.zip in project root")
        sys.exit(1)

    qapp = QApplication.instance() or QApplication(sys.argv)
    apply_dark_theme(qapp)

    print("=" * 72)
    print("MEMORY BENCHMARKS — Montaris-X")
    print("=" * 72)

    results = []

    # ── 1. Import 102 ROIs ────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("1. Import 102 ROIs: Peak Memory")
    print(f"{'─'*72}")

    from montaris.io.image_io import load_image
    from unittest.mock import patch

    gc.collect()
    rss_before = get_rss_mb()
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    app = MontarisApp()
    img_data = load_image(TIF_PATH)
    app.layer_stack.set_image(ImageLayer("test", img_data))
    with patch.object(app, '_ask_replace_or_keep', return_value='keep'):
        app.import_roi_zip(ZIP_PATH)
    qapp.processEvents()

    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    gc.collect()
    rss_after = get_rss_mb()

    alloc_before = sum(s.size for s in snap_before.statistics('filename'))
    alloc_after = sum(s.size for s in snap_after.statistics('filename'))
    delta_mb = (alloc_after - alloc_before) / (1024 * 1024)

    n_rois = len(app.layer_stack.roi_layers)
    print(f"  Loaded {n_rois} ROIs")
    print(f"  RSS: {rss_before:.0f} -> {rss_after:.0f} MB (+{rss_after - rss_before:.0f} MB)")
    print(f"  tracemalloc delta: {delta_mb:.1f} MB")
    results.append(("Import ROIs", f"+{rss_after - rss_before:.0f} MB RSS", ""))

    # ── 2. Brush painting with auto-overlap ───────────────────────────
    print(f"\n{'─'*72}")
    print("2. Brush Painting with Auto-Overlap: Memory Delta")
    print(f"{'─'*72}")

    from montaris.tools.brush import BrushTool

    target = None
    for roi in app.layer_stack.roi_layers:
        if roi.mask.any():
            target = roi
            break
    if target is None:
        target = app.layer_stack.roi_layers[0]

    app._auto_overlap = True
    canvas = FakeCanvas()

    gc.collect()
    rss_before = get_rss_mb()

    for i in range(5):
        tool = BrushTool(app)
        tool.size = 50
        cx = 5800 + i * 30
        cy = 2900 + i * 30
        tool.on_press(QPointF(cx, cy), target, canvas)
        tool.on_move(QPointF(cx + 10, cy + 10), target, canvas)
        tool.on_release(QPointF(cx + 10, cy + 10), target, canvas)

    gc.collect()
    rss_after = get_rss_mb()
    delta = rss_after - rss_before
    status = PASS if delta < 200 else FAIL
    print(f"  5 strokes with auto-overlap: RSS delta = {delta:.0f} MB  [{status}]")
    results.append(("Auto-overlap 5 strokes", f"+{delta:.0f} MB", status))

    # ── 3. Compress/decompress cycle ──────────────────────────────────
    print(f"\n{'─'*72}")
    print("3. Compress/Decompress Cycle: RSS Impact")
    print(f"{'─'*72}")

    gc.collect()
    rss_before = get_rss_mb()

    active = app.canvas._active_layer if hasattr(app.canvas, '_active_layer') else target
    app.layer_stack.compress_inactive(active)
    gc.collect()
    rss_compressed = get_rss_mb()

    # Decompress all
    for roi in app.layer_stack.roi_layers:
        if roi.is_compressed:
            _ = roi.mask
    gc.collect()
    rss_decompressed = get_rss_mb()

    saved = rss_before - rss_compressed
    print(f"  Before compress:  {rss_before:.0f} MB")
    print(f"  After compress:   {rss_compressed:.0f} MB (saved {saved:.0f} MB)")
    print(f"  After decompress: {rss_decompressed:.0f} MB")
    status = PASS if saved > 0 else FAIL
    results.append(("Compress saves", f"{saved:.0f} MB", status))

    # ── 4. 10 rapid brush strokes: undo memory ───────────────────────
    print(f"\n{'─'*72}")
    print("4. 10 Rapid Brush Strokes: Undo Memory")
    print(f"{'─'*72}")

    target.mask[:] = 0
    app.undo_stack._stack.clear()
    app.undo_stack._index = -1
    app.undo_stack._total_bytes = 0
    app._auto_overlap = False

    gc.collect()
    rss_before = get_rss_mb()

    tool = BrushTool(app)
    tool.size = 80

    for i in range(10):
        cx = 1000 + i * 400
        cy = 3000
        tool.on_press(QPointF(cx, cy), target, canvas)
        tool.on_move(QPointF(cx + 50, cy + 50), target, canvas)
        tool.on_release(QPointF(cx + 100, cy + 100), target, canvas)

    gc.collect()
    rss_after = get_rss_mb()
    undo_kb = app.undo_stack._total_bytes / 1024
    h, w = target.mask.shape
    mask_kb = h * w / 1024

    print(f"  Undo stack total: {undo_kb:.0f} KB")
    print(f"  Full mask size:   {mask_kb:.0f} KB")
    print(f"  RSS delta:        {rss_after - rss_before:.0f} MB")

    ratio = undo_kb / mask_kb if mask_kb > 0 else 0
    # 10 bbox-crops should be much less than 10 full masks
    status = PASS if ratio < 5.0 else FAIL
    print(f"  Undo/mask ratio:  {ratio:.2f}x  [{status}]")
    results.append(("10 strokes undo", f"{undo_kb:.0f} KB ({ratio:.2f}x mask)", status))

    # ── 5. Long diagonal stroke: snapshot size ────────────────────────
    print(f"\n{'─'*72}")
    print("5. Long Diagonal Stroke: Snapshot Size")
    print(f"{'─'*72}")

    target.mask[:] = 0
    app.undo_stack._stack.clear()
    app.undo_stack._index = -1
    app.undo_stack._total_bytes = 0

    tool = BrushTool(app)
    tool.size = 30
    points = [QPointF(int(w * i / 100), int(h * i / 100)) for i in range(101)]
    tool.on_press(points[0], target, canvas)
    for p in points[1:-1]:
        tool.on_move(p, target, canvas)
    tool.on_release(points[-1], target, canvas)

    undo_kb = app.undo_stack._total_bytes / 1024
    print(f"  Diagonal stroke snapshot: {undo_kb:.0f} KB")
    print(f"  Full mask:                {mask_kb:.0f} KB")
    ratio = undo_kb / mask_kb if mask_kb > 0 else 0
    # Diagonal bbox is large but still should be < full mask (it's RLE compressed)
    status = PASS if ratio < 1.5 else FAIL
    print(f"  Ratio: {ratio:.2f}x  [{status}]")
    results.append(("Diagonal snapshot", f"{undo_kb:.0f} KB ({ratio:.2f}x)", status))

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Operation':<30} {'Metric':>25} {'Status':>8}")
    print(f"  {'─'*30} {'─'*25} {'─'*8}")
    for op, metric, status in results:
        print(f"  {op:<30} {metric:>25} {status:>8}")
    print(f"{'='*72}")

    app.close()


if __name__ == "__main__":
    main()
