"""CPU performance benchmarks for Montaris-X.

Run:  QT_QPA_PLATFORM=offscreen python tests/bench_cpu.py
"""
import os
import sys
import time
import gc

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF

from montaris.core.rle import rle_encode, rle_decode, rle_decode_crop
from montaris.layers import ImageLayer, ROILayer
from montaris.app import MontarisApp, apply_dark_theme

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIF_PATH = os.path.join(BASE, "test.tif")
ZIP_PATH = os.path.join(BASE, "test.zip")


class _FakeTransform:
    def m11(self): return 1.0

class FakeCanvas:
    def __init__(self):
        self._selection_highlight_items = []
    def refresh_active_overlay(self, layer): pass
    def refresh_active_overlay_partial(self, layer, bbox): pass
    def _update_selection_highlights(self): pass
    def transform(self): return _FakeTransform()
    def scene(self): return self
    def addRect(self, *a, **kw): return _FI()
    def addEllipse(self, *a, **kw): return _FI()
    def removeItem(self, item): pass

class _FI:
    def setZValue(self, z): pass
    def setVisible(self, v): pass


def bench(name, func, n=1):
    """Run func n times, return (avg_ms, total_ms)."""
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(n):
        func()
    total = (time.perf_counter() - t0) * 1000
    return total / n, total


def main():
    has_real = os.path.exists(TIF_PATH) and os.path.exists(ZIP_PATH)

    qapp = QApplication.instance() or QApplication(sys.argv)
    apply_dark_theme(qapp)

    print("=" * 72)
    print("CPU BENCHMARKS — Montaris-X")
    print("=" * 72)

    results = []

    # ── RLE encode/decode ─────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("1. RLE Encode / Decode Speed")
    print(f"{'─'*72}")
    print(f"  {'Density':<12} {'Encode ms':>12} {'Decode ms':>12} {'Crop ms':>12}")

    for label, density in [("empty", 0.0), ("sparse", 0.05),
                           ("dense", 0.5), ("full", 1.0)]:
        rng = np.random.RandomState(42)
        mask = (rng.random((6000, 12000)) < density).astype(np.uint8) * 255

        avg_enc, _ = bench("enc", lambda: rle_encode(mask), n=3)
        data, shape = rle_encode(mask)
        avg_dec, _ = bench("dec", lambda: rle_decode(data, shape), n=3)

        bbox = (1000, 2000, 2000, 4000)
        avg_crop, _ = bench("crop", lambda: rle_decode_crop(data, shape, bbox), n=5)

        print(f"  {label:<12} {avg_enc:>9.1f} ms {avg_dec:>9.1f} ms {avg_crop:>9.1f} ms")
        results.append((f"RLE {label} encode", f"{avg_enc:.1f} ms"))
        results.append((f"RLE {label} decode", f"{avg_dec:.1f} ms"))
        results.append((f"RLE {label} crop", f"{avg_crop:.1f} ms"))

    # ── Crop vs full decode comparison ────────────────────────────────
    print(f"\n{'─'*72}")
    print("2. rle_decode_crop vs Full Decode + Slice")
    print(f"{'─'*72}")

    mask = np.zeros((6000, 12000), dtype=np.uint8)
    mask[2000:3000, 4000:6000] = 255
    data, shape = rle_encode(mask)
    bbox = (2200, 2800, 4200, 5800)

    avg_full, _ = bench("full", lambda: rle_decode(data, shape)[bbox[0]:bbox[1], bbox[2]:bbox[3]], n=3)
    avg_crop, _ = bench("crop", lambda: rle_decode_crop(data, shape, bbox), n=10)
    speedup = avg_full / max(avg_crop, 0.001)

    print(f"  Full decode + slice: {avg_full:.1f} ms")
    print(f"  rle_decode_crop:     {avg_crop:.1f} ms")
    print(f"  Speedup:             {speedup:.1f}x")
    results.append(("Crop speedup", f"{speedup:.1f}x"))

    # ── Brush stroke latency ──────────────────────────────────────────
    if has_real:
        print(f"\n{'─'*72}")
        print("3. Brush Stroke Latency (real image)")
        print(f"{'─'*72}")

        from montaris.io.image_io import load_image
        from montaris.tools.brush import BrushTool

        app = MontarisApp()
        img_data = load_image(TIF_PATH)
        app.layer_stack.set_image(ImageLayer("test", img_data))
        app.import_roi_zip(ZIP_PATH)
        qapp.processEvents()

        target = None
        for roi in app.layer_stack.roi_layers:
            if roi.mask.any():
                target = roi
                break
        if target is None:
            target = app.layer_stack.roi_layers[0]

        canvas = FakeCanvas()

        # Single stroke
        def single_stroke():
            tool = BrushTool(app)
            tool.size = 50
            tool.on_press(QPointF(6000, 3000), target, canvas)
            tool.on_release(QPointF(6000, 3000), target, canvas)

        avg_single, _ = bench("single", single_stroke, n=5)
        print(f"  Single press+release: {avg_single:.1f} ms")

        # 10-move stroke
        def multi_stroke():
            tool = BrushTool(app)
            tool.size = 50
            tool.on_press(QPointF(6000, 3000), target, canvas)
            for i in range(10):
                tool.on_move(QPointF(6000 + i*5, 3000 + i*5), target, canvas)
            tool.on_release(QPointF(6050, 3050), target, canvas)

        avg_multi, _ = bench("multi", multi_stroke, n=3)
        print(f"  10-move stroke:       {avg_multi:.1f} ms")
        results.append(("Brush single", f"{avg_single:.1f} ms"))
        results.append(("Brush 10-move", f"{avg_multi:.1f} ms"))

        # ── Session save/restore ──────────────────────────────────────
        print(f"\n{'─'*72}")
        print("4. Session Save Speed")
        print(f"{'─'*72}")

        import shutil

        def save_session():
            app._session_dir = None
            app.save_session_progress()
            qapp.processEvents()

        avg_save, _ = bench("save", save_session, n=1)
        print(f"  Session save ({len(app.layer_stack.roi_layers)} ROIs): {avg_save:.0f} ms")
        results.append(("Session save", f"{avg_save:.0f} ms"))

        # Cleanup session dirs
        root_dir = os.path.dirname(TIF_PATH)
        for d in os.listdir(root_dir):
            if d.startswith("session_"):
                shutil.rmtree(os.path.join(root_dir, d), ignore_errors=True)

        # ── Import ZIP speed ──────────────────────────────────────────
        print(f"\n{'─'*72}")
        print("5. Import ROI ZIP Speed")
        print(f"{'─'*72}")

        from unittest.mock import patch

        def import_zip():
            app.layer_stack.roi_layers.clear()
            with patch.object(app, '_ask_replace_or_keep', return_value='keep'):
                app.import_roi_zip(ZIP_PATH)
            qapp.processEvents()

        avg_import, _ = bench("import", import_zip, n=1)
        print(f"  Import {ZIP_PATH}: {avg_import:.0f} ms")
        results.append(("Import ZIP", f"{avg_import:.0f} ms"))

        app.close()

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Operation':<30} {'Time':>15}")
    print(f"  {'─'*30} {'─'*15}")
    for op, t in results:
        print(f"  {op:<30} {t:>15}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
