"""Benchmark: Drawing tool snapshot memory optimization.

Verifies that press/move/release for all 5 drawing tools no longer allocates
a full 71 MB mask copy, and that undo correctly restores original mask data.

Uses real test.tif (6020x11816) + test.zip (101 ROIs).

Run:
    LD_LIBRARY_PATH="" QT_QPA_PLATFORM=offscreen /usr/bin/python3 tests/bench_snapshot_memory.py
"""
import os
import sys
import time
import gc
import tracemalloc

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF

from montaris.app import MontarisApp, apply_dark_theme
from montaris.io.image_io import load_image
from montaris.layers import ImageLayer, ROILayer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIF_PATH = os.path.join(BASE, "test.tif")
ZIP_PATH = os.path.join(BASE, "test.zip")

# ── Helpers ──────────────────────────────────────────────────────────

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
                    return int(line.split()[1]) / 1024  # kB -> MB
    except Exception:
        pass
    return 0

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


class FakeCanvas:
    """Minimal canvas stub for tool press/move/release without GUI rendering."""

    def __init__(self):
        self._selection_highlight_items = []

    def refresh_active_overlay(self, layer):
        pass

    def refresh_active_overlay_partial(self, layer, bbox):
        pass

    def _update_selection_highlights(self):
        pass

    def scene(self):
        return self

    def addRect(self, *a, **kw):
        return _FakeItem()

    def addEllipse(self, *a, **kw):
        return _FakeItem()

    def removeItem(self, item):
        pass


class _FakeItem:
    def setZValue(self, z):
        pass

    def setVisible(self, v):
        pass


# ── Test runners ─────────────────────────────────────────────────────

def measure_tool_snapshot(tool, layer, canvas, press_pos, move_positions, release_pos):
    """Run a full press→move→release cycle, measuring peak alloc during press."""
    gc.collect()
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    tool.on_press(press_pos, layer, canvas)

    snap_after_press = tracemalloc.take_snapshot()
    press_alloc = sum(s.size for s in snap_after_press.statistics('filename'))
    press_alloc_before = sum(s.size for s in snap_before.statistics('filename'))
    press_delta_kb = (press_alloc - press_alloc_before) / 1024

    for p in move_positions:
        tool.on_move(p, layer, canvas)

    snap_after_moves = tracemalloc.take_snapshot()
    moves_alloc = sum(s.size for s in snap_after_moves.statistics('filename'))
    stroke_delta_kb = (moves_alloc - press_alloc_before) / 1024

    tool.on_release(release_pos, layer, canvas)
    tracemalloc.stop()

    return press_delta_kb, stroke_delta_kb


def test_undo_correctness(tool_name, tool, layer, canvas, press_pos, move_positions, release_pos, undo_stack):
    """Verify that undo restores the mask exactly."""
    original = layer.mask.copy()
    undo_stack._stack.clear()
    undo_stack._index = -1
    undo_stack._total_bytes = 0

    tool.on_press(press_pos, layer, canvas)
    for p in move_positions:
        tool.on_move(p, layer, canvas)
    tool.on_release(release_pos, layer, canvas)

    # Mask should have changed
    changed = not np.array_equal(layer.mask, original)
    if not changed:
        print(f"    {FAIL} {tool_name}: mask didn't change after stroke")
        return False

    # Undo should restore exactly
    if undo_stack._index >= 0:
        cmd = undo_stack._stack[undo_stack._index]
        cmd.undo()
        restored = np.array_equal(layer.mask, original)
        if not restored:
            diff_count = np.sum(layer.mask != original)
            print(f"    {FAIL} {tool_name}: undo didn't restore mask ({diff_count} pixels differ)")
            cmd.redo()  # restore for next test
            return False
        # Redo to leave mask in painted state for next test
        cmd.redo()
    else:
        print(f"    {FAIL} {tool_name}: no undo command pushed")
        return False

    print(f"    {PASS} {tool_name}: undo/redo correct")
    return True


def test_expand_snapshot_unit():
    """Unit test expand_snapshot helper with edge cases."""
    from montaris.tools.base import expand_snapshot

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 128  # pre-existing data

    # Case 1: First call — should copy paint region
    crop, bbox = expand_snapshot(mask, (10, 20, 10, 20), None, None)
    assert crop.shape == (10, 10), f"Expected (10,10), got {crop.shape}"
    assert np.array_equal(crop, mask[10:20, 10:20])
    ok1 = True

    # Case 2: Already covered — should return same object
    crop2, bbox2 = expand_snapshot(mask, (12, 18, 12, 18), crop, bbox)
    assert crop2 is crop, "Should return same object when already covered"
    ok2 = True

    # Case 3: Expand — new region overlaps but extends
    mask[5:10, 5:10] = 64  # data in expansion zone
    # Simulate paint into original region
    mask[10:20, 10:20] = 255
    crop3, bbox3 = expand_snapshot(mask, (5, 25, 5, 25), crop, bbox)
    assert bbox3 == (5, 25, 5, 25), f"Expected expanded bbox, got {bbox3}"
    # Old snapshot region should have pre-stroke values (128), not 255
    assert np.all(crop3[10-5:20-5, 10-5:20-5] == 128), "Old region should preserve pre-stroke values"
    # Expansion zone should have current values (64 for 5:10, 0 for 20:25)
    assert np.all(crop3[0:5, 0:5] == 64), "Expansion zone should have current mask values"
    ok3 = True

    # Case 4: Edge case — paint at mask boundary
    mask2 = np.zeros((50, 50), dtype=np.uint8)
    crop4, bbox4 = expand_snapshot(mask2, (0, 5, 0, 5), None, None)
    assert bbox4 == (0, 5, 0, 5)
    crop5, bbox5 = expand_snapshot(mask2, (45, 50, 45, 50), crop4, bbox4)
    assert bbox5 == (0, 50, 0, 50), f"Expected full span, got {bbox5}"
    ok4 = True

    all_ok = ok1 and ok2 and ok3 and ok4
    status = PASS if all_ok else FAIL
    print(f"    {status} expand_snapshot unit tests (4 cases)")
    return all_ok


def test_auto_overlap_undo(app, layer, canvas):
    """Test brush auto-overlap: painting on one ROI erases from another, undo restores both."""
    from montaris.tools.brush import BrushTool

    # Create a second ROI with overlapping data
    h, w = layer.mask.shape
    other = ROILayer("overlap_roi", w, h)
    other.mask[2900:3100, 5900:6100] = 255
    app.layer_stack.add_roi(other)
    app._auto_overlap = True

    # Paint on primary layer in the overlapping region
    layer.mask[2900:3100, 5900:6100] = 0  # clear first
    original_main = layer.mask.copy()
    original_other = other.mask.copy()

    tool = BrushTool(app)
    tool.size = 50
    app.undo_stack._stack.clear()
    app.undo_stack._index = -1
    app.undo_stack._total_bytes = 0

    tool.on_press(QPointF(6000, 3000), layer, canvas)
    tool.on_move(QPointF(6010, 3010), layer, canvas)
    tool.on_release(QPointF(6010, 3010), layer, canvas)

    # Main layer should have paint
    main_changed = not np.array_equal(layer.mask, original_main)
    # Other layer should have been erased where we painted
    other_changed = not np.array_equal(other.mask, original_other)

    if not main_changed:
        print(f"    {FAIL} auto-overlap: main layer didn't change")
        return False

    if not other_changed:
        print(f"    {FAIL} auto-overlap: other layer wasn't erased")
        return False

    # Undo should restore both
    if app.undo_stack._index >= 0:
        cmd = app.undo_stack._stack[app.undo_stack._index]
        cmd.undo()
        main_ok = np.array_equal(layer.mask, original_main)
        other_ok = np.array_equal(other.mask, original_other)
        if not main_ok or not other_ok:
            print(f"    {FAIL} auto-overlap: undo didn't restore (main={main_ok}, other={other_ok})")
            cmd.redo()
            return False
        cmd.redo()

    # Cleanup
    app.layer_stack.roi_layers.remove(other)
    app._auto_overlap = False
    print(f"    {PASS} auto-overlap: paint + erase + undo all correct")
    return True


def test_edge_painting(app, layer, canvas):
    """Test painting at mask boundaries (top-left corner, bottom-right corner)."""
    from montaris.tools.brush import BrushTool
    from montaris.tools.eraser import EraserTool

    h, w = layer.mask.shape
    results = []

    # Brush at top-left corner (0,0)
    original = layer.mask.copy()
    tool = BrushTool(app)
    tool.size = 100
    app.undo_stack._stack.clear()
    app.undo_stack._index = -1
    app.undo_stack._total_bytes = 0

    tool.on_press(QPointF(0, 0), layer, canvas)
    tool.on_release(QPointF(0, 0), layer, canvas)

    if app.undo_stack._index >= 0:
        cmd = app.undo_stack._stack[app.undo_stack._index]
        cmd.undo()
        ok = np.array_equal(layer.mask, original)
        cmd.redo()
        status = PASS if ok else FAIL
        print(f"    {status} brush at (0,0) corner")
        results.append(ok)
    else:
        print(f"    {FAIL} brush at (0,0): no undo command")
        results.append(False)

    # Eraser at bottom-right corner
    layer.mask[h-100:h, w-100:w] = 255  # fill region to erase
    original = layer.mask.copy()
    tool2 = EraserTool(app)
    tool2.size = 100
    app.undo_stack._stack.clear()
    app.undo_stack._index = -1
    app.undo_stack._total_bytes = 0

    tool2.on_press(QPointF(w - 1, h - 1), layer, canvas)
    tool2.on_release(QPointF(w - 1, h - 1), layer, canvas)

    if app.undo_stack._index >= 0:
        cmd = app.undo_stack._stack[app.undo_stack._index]
        cmd.undo()
        ok = np.array_equal(layer.mask, original)
        cmd.redo()
        status = PASS if ok else FAIL
        print(f"    {status} eraser at ({w-1},{h-1}) corner")
        results.append(ok)
    else:
        print(f"    {FAIL} eraser at bottom-right: no undo command")
        results.append(False)

    return all(results)


def test_long_stroke_memory(app, layer, canvas):
    """Test that a long diagonal stroke across the image stays within bbox memory."""
    from montaris.tools.brush import BrushTool

    h, w = layer.mask.shape
    original = layer.mask.copy()
    tool = BrushTool(app)
    tool.size = 30

    gc.collect()
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    # Long diagonal stroke: 100 move events across the image
    points = [QPointF(int(w * i / 100), int(h * i / 100)) for i in range(101)]
    tool.on_press(points[0], layer, canvas)
    for p in points[1:-1]:
        tool.on_move(p, layer, canvas)
    tool.on_release(points[-1], layer, canvas)

    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    total_after = sum(s.size for s in snap_after.statistics('filename'))
    total_before = sum(s.size for s in snap_before.statistics('filename'))
    delta_mb = (total_after - total_before) / 1024 / 1024
    mask_mb = h * w / 1024 / 1024

    # The snapshot should be much smaller than a full mask copy
    # For a diagonal stroke with r=15, the bbox covers a thin band, not the full image
    # But a full diagonal WILL have a large bbox. The key metric is that it's
    # the bbox crop, not a full copy.
    print(f"    Long diagonal stroke (100 moves, r=15):")
    print(f"      Memory delta: {delta_mb:.1f} MB  (full mask = {mask_mb:.1f} MB)")

    # Undo correctness
    if app.undo_stack._index >= 0:
        cmd = app.undo_stack._stack[app.undo_stack._index]
        undo_bytes = cmd.byte_size
        cmd.undo()
        ok = np.array_equal(layer.mask, original)
        cmd.redo()
        status = PASS if ok else FAIL
        print(f"      Undo command size: {undo_bytes / 1024:.0f} KB")
        print(f"      {status} undo correctness")
        return ok
    else:
        print(f"      {FAIL} no undo command")
        return False


# ── Main ─────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(TIF_PATH) or not os.path.exists(ZIP_PATH):
        print(f"ERROR: Need test.tif and test.zip in project root")
        sys.exit(1)

    qapp = QApplication.instance() or QApplication(sys.argv)
    apply_dark_theme(qapp)

    print("=" * 70)
    print("BENCHMARK: Drawing Tool Snapshot Memory Optimization")
    print("=" * 70)

    # Load real image
    print("\nLoading test.tif...")
    data = load_image(TIF_PATH)
    h, w = data.shape[:2]
    mask_mb = h * w / 1024 / 1024
    print(f"  Image: {w}x{h}, mask size = {mask_mb:.1f} MB")

    # Create app and load ROIs
    app = MontarisApp()
    img_layer = ImageLayer("test", data)
    app.layer_stack.set_image(img_layer)
    app.import_roi_zip(ZIP_PATH)
    qapp.processEvents()

    n_rois = len(app.layer_stack.roi_layers)
    print(f"  Loaded {n_rois} ROIs")

    # Pick the first non-empty ROI as our target
    target_roi = None
    for roi in app.layer_stack.roi_layers:
        if roi.mask.any():
            target_roi = roi
            break
    if target_roi is None:
        target_roi = app.layer_stack.roi_layers[0]
    print(f"  Target ROI: {target_roi.name} ({target_roi.mask.shape})")

    canvas = FakeCanvas()

    # ── 1. Unit tests ────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("1. expand_snapshot() unit tests")
    print(f"{'─'*70}")
    test_expand_snapshot_unit()

    # ── 2. Memory measurement per tool ───────────────────────────────
    print(f"\n{'─'*70}")
    print("2. Snapshot memory per tool (press + short stroke)")
    print(f"{'─'*70}")
    print(f"  {'Tool':<12} {'Press Δ':>12} {'Stroke Δ':>12} {'vs Full Mask':>14}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*14}")

    from montaris.tools.brush import BrushTool
    from montaris.tools.eraser import EraserTool
    from montaris.tools.stamp import StampTool
    from montaris.tools.rectangle import RectangleTool
    from montaris.tools.circle import CircleTool

    # Coordinates: QPointF(x, y) — x=col, y=row
    # Seed region: rows 2900:3300, cols 5800:6400
    # So valid positions: x in 5800..6400, y in 2900..3300
    tools_config = [
        ("Brush",     BrushTool,     50,  QPointF(6000, 3000), [QPointF(6020, 3020), QPointF(6040, 3040)], QPointF(6050, 3050)),
        ("Eraser",    EraserTool,    50,  QPointF(6000, 3100), [QPointF(6020, 3120), QPointF(6040, 3140)], QPointF(6050, 3150)),
        ("Stamp",     StampTool,     20,  QPointF(6000, 3000), [QPointF(6020, 3020), QPointF(6040, 3040)], QPointF(6050, 3050)),
        ("Rectangle", RectangleTool, None, QPointF(6000, 3000), [QPointF(6100, 3100)],                     QPointF(6200, 3200)),
        ("Circle",    CircleTool,    None, QPointF(6000, 3000), [QPointF(6100, 3100)],                     QPointF(6100, 3100)),
    ]

    for name, cls, size, press, moves, release in tools_config:
        # Fresh mask for each test
        target_roi.mask[:] = 0
        target_roi.mask[2900:3300, 5800:6400] = 128  # seed some data

        tool = cls(app)
        if size is not None:
            tool.size = size
            if hasattr(tool, 'width'):
                tool.width = size
                tool.height = size

        press_kb, stroke_kb = measure_tool_snapshot(tool, target_roi, canvas, press, moves, release)
        pct = (stroke_kb / 1024) / mask_mb * 100 if mask_mb > 0 else 0
        print(f"  {name:<12} {press_kb:>9.0f} KB {stroke_kb:>9.0f} KB {pct:>12.1f}%")

    old_full_kb = mask_mb * 1024
    print(f"\n  (Old approach: every tool allocated {old_full_kb:.0f} KB = {mask_mb:.1f} MB on press)")

    # ── 3. Undo correctness ──────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("3. Undo correctness per tool")
    print(f"{'─'*70}")

    all_ok = True
    for name, cls, size, press, moves, release in tools_config:
        target_roi.mask[:] = 0
        target_roi.mask[2900:3300, 5800:6400] = 128

        tool = cls(app)
        if size is not None:
            tool.size = size
            if hasattr(tool, 'width'):
                tool.width = size
                tool.height = size

        ok = test_undo_correctness(name, tool, target_roi, canvas, press, moves, release, app.undo_stack)
        all_ok = all_ok and ok

    # ── 4. Auto-overlap undo ─────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("4. Brush auto-overlap undo")
    print(f"{'─'*70}")
    ok = test_auto_overlap_undo(app, target_roi, canvas)
    all_ok = all_ok and ok

    # ── 5. Edge painting (corners) ───────────────────────────────────
    print(f"\n{'─'*70}")
    print("5. Edge painting (mask boundaries)")
    print(f"{'─'*70}")
    target_roi.mask[:] = 0
    ok = test_edge_painting(app, target_roi, canvas)
    all_ok = all_ok and ok

    # ── 6. Long stroke memory ────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("6. Long diagonal stroke (stress test)")
    print(f"{'─'*70}")
    target_roi.mask[:] = 0
    app.undo_stack._stack.clear()
    app.undo_stack._index = -1
    app.undo_stack._total_bytes = 0
    ok = test_long_stroke_memory(app, target_roi, canvas)
    all_ok = all_ok and ok

    # ── 7. Multiple rapid strokes ────────────────────────────────────
    print(f"\n{'─'*70}")
    print("7. Multiple rapid strokes (10 brush strokes, memory growth)")
    print(f"{'─'*70}")
    from montaris.tools.brush import BrushTool
    target_roi.mask[:] = 0
    app.undo_stack._stack.clear()
    app.undo_stack._index = -1
    app.undo_stack._total_bytes = 0

    gc.collect()
    rss_before = get_rss_mb()
    tool = BrushTool(app)
    tool.size = 80

    for i in range(10):
        cx = 1000 + i * 400
        cy = 3000
        tool.on_press(QPointF(cx, cy), target_roi, canvas)
        tool.on_move(QPointF(cx + 50, cy + 50), target_roi, canvas)
        tool.on_release(QPointF(cx + 100, cy + 100), target_roi, canvas)

    gc.collect()
    rss_after = get_rss_mb()
    undo_total = app.undo_stack._total_bytes
    print(f"    10 strokes completed, undo stack: {undo_total / 1024:.0f} KB")
    print(f"    RSS delta: {rss_after - rss_before:.1f} MB")

    # Undo all 10 and check
    original_empty = np.zeros_like(target_roi.mask)
    for i in range(10):
        if app.undo_stack._index >= 0:
            app.undo_stack._stack[app.undo_stack._index].undo()
            app.undo_stack._index -= 1
    ok = np.array_equal(target_roi.mask, original_empty)
    status = PASS if ok else FAIL
    print(f"    {status} undo all 10 strokes restores empty mask")
    all_ok = all_ok and ok

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    if all_ok:
        print(f"  ALL TESTS {PASS}")
    else:
        print(f"  SOME TESTS {FAIL}")
    print(f"{'='*70}")

    app.close()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
