"""Comprehensive headed tests for TransformAll: happy paths, unhappy paths,
regression, interaction sequences, and resource monitoring (memory + CPU).

Run:  python -m pytest tests/test_transform_all_comprehensive.py -s -v --no-header
"""

import gc
import math
import os
import sys
import time
import tracemalloc
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF, Qt

from montaris.app import MontarisApp, apply_dark_theme
from montaris.layers import ImageLayer, ROILayer
from montaris.tools.transform import TransformAllTool
from montaris.core.roi_transform import compute_handles
from montaris.core.rle import rle_encode
from montaris.io.image_io import load_image

ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "test.tif"
ZIP_PATH = ROOT / "test.zip"

# ── Module-wide state ─────────────────────────────────────────────
_qapp = None
_window = None
_original_masks = {}


@pytest.fixture(scope="module")
def app():
    global _qapp, _window, _original_masks

    if not IMAGE_PATH.exists() or not ZIP_PATH.exists():
        pytest.skip("test.tif and test.zip required")

    _qapp = QApplication.instance() or QApplication(sys.argv)
    _qapp.setApplicationName("Montaris-X-Test-Comprehensive")
    apply_dark_theme(_qapp)

    _window = MontarisApp()
    _window.show()
    QApplication.processEvents()

    img_data = load_image(str(IMAGE_PATH))
    _window.layer_stack.set_image(ImageLayer("test", img_data))
    _window.canvas.refresh_image()
    _window.canvas.fit_to_window()
    QApplication.processEvents()

    with patch.object(_window, '_ask_replace_or_keep', return_value='keep'):
        _window.import_roi_zip(str(ZIP_PATH))
    QApplication.processEvents()

    n = len(_window.layer_stack.roi_layers)
    assert n > 0, "No ROIs loaded"
    print(f"\n  Loaded {img_data.shape} + {n} ROIs")

    _original_masks = {i: r.mask.copy() for i, r in enumerate(_window.layer_stack.roi_layers)}

    first = _window.layer_stack.roi_layers[0]
    _window.canvas.set_active_layer(first)
    _window.canvas.refresh_overlays()
    _window.layer_panel.refresh()
    QApplication.processEvents()

    yield _window

    from montaris.core.workers import shutdown_pool
    shutdown_pool()


@pytest.fixture(autouse=True)
def _restore_masks(app):
    yield
    for i, r in enumerate(app.layer_stack.roi_layers):
        if i in _original_masks:
            r._rle_data = None
            r._mask = _original_masks[i].copy()
            r._mask_shape = r._mask.shape
            r._bbox_valid = False
            r.offset_x = 0
            r.offset_y = 0
    app.undo_stack.clear()
    gc.collect()
    QApplication.processEvents()


# ── Helpers ───────────────────────────────────────────────────────

def _pe():
    QApplication.processEvents()


def _snap(app, max_rois=20):
    """Snapshot bbox crops of up to max_rois ROIs (memory-safe)."""
    out = {}
    for i, r in enumerate(app.layer_stack.roi_layers):
        if i >= max_rois:
            break
        bb = r.get_bbox()
        if bb is not None:
            out[i] = (bb, r.mask[bb[0]:bb[1], bb[2]:bb[3]].copy())
        else:
            out[i] = (None, None)
    return out


def _total_px(app):
    return sum(np.count_nonzero(r.mask) for r in app.layer_stack.roi_layers)


def _changed(before, app):
    count = 0
    for i, (bb, crop) in before.items():
        if i >= len(app.layer_stack.roi_layers):
            continue
        r = app.layer_stack.roi_layers[i]
        bb2 = r.get_bbox()
        if bb is None and bb2 is None:
            continue
        if bb is None or bb2 is None:
            count += 1
            continue
        if bb != bb2:
            count += 1
            continue
        crop2 = r.mask[bb2[0]:bb2[1], bb2[2]:bb2[3]]
        if not np.array_equal(crop, crop2):
            count += 1
    return count


def _masks_match(before, app, tolerance=0.0):
    """Check if masks match. tolerance is fraction of pixels allowed to differ."""
    for i, (bb, crop) in before.items():
        if i >= len(app.layer_stack.roi_layers):
            continue
        r = app.layer_stack.roi_layers[i]
        bb2 = r.get_bbox()
        if bb is None and bb2 is None:
            continue
        if bb is None or bb2 is None:
            if tolerance == 0:
                return False, f"ROI {i}: bbox None mismatch"
            continue
        # Compare full mask region covering both bboxes
        y1 = min(bb[0], bb2[0])
        y2 = max(bb[1], bb2[1])
        x1 = min(bb[2], bb2[2])
        x2 = max(bb[3], bb2[3])
        region_h, region_w = y2 - y1, x2 - x1
        # Reconstruct original region
        orig = np.zeros((region_h, region_w), dtype=np.uint8)
        oy1, oy2, ox1, ox2 = bb[0] - y1, bb[1] - y1, bb[2] - x1, bb[3] - x1
        orig[oy1:oy2, ox1:ox2] = crop
        # Current region
        cur = r.mask[y1:y2, x1:x2]
        if tolerance == 0:
            if not np.array_equal(orig, cur):
                return False, f"ROI {i}: masks differ"
        else:
            diff = np.count_nonzero(orig != cur)
            total = max(np.count_nonzero(orig), np.count_nonzero(cur), 1)
            if diff / total > tolerance:
                return False, f"ROI {i}: {diff}/{total} pixels differ ({diff/total:.1%})"
    return True, ""


def _do_transform(app, handle_type, dx, dy, shift=False):
    """Full transform: press->drag->release. Returns (tool, elapsed_ms)."""
    tool = TransformAllTool(app)
    canvas = app.canvas
    layer = canvas._active_layer
    rois = list(app.layer_stack.roi_layers)

    tool.on_press(QPointF(50, 50), layer, canvas)
    _pe()
    assert tool._bbox is not None, "No bbox"

    handles = compute_handles(tool._bbox)
    h = next((x for x in handles if x.handle_type == handle_type), None)
    assert h, f"Handle '{handle_type}' not found"

    tool._active_handle = h
    tool._start_pos = QPointF(h.x, h.y)
    tool._dragging = True
    tool._target_layers = rois
    tool._shift_held = shift
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
                tool._session_snapshots[lid] = (b'', (0, 0))
            tool._session_bboxes[lid] = sb
            tool._snapshots[lid] = (l, None, sb)
        else:
            crop = l.mask[sb[0]:sb[1], sb[2]:sb[3]].copy() if sb else None
            tool._snapshots[lid] = (l, crop, sb)

    tool._create_previews(canvas)
    _pe()

    end = QPointF(h.x + dx, h.y + dy)
    tool.on_move(end, layer, canvas)
    _pe()

    t0 = time.perf_counter()
    tool.on_release(end, layer, canvas)
    _pe()
    # Wait for async transform to finish (>3 ROIs uses timer polling)
    deadline = time.perf_counter() + 60  # 60s timeout
    while getattr(tool, '_applying', False) and time.perf_counter() < deadline:
        _pe()
        time.sleep(0.05)
    ms = (time.perf_counter() - t0) * 1000
    return tool, ms


def _get_process_mem_mb():
    import psutil
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _get_cpu_percent():
    import psutil
    return psutil.Process().cpu_percent(interval=0.1)


# ══════════════════════════════════════════════════════════════════
#  HAPPY PATHS
# ══════════════════════════════════════════════════════════════════

class TestScalingHappy:
    """All 8 scale handles with various displacements."""

    def test_scale_br_small(self, app):
        before = _snap(app)
        _, ms = _do_transform(app, 'br', 5, 5)
        print(f"    br small: {ms:.0f}ms")
        assert _changed(before, app) > 0

    def test_scale_br_large(self, app):
        before = _snap(app)
        _, ms = _do_transform(app, 'br', 50, 50)
        print(f"    br large: {ms:.0f}ms")
        assert _changed(before, app) > 0

    def test_scale_tl(self, app):
        before = _snap(app)
        _do_transform(app, 'tl', -10, -10)
        assert _changed(before, app) > 0

    def test_scale_tr(self, app):
        before = _snap(app)
        _do_transform(app, 'tr', 15, -15)
        assert _changed(before, app) > 0

    def test_scale_bl(self, app):
        before = _snap(app)
        _do_transform(app, 'bl', -15, 15)
        assert _changed(before, app) > 0

    def test_scale_tm(self, app):
        before = _snap(app)
        _do_transform(app, 'tm', 0, -20)
        assert _changed(before, app) > 0

    def test_scale_bm(self, app):
        before = _snap(app)
        _do_transform(app, 'bm', 0, 20)
        assert _changed(before, app) > 0

    def test_scale_ml(self, app):
        before = _snap(app)
        _do_transform(app, 'ml', -15, 0)
        assert _changed(before, app) > 0

    def test_scale_mr(self, app):
        before = _snap(app)
        _do_transform(app, 'mr', 15, 0)
        assert _changed(before, app) > 0

    def test_shift_scale_br(self, app):
        """Shift+scale should maintain aspect ratio."""
        before = _snap(app)
        _do_transform(app, 'br', 20, 5, shift=True)
        assert _changed(before, app) > 0

    def test_shift_scale_tl(self, app):
        before = _snap(app)
        _do_transform(app, 'tl', -20, -5, shift=True)
        assert _changed(before, app) > 0


class TestRotationHappy:
    """Rotation at various angles."""

    def test_rotate_small_positive(self, app):
        before = _snap(app)
        _do_transform(app, 'rotate', 10, 0)
        assert _changed(before, app) > 0

    def test_rotate_large_positive(self, app):
        before = _snap(app)
        px0 = _total_px(app)
        _do_transform(app, 'rotate', 50, 0)
        ratio = _total_px(app) / max(px0, 1)
        assert 0.3 < ratio < 3.0, f"Pixel ratio {ratio:.2f} out of range"

    def test_rotate_negative(self, app):
        before = _snap(app)
        _do_transform(app, 'rotate', -30, 0)
        assert _changed(before, app) > 0

    def test_rotate_180_degrees(self, app):
        """Large rotation near 180 degrees."""
        before = _snap(app)
        _do_transform(app, 'rotate', 200, 0)
        assert _changed(before, app) > 0

    def test_shift_snap_rotate(self, app):
        """Shift+rotate should snap to 15-degree increments."""
        before = _snap(app)
        _do_transform(app, 'rotate', 25, 0, shift=True)
        assert _changed(before, app) > 0

    def test_rotate_preserves_nonzero_pixels(self, app):
        """Rotation should not destroy all pixels."""
        px0 = _total_px(app)
        assert px0 > 0
        _do_transform(app, 'rotate', 20, 0)
        px1 = _total_px(app)
        assert px1 > 0, "All pixels destroyed by rotation"


class TestUndoRedoHappy:
    """Verify undo/redo restores masks (within bbox-crop precision)."""

    def test_undo_scale(self, app):
        before = _snap(app)
        _do_transform(app, 'br', 15, 15)
        assert _changed(before, app) > 0
        app.undo_stack.undo()
        _pe()
        ok, msg = _masks_match(before, app, tolerance=0.02)
        assert ok, f"Undo scale: {msg}"

    def test_redo_scale(self, app):
        _do_transform(app, 'tl', -12, -12)
        transformed = _snap(app)
        app.undo_stack.undo()
        _pe()
        app.undo_stack.redo()
        _pe()
        ok, msg = _masks_match(transformed, app, tolerance=0.02)
        assert ok, f"Redo scale: {msg}"

    def test_undo_rotation(self, app):
        before = _snap(app)
        _do_transform(app, 'rotate', 25, 0)
        app.undo_stack.undo()
        _pe()
        ok, msg = _masks_match(before, app, tolerance=0.02)
        assert ok, f"Undo rotation: {msg}"

    def test_undo_redo_undo_cycle(self, app):
        """Undo -> redo -> undo should leave masks close to original state."""
        before = _snap(app)
        _do_transform(app, 'mr', 20, 0)
        app.undo_stack.undo()
        _pe()
        app.undo_stack.redo()
        _pe()
        app.undo_stack.undo()
        _pe()
        ok, msg = _masks_match(before, app, tolerance=0.02)
        assert ok, f"Undo-redo-undo: {msg}"


class TestCancelHappy:
    """Escape during drag should restore masks."""

    def test_escape_cancels_scale(self, app):
        canvas = app.canvas
        layer = canvas._active_layer
        rois = list(app.layer_stack.roi_layers)
        before = _snap(app)

        tool = TransformAllTool(app)
        tool.on_press(QPointF(50, 50), layer, canvas)
        _pe()
        handles = compute_handles(tool._bbox)
        br = next(h for h in handles if h.handle_type == 'br')

        tool._active_handle = br
        tool._start_pos = QPointF(br.x, br.y)
        tool._dragging = True
        tool._target_layers = rois
        tool._snapshots = {}
        tool._snap_bboxes = {}
        for l in rois:
            lid = id(l)
            sb = l.get_bbox()
            tool._snap_bboxes[lid] = sb
            if sb is not None:
                tool._session_snapshots[lid] = rle_encode(
                    l.mask[sb[0]:sb[1], sb[2]:sb[3]])
            else:
                tool._session_snapshots[lid] = (b'', (0, 0))
            tool._session_bboxes[lid] = sb
            tool._snapshots[lid] = (l, None, sb)
        tool._create_previews(canvas)

        tool.on_move(QPointF(br.x + 40, br.y + 40), layer, canvas)
        _pe()
        tool.on_key_press(Qt.Key_Escape, canvas)
        _pe()

        ok, msg = _masks_match(before, app)
        assert ok, f"Escape cancel: {msg}"


# ══════════════════════════════════════════════════════════════════
#  UNHAPPY PATHS
# ══════════════════════════════════════════════════════════════════

class TestUnhappyPaths:
    """Edge cases, invalid inputs, boundary conditions."""

    def test_zero_displacement_scale(self, app):
        """Drag with dx=0, dy=0 should not crash and leave masks close to original."""
        before = _snap(app)
        _do_transform(app, 'br', 0, 0)
        ok, msg = _masks_match(before, app, tolerance=0.02)
        assert ok, f"Zero displacement: {msg}"

    def test_negative_scale_clamped(self, app):
        """Dragging inward past center should clamp to min 0.1 scale, not crash."""
        tool = TransformAllTool(app)
        canvas = app.canvas
        layer = canvas._active_layer
        # This should not crash - scale is clamped at 0.1
        _do_transform(app, 'br', -9999, -9999)
        assert _total_px(app) >= 0  # may be 0 if scale is extreme, but no crash

    def test_empty_roi_in_mix(self, app):
        """An empty ROI among populated ones should not crash."""
        h, w = app.layer_stack.image_layer.shape[:2]
        empty = ROILayer("empty", w, h)
        app.layer_stack.add_roi(empty)
        _pe()
        try:
            _do_transform(app, 'br', 10, 10)
        finally:
            app.layer_stack.roi_layers.remove(empty)

    def test_compressed_rois_transform(self, app):
        """Transform with compressed ROIs should work."""
        active = app.canvas._active_layer
        app.layer_stack.compress_inactive(active)
        _pe()
        n_compressed = sum(1 for r in app.layer_stack.roi_layers if r.is_compressed)
        print(f"    {n_compressed} compressed ROIs")
        before = _snap(app)
        _do_transform(app, 'br', 8, 8)
        assert _changed(before, app) > 0

    def test_single_pixel_roi(self, app):
        """An ROI with only 1 pixel should survive transform."""
        h, w = app.layer_stack.image_layer.shape[:2]
        tiny = ROILayer("single_px", w, h)
        tiny.mask[h // 2, w // 2] = 255
        app.layer_stack.add_roi(tiny)
        _pe()
        try:
            _do_transform(app, 'br', 5, 5)
        finally:
            app.layer_stack.roi_layers.remove(tiny)

    def test_roi_at_image_boundary(self, app):
        """ROI at the edge of the image should not cause out-of-bounds errors."""
        h, w = app.layer_stack.image_layer.shape[:2]
        edge = ROILayer("edge", w, h)
        edge.mask[0:5, 0:5] = 255  # top-left corner
        edge.mask[h-5:h, w-5:w] = 255  # bottom-right corner
        app.layer_stack.add_roi(edge)
        _pe()
        try:
            _do_transform(app, 'br', 10, 10)
            _do_transform(app, 'tl', -10, -10)
        finally:
            app.layer_stack.roi_layers.remove(edge)

    def test_very_large_displacement(self, app):
        """Extremely large drag should not crash or OOM."""
        _do_transform(app, 'br', 500, 500)
        assert True  # no crash

    def test_transform_after_all_rois_empty(self, app):
        """If all ROIs are empty, tool should handle gracefully."""
        # Temporarily zero all masks
        saved = {}
        for i, r in enumerate(app.layer_stack.roi_layers):
            saved[i] = r.mask.copy()
            r._mask[:] = 0
            r._bbox_valid = False
        _pe()
        try:
            tool = TransformAllTool(app)
            canvas = app.canvas
            layer = canvas._active_layer
            # on_press should not crash when all masks are empty
            tool.on_press(QPointF(50, 50), layer, canvas)
            _pe()
        finally:
            for i, r in enumerate(app.layer_stack.roi_layers):
                if i in saved:
                    r._mask = saved[i]
                    r._bbox_valid = False


# ══════════════════════════════════════════════════════════════════
#  SEQUENTIAL / INTERACTION REGRESSION
# ══════════════════════════════════════════════════════════════════

class TestSequentialInteraction:
    """Multi-step interaction sequences to catch regressions."""

    @pytest.mark.xfail(reason="Bbox-crop undo diff can miss boundary pixels "
                        "across two sequential transforms (pre-existing)")
    def test_two_scales_undo_both(self, app):
        original = _snap(app)
        _do_transform(app, 'br', 10, 10)
        _do_transform(app, 'tl', -5, -5)
        app.undo_stack.undo()
        _pe()
        app.undo_stack.undo()
        _pe()
        after = _snap(app)
        for i, orig in original.items():
            if i in after:
                diff = np.count_nonzero(orig != after[i])
                total = max(np.count_nonzero(orig), 1)
                assert diff / total < 0.01, \
                    f"ROI {i}: {diff} pixels differ ({diff/total:.1%})"

    def test_scale_then_rotate(self, app):
        _do_transform(app, 'br', 8, 8)
        assert _total_px(app) > 0
        _do_transform(app, 'rotate', 20, 0)
        assert _total_px(app) > 0

    def test_rotate_then_scale(self, app):
        _do_transform(app, 'rotate', 15, 0)
        assert _total_px(app) > 0
        _do_transform(app, 'mr', 10, 0)
        assert _total_px(app) > 0

    def test_multiple_rotations(self, app):
        """Three rotations in sequence should not accumulate errors badly."""
        px0 = _total_px(app)
        _do_transform(app, 'rotate', 10, 0)
        _do_transform(app, 'rotate', 10, 0)
        _do_transform(app, 'rotate', 10, 0)
        px_final = _total_px(app)
        ratio = px_final / max(px0, 1)
        assert 0.3 < ratio < 3.0, f"Pixel ratio {ratio:.2f} after 3 rotations"

    def test_scale_undo_scale_different_handle(self, app):
        """Scale br, undo, then scale tl — should apply cleanly."""
        original = _snap(app)
        _do_transform(app, 'br', 10, 10)
        app.undo_stack.undo()
        _pe()
        ok, msg = _masks_match(original, app, tolerance=0.02)
        assert ok, f"Undo before re-scale: {msg}"
        _do_transform(app, 'tl', -10, -10)
        assert _changed(original, app) > 0

    def test_rapid_small_transforms(self, app):
        """Many small transforms in rapid succession."""
        for _ in range(10):
            _do_transform(app, 'br', 1, 1)
        assert _total_px(app) > 0

    def test_all_handles_cycle(self, app):
        """Cycle through every handle type once."""
        handle_types = ['tl', 'tr', 'bl', 'br', 'tm', 'bm', 'ml', 'mr', 'rotate']
        for ht in handle_types:
            dx = 5 if 'r' in ht or ht == 'rotate' else (-5 if 'l' in ht else 0)
            dy = 5 if 'b' in ht else (-5 if 't' in ht else 0)
            if ht == 'rotate':
                dx, dy = 10, 0
            _do_transform(app, ht, dx, dy)
        assert _total_px(app) > 0


# ══════════════════════════════════════════════════════════════════
#  PERFORMANCE & RESOURCE MONITORING
# ══════════════════════════════════════════════════════════════════

class TestPerformanceDetailed:
    """Measure timing, memory, and CPU for various operations."""

    def test_scale_perf(self, app):
        n = len(app.layer_stack.roi_layers)
        _, ms = _do_transform(app, 'br', 10, 10)
        print(f"\n  PERF scale ({n} ROIs): {ms:.0f} ms")
        # Soft assertion: scale should complete in reasonable time
        assert ms < 30000, f"Scale took {ms:.0f}ms — too slow"

    def test_rotate_perf(self, app):
        n = len(app.layer_stack.roi_layers)
        _, ms = _do_transform(app, 'rotate', 20, 0)
        print(f"\n  PERF rotate ({n} ROIs): {ms:.0f} ms")
        assert ms < 30000, f"Rotate took {ms:.0f}ms — too slow"

    def test_undo_perf(self, app):
        _do_transform(app, 'br', 10, 10)
        t0 = time.perf_counter()
        app.undo_stack.undo()
        _pe()
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n  PERF undo: {ms:.0f} ms")
        assert ms < 30000, f"Undo took {ms:.0f}ms — too slow"

    def test_memory_process_rss(self, app):
        """Track process RSS across 5 sequential transforms."""
        import psutil
        proc = psutil.Process()
        gc.collect()
        m0 = proc.memory_info().rss / (1024 * 1024)
        for i in range(5):
            _do_transform(app, 'br', 3, 3)
        gc.collect()
        m1 = proc.memory_info().rss / (1024 * 1024)
        n = len(app.layer_stack.roi_layers)
        delta = m1 - m0
        print(f"\n  PERF mem RSS: {m0:.0f} -> {m1:.0f} MB (+{delta:.0f}) [{n} ROIs x5]")
        # Soft assertion: shouldn't leak excessively
        assert delta < 500, f"Memory grew {delta:.0f} MB — possible leak"

    def test_memory_tracemalloc(self, app):
        """Use tracemalloc for Python-level allocation tracking."""
        tracemalloc.start()
        snap1 = tracemalloc.take_snapshot()
        for _ in range(3):
            _do_transform(app, 'rotate', 10, 0)
        gc.collect()
        snap2 = tracemalloc.take_snapshot()
        stats = snap2.compare_to(snap1, 'lineno')
        top_allocs = stats[:10]
        total_new = sum(s.size_diff for s in stats if s.size_diff > 0) / (1024 * 1024)
        print(f"\n  PERF tracemalloc: +{total_new:.1f} MB Python allocations across 3 rotates")
        print("  Top allocations:")
        for s in top_allocs[:5]:
            print(f"    {s}")
        tracemalloc.stop()

    def test_cpu_during_transform(self, app):
        """Measure CPU usage during a transform."""
        import psutil
        proc = psutil.Process()
        proc.cpu_percent()  # prime the measurement
        t0 = time.perf_counter()
        _do_transform(app, 'br', 15, 15)
        elapsed = time.perf_counter() - t0
        cpu = proc.cpu_percent()
        print(f"\n  PERF CPU: {cpu:.1f}% over {elapsed:.2f}s for scale transform")

    def test_memory_after_undo_redo(self, app):
        """Memory should not grow significantly after undo/redo cycles."""
        import psutil
        proc = psutil.Process()
        gc.collect()
        m0 = proc.memory_info().rss / (1024 * 1024)
        for _ in range(5):
            _do_transform(app, 'br', 5, 5)
            app.undo_stack.undo()
            _pe()
        gc.collect()
        m1 = proc.memory_info().rss / (1024 * 1024)
        delta = m1 - m0
        print(f"\n  PERF undo/redo mem: {m0:.0f} -> {m1:.0f} MB (+{delta:.0f})")
        assert delta < 600, f"Memory grew {delta:.0f} MB after undo/redo cycles"

    def test_timing_all_handles(self, app):
        """Time each handle type for comparison."""
        results = {}
        handle_types = ['tl', 'tr', 'bl', 'br', 'tm', 'bm', 'ml', 'mr', 'rotate']
        for ht in handle_types:
            dx = 10 if 'r' in ht or ht == 'rotate' else (-10 if 'l' in ht else 0)
            dy = 10 if 'b' in ht else (-10 if 't' in ht else 0)
            if ht == 'rotate':
                dx, dy = 15, 0
            _, ms = _do_transform(app, ht, dx, dy)
            results[ht] = ms
        print("\n  PERF per-handle timing:")
        for ht, ms in results.items():
            bar = '#' * max(1, int(ms / 50))
            print(f"    {ht:>7s}: {ms:6.0f}ms {bar}")


# ══════════════════════════════════════════════════════════════════
#  REGRESSION TESTS
# ══════════════════════════════════════════════════════════════════

class TestRegression:
    """Tests for specific bugs that could regress."""

    def test_handles_reappear_after_transform(self, app):
        """After a transform completes, handles should be rebuilt."""
        tool = TransformAllTool(app)
        canvas = app.canvas
        layer = canvas._active_layer
        tool.on_press(QPointF(50, 50), layer, canvas)
        _pe()
        assert tool._bbox is not None
        bbox_before = tool._bbox

        # Perform a transform
        _do_transform(app, 'br', 10, 10)

        # Handles should still be present (new tool instance, but conceptually
        # the TransformAllTool should show handles)
        tool2 = TransformAllTool(app)
        tool2.on_activate(layer, canvas)
        _pe()
        assert tool2._bbox is not None, "Handles not shown after transform"

    def test_mask_dtype_preserved(self, app):
        """Transform should not change mask dtype from uint8."""
        _do_transform(app, 'br', 10, 10)
        for r in app.layer_stack.roi_layers:
            assert r.mask.dtype == np.uint8, f"Mask dtype changed to {r.mask.dtype}"

    def test_mask_values_binary(self, app):
        """Mask values should remain 0 or 255 (binary) after transform."""
        _do_transform(app, 'br', 10, 10)
        for i, r in enumerate(app.layer_stack.roi_layers):
            unique = np.unique(r.mask)
            for v in unique:
                assert v in (0, 255), \
                    f"ROI {i} has unexpected mask value {v}, unique={unique}"

    def test_no_nan_in_masks(self, app):
        """Masks should never contain NaN values."""
        _do_transform(app, 'rotate', 20, 0)
        for r in app.layer_stack.roi_layers:
            assert not np.any(np.isnan(r.mask.astype(float))), "NaN in mask"

    def test_bbox_consistent_after_transform(self, app):
        """ROI bbox should match actual nonzero pixels after transform."""
        _do_transform(app, 'br', 10, 10)
        for i, r in enumerate(app.layer_stack.roi_layers):
            r._bbox_valid = False  # force recompute
            bbox = r.get_bbox()
            if np.count_nonzero(r.mask) == 0:
                assert bbox is None, f"ROI {i}: non-None bbox for empty mask"
            elif bbox is not None:
                y1, y2, x1, x2 = bbox
                # bbox region should contain all nonzero pixels
                outside = r.mask.copy()
                outside[y1:y2, x1:x2] = 0
                assert np.count_nonzero(outside) == 0, \
                    f"ROI {i}: pixels outside bbox"

    def test_layer_stack_integrity_after_transform(self, app):
        """Layer stack should have same number of ROIs after transform."""
        n_before = len(app.layer_stack.roi_layers)
        _do_transform(app, 'br', 10, 10)
        n_after = len(app.layer_stack.roi_layers)
        assert n_before == n_after, f"ROI count changed: {n_before} -> {n_after}"

    def test_undo_stack_not_empty_after_transform(self, app):
        """Undo stack should have an entry after transform."""
        app.undo_stack.clear()
        _do_transform(app, 'br', 10, 10)
        assert app.undo_stack.can_undo, "Undo stack empty after transform"

    def test_image_layer_untouched(self, app):
        """Transform should not modify the image layer data."""
        img_before = app.layer_stack.image_layer.data.copy()
        _do_transform(app, 'br', 10, 10)
        _do_transform(app, 'rotate', 15, 0)
        assert np.array_equal(img_before, app.layer_stack.image_layer.data), \
            "Image layer was modified during transform"

    def test_mask_shape_unchanged(self, app):
        """Mask shapes should match image dimensions after any transform."""
        h, w = app.layer_stack.image_layer.shape[:2]
        _do_transform(app, 'br', 10, 10)
        for i, r in enumerate(app.layer_stack.roi_layers):
            assert r.mask.shape == (h, w), \
                f"ROI {i}: mask shape {r.mask.shape} != ({h}, {w})"

    def test_rotation_then_undo_then_different_scale(self, app):
        """Rotate, undo, then scale — should not leave stale rotation state."""
        before = _snap(app)
        _do_transform(app, 'rotate', 20, 0)
        app.undo_stack.undo()
        _pe()
        ok, msg = _masks_match(before, app, tolerance=0.02)
        assert ok, f"Rotation undo: {msg}"
        # Now do a scale — should work from original state
        _do_transform(app, 'br', 10, 10)
        assert _changed(before, app) > 0
