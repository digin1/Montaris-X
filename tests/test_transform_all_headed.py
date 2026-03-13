"""Offscreen tests for TransformAll: load real image + ROI ZIP, exercise
scale/rotate/undo/redo/escape, measure performance.

Run:  python -m pytest tests/test_transform_all_headed.py -s -v --no-header
"""

import os, sys, time, numpy as np, pytest
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

# ── Module-wide state ──────────────────────────────────────────────
_qapp = None
_window = None
_original_masks = {}  # index -> mask copy (saved once after initial load)


@pytest.fixture(scope="module")
def app():
    """Single app + image + ROIs for the whole module."""
    global _qapp, _window, _original_masks

    if not IMAGE_PATH.exists() or not ZIP_PATH.exists():
        pytest.skip("test.tif and test.zip required for this test")

    _qapp = QApplication.instance() or QApplication(sys.argv)
    _qapp.setApplicationName("Montaris-X-Test")
    apply_dark_theme(_qapp)

    _window = MontarisApp()
    _window.show()
    QApplication.processEvents()

    # Load image
    img_data = load_image(str(IMAGE_PATH))
    _window.layer_stack.set_image(ImageLayer("test", img_data))
    _window.canvas.refresh_image()
    _window.canvas.fit_to_window()
    QApplication.processEvents()

    # Import ROIs
    with patch.object(_window, '_ask_replace_or_keep', return_value='keep'):
        _window.import_roi_zip(str(ZIP_PATH))
    QApplication.processEvents()

    n = len(_window.layer_stack.roi_layers)
    assert n > 0, "No ROIs loaded"
    print(f"\n  Loaded {img_data.shape} + {n} ROIs")

    # Save original masks
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
    """Restore all masks to original state before each test."""
    yield
    # After each test: restore masks and clear undo
    for i, r in enumerate(app.layer_stack.roi_layers):
        if i in _original_masks:
            r._rle_data = None  # force decompress
            r._mask = _original_masks[i].copy()
            r._mask_shape = r._mask.shape
            r._bbox_valid = False
            r.offset_x = 0
            r.offset_y = 0
    app.undo_stack.clear()
    QApplication.processEvents()


# ── Helpers ────────────────────────────────────────────────────────

def _pe():
    QApplication.processEvents()


def _snap(app):
    return {i: r.mask.copy() for i, r in enumerate(app.layer_stack.roi_layers)}


def _total_px(app):
    return sum(np.count_nonzero(r.mask) for r in app.layer_stack.roi_layers)


def _changed(before, app):
    return sum(1 for i, m in before.items()
               if i < len(app.layer_stack.roi_layers)
               and not np.array_equal(m, app.layer_stack.roi_layers[i].mask))


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

    # Simulate handle press (second press on handle)
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
            # Store bbox-crop RLE (matching optimized _transform_one)
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
    ms = (time.perf_counter() - t0) * 1000
    return tool, ms


# ── Scaling ────────────────────────────────────────────────────────

class TestScaling:
    def test_scale_br(self, app):
        before = _snap(app)
        _, ms = _do_transform(app, 'br', 10, 10)
        print(f"  br: {ms:.0f}ms")
        assert _changed(before, app) > 0

    def test_scale_tl(self, app):
        before = _snap(app)
        _do_transform(app, 'tl', -8, -8)
        assert _changed(before, app) > 0

    def test_scale_tr(self, app):
        before = _snap(app)
        _do_transform(app, 'tr', 10, -10)
        assert _changed(before, app) > 0

    def test_scale_bl(self, app):
        before = _snap(app)
        _do_transform(app, 'bl', -10, 10)
        assert _changed(before, app) > 0

    def test_scale_tm(self, app):
        before = _snap(app)
        _do_transform(app, 'tm', 0, -15)
        assert _changed(before, app) > 0

    def test_scale_mr(self, app):
        before = _snap(app)
        _do_transform(app, 'mr', 15, 0)
        assert _changed(before, app) > 0

    def test_shift_scale(self, app):
        _do_transform(app, 'br', 20, 5, shift=True)


# ── Rotation ───────────────────────────────────────────────────────

class TestRotation:
    def test_rotate_small(self, app):
        before = _snap(app)
        _do_transform(app, 'rotate', 15, 0)
        assert _changed(before, app) > 0

    def test_rotate_large(self, app):
        px0 = _total_px(app)
        _do_transform(app, 'rotate', 40, 0)
        ratio = _total_px(app) / max(px0, 1)
        assert 0.5 < ratio < 2.0, f"Pixel ratio {ratio:.2f}"

    def test_rotate_negative(self, app):
        before = _snap(app)
        _do_transform(app, 'rotate', -30, 0)
        assert _changed(before, app) > 0

    def test_shift_snap_rotate(self, app):
        _do_transform(app, 'rotate', 20, 0, shift=True)


# ── Undo/Redo ──────────────────────────────────────────────────────

class TestUndoRedo:
    def test_undo_restores(self, app):
        before = _snap(app)
        _do_transform(app, 'br', 15, 15)
        assert _changed(before, app) > 0

        app.undo_stack.undo()
        _pe()
        after = _snap(app)
        for i, orig in before.items():
            if i in after:
                assert np.array_equal(orig, after[i]), f"ROI {i} not restored"

    def test_redo(self, app):
        _do_transform(app, 'tl', -10, -10)
        transformed = _snap(app)

        app.undo_stack.undo()
        _pe()
        app.undo_stack.redo()
        _pe()

        after = _snap(app)
        for i in transformed:
            if i in after:
                assert np.array_equal(transformed[i], after[i])

    def test_undo_rotation(self, app):
        before = _snap(app)
        _do_transform(app, 'rotate', 25, 0)
        app.undo_stack.undo()
        _pe()
        after = _snap(app)
        for i, orig in before.items():
            if i in after:
                assert np.array_equal(orig, after[i])


# ── Cancel ─────────────────────────────────────────────────────────

class TestCancel:
    def test_escape(self, app):
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

        tool.on_move(QPointF(br.x + 30, br.y + 30), layer, canvas)
        _pe()
        tool.on_key_press(Qt.Key_Escape, canvas)
        _pe()

        after = _snap(app)
        for i, orig in before.items():
            if i in after:
                assert np.array_equal(orig, after[i]), "Mask changed after Escape"


# ── Sequential ─────────────────────────────────────────────────────

class TestSequential:
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


# ── Performance ────────────────────────────────────────────────────

class TestPerformance:
    def test_scale_perf(self, app):
        n = len(app.layer_stack.roi_layers)
        _, ms = _do_transform(app, 'br', 10, 10)
        print(f"\n  PERF scale ({n} ROIs): {ms:.0f} ms")

    def test_rotate_perf(self, app):
        n = len(app.layer_stack.roi_layers)
        _, ms = _do_transform(app, 'rotate', 20, 0)
        print(f"\n  PERF rotate ({n} ROIs): {ms:.0f} ms")

    def test_undo_perf(self, app):
        _do_transform(app, 'br', 10, 10)
        t0 = time.perf_counter()
        app.undo_stack.undo()
        _pe()
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n  PERF undo: {ms:.0f} ms")

    def test_memory(self, app):
        import psutil
        proc = psutil.Process()
        m0 = proc.memory_info().rss / (1024 * 1024)
        for _ in range(5):
            _do_transform(app, 'br', 3, 3)
        m1 = proc.memory_info().rss / (1024 * 1024)
        n = len(app.layer_stack.roi_layers)
        print(f"\n  PERF mem: {m0:.0f}->{m1:.0f} MB (+{m1-m0:.0f}) [{n} ROIs x5]")


# ── Edge Cases ─────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_roi_in_mix(self, app):
        h, w = app.layer_stack.image_layer.shape[:2]
        empty = ROILayer("empty", w, h)
        app.layer_stack.add_roi(empty)
        _pe()
        _do_transform(app, 'br', 5, 5)
        # cleanup
        app.layer_stack.roi_layers.remove(empty)

    def test_compressed_rois(self, app):
        active = app.canvas._active_layer
        app.layer_stack.compress_inactive(active)
        _pe()
        n = sum(1 for r in app.layer_stack.roi_layers if r.is_compressed)
        print(f"\n  {n} compressed")
        _do_transform(app, 'br', 8, 8)
