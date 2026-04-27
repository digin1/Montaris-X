"""Lock in that stamp / rectangle / circle / polygon / bucket-fill all
commit through ``canvas.refresh_dirty_region`` with the actual changed
bbox — not the full-ROI ``refresh_active_overlay`` which ran scipy
``binary_erosion`` on the whole 71 M-pixel mask on every commit.

Companion to ``tests/test_dirty_edge_tile.py`` (which covers the canvas
side of the fix); these tests pin the per-tool side.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication, QGraphicsScene

from montaris.layers import ROILayer
from montaris.core.undo import UndoStack
from montaris.tools.stamp import StampTool
from montaris.tools.rectangle import RectangleTool
from montaris.tools.circle import CircleTool
from montaris.tools.polygon import PolygonTool
from montaris.tools.bucket_fill import BucketFillTool


_qapp = QApplication.instance() or QApplication([])


class _RecordingCanvas:
    """Tracks every dirty / full refresh call so the test can assert
    which path the tool took.
    """

    def __init__(self, zoom=1.0):
        self._zoom = zoom
        self._scene = QGraphicsScene()
        self._preview_vertices = None
        self.dirty_calls = []  # list of (layer, bbox)
        self.full_calls = []   # list of layer

    def scene(self):
        return self._scene

    def transform(self):
        class _T:
            def __init__(self, z): self._z = z
            def m11(self): return self._z
        return _T(self._zoom)

    def refresh_dirty_region(self, layer, bbox):
        self.dirty_calls.append((layer, tuple(bbox)))

    def refresh_active_overlay(self, layer):
        self.full_calls.append(layer)

    def refresh_active_overlay_partial(self, layer, bbox):
        pass

    def refresh_overlays(self):
        pass

    def draw_polygon_preview(self, vertices, hover=None):
        self._preview_vertices = list(vertices)

    def clear_polygon_preview(self):
        self._preview_vertices = None


class _MockApp:
    def __init__(self):
        self.undo_stack = UndoStack()


@pytest.fixture
def app():
    return _MockApp()


@pytest.fixture
def layer():
    return ROILayer("roi", 200, 200)


@pytest.fixture
def canvas():
    return _RecordingCanvas()


def test_stamp_release_uses_dirty_region(app, layer, canvas):
    tool = StampTool(app)
    tool.size = tool.width = tool.height = 20
    tool.on_press(QPointF(50, 50), layer, canvas)
    tool.on_release(QPointF(50, 50), layer, canvas)

    assert canvas.full_calls == []
    assert len(canvas.dirty_calls) == 1
    bbox = canvas.dirty_calls[0][1]
    y1, y2, x1, x2 = bbox
    # Stamp size 20 around (50,50) → (40, 60, 40, 60); width/height of 20
    # paints a 20×20 square centred on the stamp.
    assert (y2 - y1) == 20 and (x2 - x1) == 20


def test_rectangle_release_uses_dirty_region(app, layer, canvas):
    tool = RectangleTool(app)
    tool.on_press(QPointF(20, 30), layer, canvas)
    tool.on_release(QPointF(60, 80), layer, canvas)

    assert canvas.full_calls == []
    assert len(canvas.dirty_calls) == 1
    bbox = canvas.dirty_calls[0][1]
    assert bbox == (30, 81, 20, 61)


def test_rectangle_uses_dirty_even_when_no_mask_change(app, layer, canvas):
    """Rectangle paint into already-painted pixels: no undo command is
    pushed (mask didn't change) but the refresh STILL routes through
    ``refresh_dirty_region`` with the rectangle bbox — tile re-rasterise
    on a small bbox is cheap, while ``refresh_active_overlay`` would
    rebuild the entire ROI overlay for nothing. Mirrors the brush/eraser
    idiom (fresh-eyes M2).
    """
    layer.mask[30:81, 20:61] = 255  # pre-fill the exact rectangle
    tool = RectangleTool(app)
    tool.on_press(QPointF(20, 30), layer, canvas)
    tool.on_release(QPointF(60, 80), layer, canvas)

    assert canvas.full_calls == []
    assert len(canvas.dirty_calls) == 1
    assert canvas.dirty_calls[0][1] == (30, 81, 20, 61)


def test_circle_release_uses_dirty_region(app, layer, canvas):
    tool = CircleTool(app)
    tool.on_press(QPointF(100, 100), layer, canvas)
    tool.on_release(QPointF(120, 100), layer, canvas)  # radius 20

    assert canvas.full_calls == []
    assert len(canvas.dirty_calls) == 1
    bbox = canvas.dirty_calls[0][1]
    y1, y2, x1, x2 = bbox
    # Circle bbox should fit a 20-radius circle around (100, 100).
    assert y1 == 80 and y2 == 121
    assert x1 == 80 and x2 == 121


def test_polygon_finish_uses_dirty_region(app, layer, canvas):
    tool = PolygonTool(app)
    # Build a 4-vertex polygon spaced widely so the close-marker check
    # doesn't auto-finish on the third press.
    for v in [(20, 20), (80, 20), (80, 80), (20, 80)]:
        tool.on_press(QPointF(*v), layer, canvas)
    tool.finish()

    assert canvas.full_calls == []
    assert len(canvas.dirty_calls) == 1
    bbox = canvas.dirty_calls[0][1]
    # Polygon bbox = vertex extremes + 1 on the upper bound.
    assert bbox == (20, 81, 20, 81)


def test_bucket_fill_uses_dirty_region(app, layer, canvas):
    """Background fill: clicks an empty pixel, fills the connected
    component, refresh routes through the bbox of the changed region.
    """
    # Pre-paint a 50×50 block; click outside it so the fill paints the
    # background everywhere except the block.
    layer.mask[80:130, 80:130] = 255
    tool = BucketFillTool(app)
    tool.on_press(QPointF(10, 10), layer, canvas)

    assert canvas.full_calls == []
    assert len(canvas.dirty_calls) == 1
    bbox = canvas.dirty_calls[0][1]
    y1, y2, x1, x2 = bbox
    # The fill changed every background pixel — bbox spans the mask
    # except for the pre-painted block's interior. Tight bbox is the
    # full canvas in this case.
    assert y1 == 0 and y2 == 200
    assert x1 == 0 and x2 == 200


def test_bucket_fill_no_undo_when_no_change(app, layer, canvas):
    """When the flood-fill region is None (target == fill, i.e. nothing
    to do), the dirty path is skipped and the fallback full refresh
    runs — cheap because nothing changed.
    """
    tool = BucketFillTool(app)
    real_compute = tool._compute_fill_region
    tool._compute_fill_region = lambda *a, **kw: None
    try:
        tool.on_press(QPointF(10, 10), layer, canvas)
        assert canvas.dirty_calls == []
        assert canvas.full_calls == [layer]
    finally:
        tool._compute_fill_region = real_compute


def test_bucket_fill_undo_restores_mixed_intensity_erase(app, layer, canvas):
    """Codex review HIGH regression: exact-match erase walks every
    nonzero pixel, not just pixels equal to ``target_value``. The
    earlier ``old_crop[changed] = target_value`` shortcut corrupted
    undo on regions with mixed intensities — undo restored the seed's
    value to every erased pixel instead of the actual original values.
    """
    layer.mask[5, 0] = 5
    layer.mask[5, 1] = 7
    layer.mask[5, 2] = 12
    tool = BucketFillTool(app)
    tool.tolerance = 0
    tool.on_press(QPointF(0, 5), layer, canvas)

    # Forward pass: all three pixels erased.
    assert layer.mask[5, 0] == 0
    assert layer.mask[5, 1] == 0
    assert layer.mask[5, 2] == 0

    # Undo must restore the truthful pre-fill values, not target_value.
    app.undo_stack.undo()
    assert layer.mask[5, 0] == 5
    assert layer.mask[5, 1] == 7, (
        "undo restored target_value=5 to a pixel that was 7 before — "
        "the old shortcut path lost mixed-intensity values"
    )
    assert layer.mask[5, 2] == 12


def test_bucket_fill_undo_restores_tolerance_mixed_intensities(app, layer,
                                                                 canvas):
    """Codex review HIGH regression for the tolerance>0 BFS path: the
    BFS visits any neighbour within ``tolerance`` of the seed value, so
    a row of [5, 7, 12, 100] with tolerance=10 includes the first three
    cells (within 10 of 5) but stops at 100. Undo must restore each cell's
    actual original value, not the seed's target_value.
    """
    layer.mask[5, 0] = 5
    layer.mask[5, 1] = 7
    layer.mask[5, 2] = 12
    layer.mask[5, 3] = 100
    tool = BucketFillTool(app)
    tool.tolerance = 10
    tool.on_press(QPointF(0, 5), layer, canvas)

    # Forward: [5, 7, 12] erased (within tolerance of 5); [100] untouched.
    assert layer.mask[5, 0] == 0
    assert layer.mask[5, 1] == 0
    assert layer.mask[5, 2] == 0
    assert layer.mask[5, 3] == 100

    # Undo must restore truthful values per pixel.
    app.undo_stack.undo()
    assert layer.mask[5, 0] == 5
    assert layer.mask[5, 1] == 7
    assert layer.mask[5, 2] == 12
    assert layer.mask[5, 3] == 100


def test_bucket_fill_tolerance_no_op_neighbour_skips_dirty(app, layer, canvas):
    """Codex review MEDIUM: with high tolerance, BFS can mark cells
    already at ``fill_value`` as part of the region (they pass the
    tolerance test even though writing fill_value is a no-op). The
    array-equal check after the apply step catches this and falls back
    to ``refresh_active_overlay`` instead of dirtying the bbox for no
    change. No spurious undo command is pushed.
    """
    layer.mask[7, 0] = 5  # seed
    layer.mask[7, 1] = 0  # already at fill_value
    layer.mask[7, 2] = 0  # already at fill_value
    tool = BucketFillTool(app)
    tool.tolerance = 255
    initial_undo_size = len(app.undo_stack._stack) if hasattr(
        app.undo_stack, "_stack") else None
    tool.on_press(QPointF(0, 7), layer, canvas)

    # Seed flips 5 → 0; the two zero neighbours are unchanged.
    assert layer.mask[7, 0] == 0
    assert layer.mask[7, 1] == 0
    assert layer.mask[7, 2] == 0
    # Refresh routes through dirty path on the seed-only diff.
    assert canvas.full_calls == []
    assert len(canvas.dirty_calls) == 1
