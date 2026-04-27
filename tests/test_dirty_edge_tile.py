"""Pytest coverage for the brush/eraser lag fix.

Targets the new tile-based partial refresh (``ImageCanvas._render_dirty_edge_tile``
and ``ImageCanvas.refresh_dirty_region``) that re-rasterizes the ROI overlay
on a small bbox rather than rebuilding the whole 71 M-pixel ROI on every
brush stroke. Specifically asserts:

- Tile path produces the *same* RGBA output as the full-ROI rebuild on a
  small synthetic mask (byte-equivalent fast path).
- Edge state of pixels just outside the dirty bbox (within ``thickness``)
  is updated too (M2 fringe-correctness from fresh-eyes review).
- ``refresh_dirty_region`` clears any pending throttled-flush entry so a
  later timer tick can't re-render the same region.
- The tile path is *not* taken on first render (no existing pixmap item
  yet) — the caller's full-refresh fallback handles that.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from montaris.app import apply_dark_theme
from montaris.canvas import ImageCanvas
from montaris.layers import LayerStack, ImageLayer, ROILayer


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        apply_dark_theme(app)
    yield app


@pytest.fixture
def canvas_with_roi(qapp):
    """ImageCanvas with one selected ROI rendered fully (existing pixmap
    item), ready for partial-refresh testing. Selection forces the edge
    tile path because the active boundary recolours on selection.
    """
    stack = LayerStack()
    img = np.zeros((200, 240), dtype=np.uint8)
    stack.set_image(ImageLayer("img", img))
    roi = ROILayer("roi1", 240, 200)
    # Make a 60×60 painted block so the ROI has a real bbox + edge.
    roi.mask[60:120, 80:140] = 255
    stack.add_roi(roi)

    canvas = ImageCanvas(stack)
    canvas.resize(800, 600)
    # Mark the ROI as selected so the partial-refresh path actually
    # routes through ``_render_dirty_edge_tile`` (the solid-mode fast
    # path is taken otherwise — the original test fixture missed this
    # and silently exercised a different code path).
    canvas._selection.set([roi])
    canvas._active_layer = roi
    # Force an initial full render so an existing pixmap item exists for
    # the partial path to update in place.
    canvas.refresh_image()
    canvas.refresh_active_overlay(roi)
    yield canvas, roi
    canvas.close()


def _pixmap_pixel_at_scene(canvas, roi, scene_y, scene_x):
    """Return RGBA at scene coordinate ``(scene_y, scene_x)`` for ``roi``'s
    pixmap, or ``None`` if outside the pixmap. Lets us compare two
    different-sized pixmaps at matching world positions — a raw
    byte-equality compare fails when the tile path's pixmap is
    legitimately larger (it inflates by ``thickness`` for boundary
    safety) than what a full rebuild produces from ``layer.get_bbox()``.
    """
    item = canvas._roi_items[id(roi)]
    pos_x, pos_y = int(item.pos().x()), int(item.pos().y())
    img = item.image()
    px = scene_x - pos_x
    py = scene_y - pos_y
    if px < 0 or py < 0 or px >= img.width() or py >= img.height():
        return None
    c = img.pixelColor(px, py)
    return (c.red(), c.green(), c.blue(), c.alpha())


def test_tile_path_matches_full_refresh(canvas_with_roi):
    """Stroke a small region into the ROI and confirm the tile-path
    output matches the full ``_refresh_roi_item`` rebuild at every
    scene-coordinate sample point inside the ROI bbox + a margin.

    Compares per-pixel at scene coordinates rather than raw bytes
    because the tile path's pixmap is legitimately larger than the
    full path's (it inflates by ``thickness`` for boundary safety) —
    raw byte equality would fail on the extra transparent rim alone.
    """
    canvas, roi = canvas_with_roi
    rid = id(roi)
    assert canvas._roi_items.get(rid) is not None

    # Paint a small extension to the existing block.
    roi.mask[100:115, 140:155] = 255
    dirty_bbox = (100, 115, 140, 155)

    canvas.refresh_dirty_region(roi, dirty_bbox)
    tile_size = canvas._roi_items[rid].image().size()
    tile_pos = canvas._roi_items[rid].pos()
    tile_pixels = {
        (y, x): _pixmap_pixel_at_scene(canvas, roi, y, x)
        for y in range(50, 130, 5)
        for x in range(70, 165, 5)
    }

    # Now do a full rebuild and compare at the same scene coordinates.
    roi.invalidate_bbox()
    canvas.refresh_active_overlay(roi)
    full_size = canvas._roi_items[rid].image().size()
    full_pos = canvas._roi_items[rid].pos()
    full_pixels = {
        (y, x): _pixmap_pixel_at_scene(canvas, roi, y, x)
        for y in range(50, 130, 5)
        for x in range(70, 165, 5)
    }

    diffs = [(yx, tile_pixels[yx], full_pixels[yx])
             for yx in tile_pixels
             if tile_pixels[yx] != full_pixels[yx]
             and (tile_pixels[yx] is not None
                  or full_pixels[yx] is not None)]
    assert not diffs, (
        f"tile vs full mismatch at {len(diffs)} scene positions; "
        f"first diff: {diffs[0]}\n"
        f"tile pixmap size={tile_size} pos={tile_pos}; "
        f"full size={full_size} pos={full_pos}"
    )


def test_tile_path_updates_edge_outside_dirty_bbox(canvas_with_roi):
    """Fringe correctness (fresh-eyes M2): a pixel just outside the
    dirty bbox whose edge state flipped because of a change inside the
    dirty bbox must be repainted.

    Specifically: the brush adds a voxel adjacent to an existing edge
    pixel; that edge pixel was previously marked as boundary, and after
    the stroke it is interior (boundary state flipped). The output
    tile must inflate by ``thickness`` so the now-interior pixel is
    repainted with the fill colour, not left as a stale yellow line.
    """
    canvas, roi = canvas_with_roi
    rid = id(roi)
    img_before = canvas._roi_items[rid].image().copy()

    # Extend the painted block downward by one row. This makes the
    # pixels at y=119 (formerly the bottom edge) become interior.
    roi.mask[120:121, 80:140] = 255
    dirty_bbox = (120, 121, 80, 140)
    canvas.refresh_dirty_region(roi, dirty_bbox)

    img_after = canvas._roi_items[rid].image().copy()
    pos = canvas._roi_items[rid].pos()

    # Inspect pixel at (90, 119) — was on the old edge, should now be
    # interior. Compute its RGBA in the after image.
    px_x = 90 - int(pos.x())
    px_y = 119 - int(pos.y())
    if 0 <= px_x < img_after.width() and 0 <= px_y < img_after.height():
        c_after = img_after.pixelColor(px_x, px_y)
        # Now-interior pixel must NOT be yellow (boundary). Yellow is
        # rgb≈(255,255,0); interior of an unselected ROI is the layer
        # colour with no yellow channel mix. Check by ensuring the blue
        # channel is not strictly 0 + red+green not at max — i.e. the
        # tile painted the pixel as fill, not as boundary.
        is_yellow = (c_after.red() > 200 and c_after.green() > 200
                     and c_after.blue() < 80)
        assert not is_yellow, (
            f"pixel at (90,119) is still yellow ({c_after.red()},"
            f"{c_after.green()},{c_after.blue()}) after the bottom edge "
            "moved — output region wasn't inflated by thickness"
        )


def test_refresh_dirty_region_drops_pending_flush(canvas_with_roi):
    """``refresh_dirty_region`` is synchronous; the throttled flush
    queue must be cleared so the dirty timer doesn't render the same
    region a second time on its next tick.
    """
    canvas, roi = canvas_with_roi
    canvas.refresh_active_overlay_partial(roi, (60, 70, 80, 90))
    assert id(roi) in canvas._pending_dirty

    canvas.refresh_dirty_region(roi, (60, 70, 80, 90))
    assert id(roi) not in canvas._pending_dirty


def test_tile_path_skips_first_render(qapp):
    """When no pixmap item exists yet, the tile path must decline so the
    caller falls back to a full ``_refresh_roi_item``. Otherwise the
    first paint stroke has nothing to composite onto.
    """
    stack = LayerStack()
    stack.set_image(ImageLayer("img", np.zeros((50, 50), dtype=np.uint8)))
    roi = ROILayer("blank", 50, 50)
    stack.add_roi(roi)
    canvas = ImageCanvas(stack)
    try:
        canvas.refresh_image()
        # No initial refresh_active_overlay → no pixmap item yet.
        assert canvas._roi_items.get(id(roi)) is None

        ok = canvas._render_dirty_edge_tile(
            roi, (0, 5, 0, 5), lod_level=0,
            fill_mode="solid", is_selected=True,
        )
        assert ok is False, "tile path must return False when there is " \
                            "no existing pixmap to composite onto"
    finally:
        canvas.close()


def test_tile_path_skips_lod_levels(canvas_with_roi):
    """LOD > 0 paths still need whole-mask max-pool semantics that don't
    compose tile-by-tile, so the tile path must decline at LOD > 0.
    """
    canvas, roi = canvas_with_roi
    ok = canvas._render_dirty_edge_tile(
        roi, (60, 80, 80, 100), lod_level=1,
        fill_mode="solid", is_selected=True,
    )
    assert ok is False


def test_solid_path_skips_scaled_item(qapp):
    """Codex HIGH (re-review): the scaled-item guard must apply to the
    solid (non-edge) dirty branch too, not only the edge tile path.
    Brush auto-overlap routes non-active layers through
    ``refresh_dirty_region`` — a recipient previously rendered at
    LOD > 0 has ``existing.scale() > 1`` and would otherwise have
    full-res pixels written into a downsampled image at the wrong
    coordinates.
    """
    stack = LayerStack()
    stack.set_image(ImageLayer("img", np.zeros((100, 100), dtype=np.uint8)))
    roi = ROILayer("roi", 100, 100)
    roi.mask[20:40, 20:40] = 255
    stack.add_roi(roi)

    canvas = ImageCanvas(stack)
    canvas.resize(400, 400)
    # Layer NOT selected — solid path, not edge path.
    canvas.refresh_image()
    canvas.refresh_active_overlay(roi)
    try:
        item = canvas._roi_items[id(roi)]
        # Simulate prior LOD > 0 render that scaled the item.
        item.setScale(2.0)
        before_pos = (int(item.pos().x()), int(item.pos().y()))

        roi.mask[50:60, 50:60] = 255
        canvas.refresh_dirty_region(roi, (50, 60, 50, 60))

        # After dirty refresh, the item should have been rebuilt at
        # scale=1 (full LOD-0 rebuild because the guard declined).
        assert abs(item.scale() - 1.0) < 1e-6, (
            f"item still at scale {item.scale()} — solid dirty path "
            "missed the scaled-item guard and composited at the wrong "
            "coordinates"
        )
    finally:
        canvas.close()


def test_tile_path_does_not_unhide_invisible_layer(canvas_with_roi):
    """Codex MEDIUM (re-review): ``_render_dirty_edge_tile`` must not
    re-show a layer the user has hidden via ``layer.visible = False``.
    The full-refresh path respects this; the tile path used to call
    ``existing.setVisible(True)`` unconditionally.
    """
    canvas, roi = canvas_with_roi
    item = canvas._roi_items[id(roi)]
    roi.visible = False
    item.setVisible(False)

    # Stroke into the ROI; tile path runs but must keep item invisible.
    roi.mask[100:110, 100:110] = 255
    canvas.refresh_dirty_region(roi, (100, 110, 100, 110))

    assert not item.isVisible(), (
        "tile path called setVisible(True) on a hidden layer — old code "
        "did this unconditionally and would resurrect hidden ROIs on "
        "auto-overlap strokes"
    )


def test_dirty_path_preserves_compressed_mask(qapp):
    """Codex MEDIUM (re-review): the dirty render path must not fully
    decompress a compressed RLE mask just to render a tile of it. Brush
    auto-overlap explicitly compresses recipients to bound memory, then
    immediately calls ``refresh_dirty_region`` on them — if the dirty
    path goes through ``layer.mask`` (which forces a full decompress),
    repeated overlap strokes leak whole 71 M-pixel masks into RAM.
    """
    stack = LayerStack()
    stack.set_image(ImageLayer("img", np.zeros((200, 200), dtype=np.uint8)))
    roi = ROILayer("roi", 200, 200)
    roi.mask[40:80, 40:80] = 255
    stack.add_roi(roi)

    canvas = ImageCanvas(stack)
    canvas.resize(400, 400)
    canvas.refresh_image()
    canvas.refresh_active_overlay(roi)
    try:
        # Compress the mask so the next refresh has to crop-decode.
        roi.compress()
        assert roi.is_compressed

        # Solid path — non-selected, non-boundary mode.
        canvas.refresh_dirty_region(roi, (50, 60, 50, 60))
        assert roi.is_compressed, (
            "solid dirty render decompressed the whole mask via "
            "layer.mask[...]; should have used get_mask_crop"
        )

        # Edge path — select + boundary thickness > 1.
        roi.compress()
        stack.boundary_thickness = 2
        canvas._selection.set([roi])
        canvas._active_layer = roi
        canvas.refresh_dirty_region(roi, (50, 60, 50, 60))
        assert roi.is_compressed, (
            "edge tile render decompressed the whole mask; should have "
            "used get_mask_crop in _render_dirty_edge_tile too"
        )
    finally:
        canvas.close()


def test_tile_path_skips_during_active_stroke(canvas_with_roi):
    """Drag stutter regression: while a brush/eraser stroke is in
    progress, the partial-flush path must NOT take the expensive edge
    tile branch — at low zoom each thick-edge recompute is tens of ms
    and stacking them every 16 ms (the dirty timer) blocked the event
    loop after ~5 mouse-move events. The active-stroke flag short-
    circuits the edge branch so mid-drag flushes are cheap solid
    fills; the proper thick edge appears once on release.
    """
    canvas, roi = canvas_with_roi
    # Selected → would normally take the edge tile path.
    assert roi in canvas._selection.layers

    # Mark a stroke as active and call _render_dirty_region directly.
    canvas._stroke_in_progress = True
    try:
        # Spy on the edge-tile method so we can assert it's NOT called.
        edge_calls = []
        real_edge = canvas._render_dirty_edge_tile

        def spy(*args, **kwargs):
            edge_calls.append((args, kwargs))
            return real_edge(*args, **kwargs)

        canvas._render_dirty_edge_tile = spy  # type: ignore[assignment]
        roi.mask[100:115, 100:115] = 255
        canvas._render_dirty_region(roi, (100, 115, 100, 115), lod_level=0)
        assert edge_calls == [], (
            "edge tile path ran during a stroke — drag stutter will "
            "return because each flush blocks ~50 ms at low zoom"
        )
    finally:
        canvas._stroke_in_progress = False
        canvas._render_dirty_edge_tile = real_edge  # type: ignore[assignment]


def test_tool_switch_mid_stroke_clears_flag(canvas_with_roi):
    """Codex review HIGH (repro A): switching tools mid-stroke must
    abort the in-progress stroke. Without this, the original tool's
    ``on_release`` never fires, so ``_stroke_in_progress`` stays True
    forever and every subsequent dirty-region refresh silently skips
    the boundary recompute even in outline/boundary modes.
    """
    from montaris.tools.brush import BrushTool
    from montaris.tools.hand import HandTool

    class _StubApp:
        def __init__(self):
            from montaris.core.undo import UndoStack
            self.undo_stack = UndoStack()
            self._auto_overlap = False
            self.layer_stack = canvas_with_roi[0].layer_stack

    canvas, roi = canvas_with_roi
    brush = BrushTool(_StubApp())
    canvas.set_tool(brush)
    from PySide6.QtCore import QPointF
    brush.on_press(QPointF(100, 100), roi, canvas)
    assert canvas._stroke_in_progress is True
    assert brush._painting is True

    # User switches to the hand tool mid-drag.
    canvas.set_tool(HandTool(_StubApp()))

    assert canvas._stroke_in_progress is False, (
        "tool switch mid-stroke left the canvas flag latched True"
    )
    assert brush._painting is False, (
        "tool switch left the previous tool's _painting latched True; "
        "a stray on_move from a different code path could keep painting"
    )


def test_release_with_layer_none_clears_flag(canvas_with_roi):
    """Codex review HIGH (repro B): release with ``layer=None`` (image
    closed mid-drag, layer deselected) must still clear the flag, not
    return early before the cleanup.
    """
    from montaris.tools.brush import BrushTool

    class _StubApp:
        def __init__(self):
            from montaris.core.undo import UndoStack
            self.undo_stack = UndoStack()
            self._auto_overlap = False
            self.layer_stack = canvas_with_roi[0].layer_stack

    canvas, roi = canvas_with_roi
    brush = BrushTool(_StubApp())
    from PySide6.QtCore import QPointF
    brush.on_press(QPointF(100, 100), roi, canvas)
    assert canvas._stroke_in_progress is True

    # User closes the image; the active layer becomes None before
    # the mouse-release event.
    brush.on_release(QPointF(100, 100), None, canvas)

    assert canvas._stroke_in_progress is False, (
        "release with layer=None left the flag latched True"
    )
    assert brush._painting is False


def test_release_runs_edge_after_stroke_ends(canvas_with_roi):
    """Counterpart to the previous test: once the stroke ends and
    ``refresh_dirty_region`` runs, the edge tile path MUST execute so
    the user sees the proper boundary on release.
    """
    canvas, roi = canvas_with_roi
    # Simulate a press without a release by setting the flag directly.
    canvas._stroke_in_progress = False  # release happened
    edge_calls = []
    real_edge = canvas._render_dirty_edge_tile

    def spy(*args, **kwargs):
        edge_calls.append(True)
        return real_edge(*args, **kwargs)

    canvas._render_dirty_edge_tile = spy  # type: ignore[assignment]
    try:
        roi.mask[100:115, 100:115] = 255
        canvas.refresh_dirty_region(roi, (100, 115, 100, 115))
        assert edge_calls, (
            "release path skipped the edge tile — boundary will look "
            "stale until the next full refresh"
        )
    finally:
        canvas._render_dirty_edge_tile = real_edge  # type: ignore[assignment]


def test_tile_path_skips_when_existing_item_is_scaled(canvas_with_roi):
    """A non-active ROI rendered at scale > 1 (LOD downsampled by an
    earlier `_refresh_roi_item`) must NOT be composited into as if it
    were full-res — the tile path declines so the caller does a proper
    full-LOD rebuild (Codex review HIGH #2).
    """
    canvas, roi = canvas_with_roi
    item = canvas._roi_items[id(roi)]
    item.setScale(2.0)  # simulate a previously-LOD-1 render

    ok = canvas._render_dirty_edge_tile(
        roi, (60, 80, 80, 100), lod_level=0,
        fill_mode="solid", is_selected=True,
    )
    assert ok is False, (
        "tile path must decline when the existing pixmap is scaled — "
        "compositing at full-res would produce a misaligned overlay"
    )


def test_tile_path_correct_with_thick_boundary_and_distant_structure(qapp):
    """Codex review HIGH #1: with ``boundary_thickness > 1`` and a
    pre-existing painted region outside the dirty bbox, the mask
    context crop must be at least ``thickness`` voxels wider than the
    output region — otherwise the thick-boundary at the inner edge of
    the output is computed without seeing the nearby structure and
    pixels get dropped or false-edged.

    Asserts the tile-path RGBA matches the full-refresh RGBA byte for
    byte, which would fail if the mask context pad were too small.
    """
    stack = LayerStack()
    stack.set_image(ImageLayer("img", np.zeros((80, 80), dtype=np.uint8)))
    roi = ROILayer("roi", 80, 80)
    # Two painted regions: one inside the dirty bbox, one ``thickness``
    # voxels away from the dirty bbox so the thick-edge at the dirty-
    # region boundary depends on it.
    roi.mask[20:30, 20:30] = 255  # primary block
    roi.mask[5:7, 5:7] = 255      # secondary block, distant
    stack.add_roi(roi)
    stack.boundary_thickness = 3  # forces the thick-edge dilation path

    canvas = ImageCanvas(stack)
    canvas.resize(400, 400)
    canvas._selection.set([roi])
    canvas._active_layer = roi
    canvas.refresh_image()
    canvas.refresh_active_overlay(roi)
    try:
        # Stroke just outside the secondary block, dirty bbox does NOT
        # cover the secondary block but its inflated output does.
        roi.mask[8:10, 8:10] = 255
        dirty = (8, 10, 8, 10)
        canvas.refresh_dirty_region(roi, dirty)
        tile_pixels = {
            (y, x): _pixmap_pixel_at_scene(canvas, roi, y, x)
            for y in range(0, 40)
            for x in range(0, 40)
        }

        roi.invalidate_bbox()
        canvas.refresh_active_overlay(roi)
        full_pixels = {
            (y, x): _pixmap_pixel_at_scene(canvas, roi, y, x)
            for y in range(0, 40)
            for x in range(0, 40)
        }

        diffs = [(yx, tile_pixels[yx], full_pixels[yx])
                 for yx in tile_pixels
                 if tile_pixels[yx] != full_pixels[yx]]
        assert not diffs, (
            f"thick-boundary tile mismatch at {len(diffs)} positions; "
            f"first 3: {diffs[:3]}. Mask-context pad is too small."
        )
    finally:
        canvas.close()


def test_tile_path_correct_with_thick_boundary_only_dirty_region(qapp):
    """Same setup as above but the dirty region is a disjoint block —
    confirms the inflated output region is large enough that the
    corners of the new painted square get their full ``thickness``
    boundary instead of a clipped one.
    """
    stack = LayerStack()
    stack.set_image(ImageLayer("img", np.zeros((60, 60), dtype=np.uint8)))
    roi = ROILayer("roi", 60, 60)
    roi.mask[20:30, 20:30] = 255  # pre-existing block so item exists
    stack.add_roi(roi)
    stack.boundary_thickness = 3

    canvas = ImageCanvas(stack)
    canvas.resize(400, 400)
    canvas._selection.set([roi])
    canvas._active_layer = roi
    canvas.refresh_image()
    canvas.refresh_active_overlay(roi)
    try:
        roi.mask[40:50, 40:50] = 255
        dirty = (40, 50, 40, 50)
        canvas.refresh_dirty_region(roi, dirty)
        tile_pixels = {
            (y, x): _pixmap_pixel_at_scene(canvas, roi, y, x)
            for y in range(15, 55)
            for x in range(15, 55)
        }

        roi.invalidate_bbox()
        canvas.refresh_active_overlay(roi)
        full_pixels = {
            (y, x): _pixmap_pixel_at_scene(canvas, roi, y, x)
            for y in range(15, 55)
            for x in range(15, 55)
        }

        diffs = [(yx, tile_pixels[yx], full_pixels[yx])
                 for yx in tile_pixels
                 if tile_pixels[yx] != full_pixels[yx]]
        assert not diffs, (
            f"disjoint-block thick-boundary mismatch at {len(diffs)} "
            f"positions; first 3: {diffs[:3]}."
        )
    finally:
        canvas.close()
