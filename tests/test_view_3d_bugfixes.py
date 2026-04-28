"""Regression tests for 3D viewer + z-slider bug fixes.

Covers three bugs surfaced in the 2026-04-18 code review:

- ``channels_from_documents`` duplicated a shared z-stack across N docs
  (synced-mode channels sharing one volume), making the 3D viewer render
  tiled copies.
- ``_on_drag_refresh_tick`` uploaded the sub-region with xyz offset where
  vispy's Texture3D expects zyx.
- The bottom Z-slider only nudged ROI bboxes; it never swapped
  ``image_layer.data`` to ``volume_data[z]``, so the 2D image was frozen
  outside MIP mode.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from montaris.layers import (
    ImageLayer,
    MontageDocument,
    ROILayer,
    VolumeROILayer,
)
from montaris.widgets.view_3d import (
    VISPY_AVAILABLE,
    View3DPanel,
    _coerce_gl_int,
    _max_pool_labels,
    channels_from_documents,
)


def _doc_with_volume(name, vol):
    """Build a MontageDocument whose 2D image is the volume's MIP."""
    mp = vol.max(axis=0)
    doc = MontageDocument(name=name, image_layer=ImageLayer(name, mp))
    doc.volume_data = vol
    doc.volume_axes = "ZYX"
    return doc


def test_channels_from_documents_dedupes_shared_volume():
    """N docs sharing one volume (synced mode) should dedupe to one channel."""
    vol = np.arange(2 * 3 * 4, dtype=np.uint8).reshape((2, 3, 4))
    docs = [_doc_with_volume(f"stack_z{i}", vol) for i in range(vol.shape[0])]
    # Each doc's 2D image is a distinct slice but all share the same `vol`.
    assert all(d.volume_data is vol for d in docs)
    channels = channels_from_documents(docs)
    assert len(channels) == 1
    assert channels[0][0] == "stack_z0"
    assert channels[0][1] is vol


def test_channels_from_documents_keeps_distinct_volumes():
    """Two independently-loaded stacks still produce two channels."""
    vol_a = np.zeros((2, 3, 4), dtype=np.uint8)
    vol_b = np.ones((2, 3, 4), dtype=np.uint8)
    docs = [_doc_with_volume("A", vol_a), _doc_with_volume("B", vol_b)]
    channels = channels_from_documents(docs)
    assert len(channels) == 2
    assert {c[0] for c in channels} == {"A", "B"}


def test_coerce_gl_int_accepts_plain_int():
    """Well-behaved driver returning a scalar int → same int."""
    assert _coerce_gl_int(2048, fallback=-1) == 2048
    assert _coerce_gl_int(0, fallback=-1) == 0


def test_coerce_gl_int_unwraps_tuple_return():
    """Some GL wrappers report GL_MAX_3D_TEXTURE_SIZE / NVX memory info as a
    tuple (single-element or multi-component). If we treat that as opaque
    and drop through to the NVIDIA heuristic, the RTX 4090's 24 GB card
    gets capped at 1 GB — the user sees visibly worse downsample than
    they should. The helper must unwrap the first numeric element.
    """
    # NVIDIA NVX_CURRENT_AVAILABLE_VIDMEM on some driver builds returns a
    # 1-element tuple of kilobytes.
    assert _coerce_gl_int((20_000_000,), fallback=-1) == 20_000_000
    # GL_MAX_3D_TEXTURE_SIZE on wrappers that always return arrays.
    assert _coerce_gl_int((2048, 0, 0, 0), fallback=-1) == 2048


def test_coerce_gl_int_falls_back_on_unknown():
    """Bytes / strings / None / empty tuple → fallback, don't crash."""
    assert _coerce_gl_int(None, fallback=7) == 7
    assert _coerce_gl_int((), fallback=7) == 7
    assert _coerce_gl_int(b"NVIDIA", fallback=7) == 7
    assert _coerce_gl_int(True, fallback=7) == 7  # bool is skipped


def test_detect_max_3d_dim_handles_tuple_return(monkeypatch):
    """End-to-end: GL wrapper returns a tuple for GL_MAX_3D_TEXTURE_SIZE;
    _detect_max_3d_dim must unwrap it instead of falling back to the
    conservative ``_MAX_3D_DIM_FALLBACK`` default.
    """
    import montaris.widgets.view_3d as v3d
    monkeypatch.setattr(v3d, '_query_gl',
                        lambda canvas, enum, fallback: (2048, 0, 0, 0))
    assert v3d._detect_max_3d_dim(canvas=None) == 2048


def test_detect_vram_budget_unwraps_nvx_tuple(monkeypatch):
    """End-to-end regression for the RTX 4090 report: NVX_CURRENT_AVAILABLE
    returns a tuple on the affected driver build. Without coercion, the
    budget silently drops to the 4 GB heuristic fallback instead of using
    the driver's real ~24 GB reading.
    """
    import montaris.widgets.view_3d as v3d
    # 20 GB free, reported as a 1-element tuple of kilobytes.
    kb_free = 20 * 1024 * 1024  # 20 GB in kB
    monkeypatch.setattr(v3d, '_query_gl',
                        lambda canvas, enum, fallback: (kb_free,))
    # No env override, NVIDIA GPU description.
    monkeypatch.delenv('MONTARIS_GPU_BUDGET_MB', raising=False)
    budget = v3d._detect_vram_budget_bytes(canvas=None,
                                           gpu_desc='NVIDIA RTX 4090')
    # Half of free VRAM = 10 GB.
    assert budget == kb_free * 1024 // 2
    # Sanity: well above the 4 GB heuristic fallback.
    assert budget > 4 * 1024 * 1024 * 1024


def test_detect_vram_budget_falls_through_when_nvx_is_garbage(monkeypatch):
    """If NVX returns an unusable value (None, empty tuple), fall through
    to the GL_RENDERER heuristic — must not crash and must return 4 GB
    for an NVIDIA string.
    """
    import montaris.widgets.view_3d as v3d
    monkeypatch.setattr(v3d, '_query_gl',
                        lambda canvas, enum, fallback: ())  # bogus
    monkeypatch.delenv('MONTARIS_GPU_BUDGET_MB', raising=False)
    budget = v3d._detect_vram_budget_bytes(canvas=None,
                                           gpu_desc='NVIDIA RTX 4090')
    assert budget == 4 * 1024 * 1024 * 1024


def test_detect_vram_budget_uses_amd_ati_meminfo(monkeypatch):
    """AMD cards don't expose NVX but do expose GL_ATI_meminfo. Querying
    TEXTURE_FREE_MEMORY_ATI (0x87FC) returns a 4-tuple of kilobytes where
    the first element is total free texture memory. Without this, AMD
    users get the 4 GB heuristic fallback regardless of actual VRAM.
    """
    import montaris.widgets.view_3d as v3d

    def fake_query(canvas, enum, fallback):
        # NVX enums return nothing (non-NVIDIA), ATI returns free VRAM.
        if enum == v3d._TEXTURE_FREE_MEMORY_ATI:
            # 16 GB free, reported as (total_free, largest_block, aux, aux)
            return (16 * 1024 * 1024, 8 * 1024 * 1024, 0, 0)
        return fallback

    monkeypatch.setattr(v3d, '_query_gl', fake_query)
    monkeypatch.delenv('MONTARIS_GPU_BUDGET_MB', raising=False)
    budget = v3d._detect_vram_budget_bytes(canvas=None,
                                           gpu_desc='AMD Radeon RX 7900 XTX')
    # Half of the 16 GB pool.
    assert budget == 16 * 1024 * 1024 * 1024 // 2
    # And it's above the 4 GB heuristic, proving we took the ATI path.
    assert budget > 4 * 1024 * 1024 * 1024


def test_detect_vram_budget_amd_falls_through_without_ati(monkeypatch):
    """AMD card without the ATI_meminfo extension (old driver / mesa
    subset) must still land on the 4 GB heuristic, not crash.
    """
    import montaris.widgets.view_3d as v3d
    monkeypatch.setattr(v3d, '_query_gl',
                        lambda canvas, enum, fallback: fallback)
    monkeypatch.delenv('MONTARIS_GPU_BUDGET_MB', raising=False)
    budget = v3d._detect_vram_budget_bytes(canvas=None,
                                           gpu_desc='AMD Radeon RX 580')
    assert budget == 4 * 1024 * 1024 * 1024


def test_max_pool_labels_preserves_thin_structures():
    """The 3D labels overlay is downsampled when the stack exceeds the GPU's
    3D-texture cap. Stride decimation destroys sparse thin structures (a
    1-voxel-wide dendrite at an odd index collapses into nothing, producing
    the "skeleton" effect users see on re-toggle). Max-pool must keep any
    block that contained a nonzero label.
    """
    labels = np.zeros((4, 16, 16), dtype=np.uint16)
    # 1-voxel-wide horizontal line at an odd Y — stride-2 misses every voxel.
    labels[1, 1, :] = 7
    strided = labels[::1, ::2, ::1]
    assert (strided > 0).sum() == 0, "sanity: stride loses this sparse line"

    pooled = _max_pool_labels(labels, (1, 2, 1))
    assert pooled.shape == (4, 8, 16)
    # Every X column had a label voxel in its block — all 16 should survive.
    assert (pooled > 0).sum() == 16
    # Max-pool preserves the label id.
    assert int(pooled.max()) == 7


def test_max_pool_labels_single_voxel_survives():
    """A single ROI voxel at an odd coordinate must not be stride-eaten."""
    labels = np.zeros((2, 8, 8), dtype=np.uint16)
    labels[0, 3, 5] = 42
    assert (labels[::1, ::2, ::1] > 0).sum() == 0  # stride misses it
    pooled = _max_pool_labels(labels, (1, 2, 1))
    assert (pooled > 0).sum() == 1
    assert int(pooled.max()) == 42


def test_max_pool_labels_passthrough_when_block_is_one():
    """Block size (1,1,1) → no-op, return array unchanged (identity)."""
    labels = np.arange(2 * 3 * 4, dtype=np.uint16).reshape((2, 3, 4))
    out = _max_pool_labels(labels, (1, 1, 1))
    assert out is labels  # identity, no copy


def test_max_pool_labels_preserves_trailing_remainder():
    """Non-divisible axis (e.g. Y=2599 with dy=2 → matches user's GPU-cap
    downsample). A label on the very last Y-row must not vanish, and the
    output shape must match ``ceil(S/f)`` to line up with the intensity
    volume's stride at ``_fit_to_gpu``.
    """
    labels = np.zeros((4, 7, 4), dtype=np.uint16)
    labels[0, 6, 2] = 9  # trailing row y=6, block (1,2,1) would pool at [6]
    pooled = _max_pool_labels(labels, (1, 2, 1))
    # ceil(7/2) = 4, so pooled Y should have 4 rows — not 3 (floor).
    assert pooled.shape == (4, 4, 4)
    # The trailing row's label ends up at the last output row.
    assert int(pooled[0, 3, 2]) == 9


def test_max_pool_labels_honours_id_map_for_partial_visibility():
    """Partial visibility regression: when a block contains both a visible
    ROI and a hidden one, max-pool used to pick the hidden id (whose cmap
    alpha is 0), so the visible ROI's voxels rendered transparent. With an
    id_map that remaps invisible ids to 0, max-pool picks the visible id.
    """
    labels = np.zeros((2, 4, 4), dtype=np.uint16)
    # Block (1, 2, 1) at output cell (0, 1, 2): input rows y=2 and y=3 at x=2.
    labels[0, 2, 2] = 3     # visible ROI
    labels[0, 3, 2] = 150   # hidden ROI — higher id wins raw max

    raw = _max_pool_labels(labels, (1, 2, 1))
    assert int(raw[0, 1, 2]) == 150, "raw max-pool picks the higher id"

    # id_map remaps label 150 to 0 (hidden).
    id_map = np.arange(151, dtype=labels.dtype)
    id_map[150] = 0
    masked = _max_pool_labels(labels, (1, 2, 1), id_map=id_map)
    assert int(masked[0, 1, 2]) == 3, (
        "with id_map, max-pool must ignore hidden labels so the visible "
        "ROI's voxel survives in the pooled output"
    )
    # Output dtype must match the LUT's, not the raw labels'. The dense-
    # palette path narrows the LUT to uint8 when palette_size ≤ 255, and
    # ``np.maximum(uint16_out, uint8_id_map_result)`` would silently
    # promote back to uint16, defeating the GPU-upload halving.
    assert masked.dtype == id_map.dtype, (
        f"out dtype must follow id_map.dtype ({id_map.dtype}), "
        f"got {masked.dtype}"
    )


def test_max_pool_labels_extends_short_id_map_without_index_error():
    """A malformed caller could pass an id_map too small for the voxel
    value range (e.g. partial sidecar leaves labels_meta missing an id).
    The helper must not IndexError — it extends the LUT with identity
    entries so the upload proceeds instead of crashing the 3D viewer.
    """
    labels = np.zeros((1, 4, 4), dtype=np.uint16)
    labels[0, 0, 0] = 100  # voxel id beyond short id_map
    # id_map only covers 0..9; 100 is out of range.
    short_id_map = np.arange(10, dtype=np.uint16)
    out = _max_pool_labels(labels, (1, 2, 2), id_map=short_id_map)
    # Identity extension preserves the raw label.
    assert int(out.max()) == 100


@pytest.mark.skipif(not VISPY_AVAILABLE, reason="vispy not installed")
def test_dense_palette_lut_avoids_vispy_lut_resampling(qapp):
    """Regression for the neuron_*_branch_labels_full.tif rendering bug.

    vispy's ``Colormap`` uploads a fixed 1024-pixel 1D LUT
    (``LUT_len = 1024``). When raw label ids span a wide sparse range
    (e.g. 195 distinct ids in [0..663] for a multi-branch neuron file),
    the user-supplied N-entry colormap gets resampled to 1024 entries
    and voxel-value → texel mapping shifts by ``value/N``. Voxel id 5
    ended up sampling the texel that contained slot 4's colour, etc.

    Fix: pre-remap raw ids to dense palette indices before upload, so
    the GPU sees ``[0, palette_size]`` instead of raw ``[0, max_id]``.
    This test verifies:
    - ``texture_lut`` maps each visible meta id to a unique dense slot
      starting at 1 (slot 0 = transparent / hidden).
    - The cmap has exactly ``palette_size + 1`` entries.
    - Hidden / opacity=0 ids map to slot 0 in the LUT.
    """
    vol = np.zeros((4, 8, 9), dtype=np.uint16)
    doc = _doc_with_volume("dense_palette", vol)
    doc.labels_3d = np.zeros_like(vol)
    # Sparse, wide-range ids — exactly the case the bug surfaced on.
    doc.labels_3d[0, 0, 0] = 5
    doc.labels_3d[0, 0, 1] = 100
    doc.labels_3d[0, 0, 2] = 663
    doc.labels_3d[0, 0, 3] = 50  # this one will be hidden
    doc.labels_meta = {
        5:   {"name": "a", "color": (255, 0, 0),   "opacity": 128,
              "visible": True,  "fill_mode": "solid"},
        50:  {"name": "b", "color": (0, 255, 0),   "opacity": 128,
              "visible": False, "fill_mode": "solid"},  # hidden
        100: {"name": "c", "color": (0, 0, 255),   "opacity": 128,
              "visible": True,  "fill_mode": "solid"},
        663: {"name": "d", "color": (255, 255, 0), "opacity": 128,
              "visible": True,  "fill_mode": "solid"},
    }
    panel = View3DPanel(None, channels=[("c", vol, (1.0, 1.0, 1.0))],
                        documents=[doc])
    try:
        texture_lut, cmap, palette_size = panel._build_labels_colormap_and_lut(
            doc.labels_meta, max_id=663, dtype=np.uint16,
        )
        # All 4 meta entries (visible + hidden) get a stable slot — that
        # way per-ROI visibility toggles become cmap-only (~ms) instead
        # of LUT-rebuild + texture re-upload (~2.7 s on 851 MB labels).
        # Hidden labels still render as alpha=0 via cmap, not by being
        # filtered out of the LUT.
        assert palette_size == 4
        # LUT dtype is sized to palette_size, NOT input labels.dtype —
        # palette_size=4 fits in uint8 even though raw ids are uint16.
        # Halves GPU upload bandwidth for the texture downstream.
        assert texture_lut.dtype == np.uint8, (
            f"LUT dtype should narrow to uint8 when palette_size ≤ 255, "
            f"got {texture_lut.dtype}"
        )
        # Sorted ids: 5, 50, 100, 663 → slots 1, 2, 3, 4.
        assert texture_lut[0] == 0      # background untouched
        assert texture_lut[5] == 1      # first id (sorted) → slot 1
        assert texture_lut[50] == 2     # hidden id still gets a slot
        assert texture_lut[100] == 3    # next visible id → slot 3
        assert texture_lut[663] == 4    # last visible id → slot 4
        # Cmap has palette_size + 1 colour entries (slot 0 + slots 1..N).
        from vispy.color import Colormap as VispyCmap
        assert isinstance(cmap, VispyCmap)
        # Slot 0 must be fully transparent (alpha=0).
        assert tuple(cmap.colors.rgba[0]) == (0.0, 0.0, 0.0, 0.0)
        # Slot 1 (raw id 5, red, visible) must be opaque red.
        assert cmap.colors.rgba[1][0] > 0.99   # red channel ~1.0
        assert cmap.colors.rgba[1][1] < 0.01   # green ~0
        assert cmap.colors.rgba[1][3] > 0.0    # alpha > 0
        # Slot 2 (raw id 50, hidden) keeps its colour but alpha=0 so
        # the GPU samples it as fully transparent — this is what
        # makes per-ROI visibility toggles cmap-only.
        assert cmap.colors.rgba[2][3] == 0.0, (
            "hidden labels must render as alpha=0 via cmap, not by "
            "being absent from the LUT (LUT must stay stable across "
            "visibility toggles for the cmap-only fast path)"
        )
        # The labels Volume must use NEAREST texture filtering, not the
        # vispy default 'linear'. Linear trilinearly interpolates raw
        # slot values between voxels, so a hidden slot=3 voxel next to
        # slot=0 background produces an interpolated 1.5 → cmap maps
        # to a different (potentially visible) ROI's colour. This shows
        # up as colour halos in unrelated hues — the bug the user
        # reported as "i dont see those colors independently". Pinning
        # this attribute catches a constructor-arg regression.
        panel._rebuild_labels_overlay()
        assert panel._labels_volume is not None
        assert panel._labels_volume.interpolation == 'nearest', (
            f"labels Volume must use 'nearest' filtering for categorical "
            f"slot indices; got {panel._labels_volume.interpolation!r}"
        )
    finally:
        panel.close()


@pytest.mark.skipif(not VISPY_AVAILABLE, reason="vispy not installed")
def test_refresh_labels_cmap_only_when_meta_keys_unchanged(qapp, monkeypatch):
    """Per-ROI visibility/opacity/colour edits via the LayerPanel must
    take a cmap-only fast path — no LUT rebuild, no texture re-upload.
    On the GQR183 file (851 MB / 348M voxels) the full rebuild costs
    ~2.7 s; users reported ``[per-ROI] visibility checkbox: extremely
    slow``. The fix keeps the LUT stable across visibility toggles
    (hidden labels keep their slot but render alpha=0) so when
    ``labels_meta`` keys don't change, ``refresh_labels_meta_only``
    swaps only ``volume.cmap``. Tool callers that mutate
    ``labels_3d`` (paint/erase/fill/wand) must keep using
    ``refresh_labels`` which always full-rebuilds — covered by
    pre-existing tool tests.
    """
    vol = np.zeros((4, 8, 9), dtype=np.uint16)
    vol[0, 0, 0] = 1
    doc = _doc_with_volume("cmap_only", vol)
    doc.labels_3d = np.zeros_like(vol)
    doc.labels_3d[0, 0, 0] = 1
    doc.labels_3d[1, 1, 1] = 2
    doc.labels_meta = {
        1: {"name": "a", "color": (255, 0, 0), "opacity": 128,
            "visible": True, "fill_mode": "solid"},
        2: {"name": "b", "color": (0, 255, 0), "opacity": 128,
            "visible": True, "fill_mode": "solid"},
    }
    panel = View3DPanel(None, channels=[("c", vol, (1.0, 1.0, 1.0))],
                        documents=[doc])
    try:
        qapp.processEvents()
        panel._rebuild_labels_overlay()
        assert panel._labels_volume is not None
        assert panel._labels_meta_keys == frozenset({1, 2})

        # Track which path runs next: full rebuild reassigns
        # ``_labels_tex_shape``; cmap-only does not.
        rebuild_calls = {'count': 0}
        real_do = panel._do_rebuild_labels_overlay

        def counting_do():
            rebuild_calls['count'] += 1
            real_do()

        monkeypatch.setattr(panel, '_do_rebuild_labels_overlay', counting_do)

        cmap_calls = {'count': 0}
        real_cmap = panel._refresh_labels_cmap_inplace

        def counting_cmap():
            cmap_calls['count'] += 1
            real_cmap()

        monkeypatch.setattr(panel, '_refresh_labels_cmap_inplace', counting_cmap)

        # Toggle visibility on an existing meta entry — keys unchanged.
        doc.labels_meta[2]['visible'] = False
        panel.refresh_labels_meta_only()
        assert cmap_calls['count'] == 1, (
            "visibility-only change must take the cmap-only fast path"
        )
        assert rebuild_calls['count'] == 0, (
            "no LUT rebuild should fire when meta keys are stable"
        )

        # Add a new meta entry — keys changed → must full-rebuild even
        # via the meta-only entry point.
        doc.labels_3d[2, 2, 2] = 5
        doc.labels_meta[5] = {"name": "c", "color": (0, 0, 255), "opacity": 128,
                              "visible": True, "fill_mode": "solid"}
        panel.refresh_labels_meta_only()
        assert rebuild_calls['count'] == 1, (
            "new meta entry must trigger a full rebuild — the LUT has "
            "no slot for it yet, cmap-only would render the new ROI as "
            "background (slot 0)"
        )
        # Cache should now reflect the new keyset.
        assert panel._labels_meta_keys == frozenset({1, 2, 5})

        # The bare ``refresh_labels`` (used by paint/erase/fill/wand)
        # must ALWAYS take the full rebuild — even when meta keys are
        # stable — because those tools mutate ``labels_3d`` voxel
        # values and the texture needs re-uploading.
        rebuild_calls['count'] = 0
        cmap_calls['count'] = 0
        panel.refresh_labels()  # bare hook, no meta change
        assert rebuild_calls['count'] == 1, (
            "refresh_labels must always full-rebuild — tools mutate "
            "labels_3d voxels without changing meta keys, the cmap-"
            "only path would skip the texture re-upload and the edit "
            "would be invisible"
        )
        assert cmap_calls['count'] == 0
    finally:
        panel.close()


@pytest.mark.skipif(not VISPY_AVAILABLE, reason="vispy not installed")
def test_fast_path_refreshes_via_set_data_not_raw_texture(qapp, monkeypatch):
    """Regression: the in-place fast path must go through
    ``VolumeVisual.set_data(..., clim=...)`` rather than poking
    ``_texture.set_data`` and reassigning ``clim`` separately.

    Why: vispy's ``VolumeVisual`` caches the uploaded array as ``_last_data``
    and re-uploads it from ``_last_data`` when ``clim`` widens past the
    cached data's own range. If we bypass ``set_data``, ``_last_data`` is
    stale, so a later ``clim`` widen (e.g. after a paint adds a larger
    label id) silently re-uploads OLD voxels and drops the new ones.
    """
    vol = np.zeros((4, 8, 9), dtype=np.uint16)
    vol[0, 0, 0] = 1
    doc = _doc_with_volume("regression", vol)
    doc.labels_3d = np.zeros_like(vol)
    doc.labels_3d[0, 0, 0] = 1
    doc.labels_meta = {1: {"name": "r1", "color": (255, 0, 0),
                           "opacity": 128, "visible": True,
                           "fill_mode": "solid"}}
    panel = View3DPanel(None, channels=[("c", vol, (1.0, 1.0, 1.0))],
                        documents=[doc])
    try:
        # Drain the deferred QTimer.singleShot(0, _rebuild_volumes) the
        # panel constructor schedules — otherwise it fires during the
        # first build's show_progress→processEvents and clobbers
        # _last_total_ds_axes back to (1, 1, 1) for this tiny test
        # volume, breaking the shape-match check on the second build.
        qapp.processEvents()
        # Force the pool path by setting a non-identity downsample so the
        # fast-path shape-match check is actually exercised.
        panel._last_total_ds_axes = (1, 2, 1)
        # Initial build (slow path — no prior visual).
        panel._rebuild_labels_overlay()
        assert panel._labels_volume is not None

        captured = {}

        def fake_set_data(vol_data, clim=None, copy=True):
            captured['shape'] = tuple(vol_data.shape)
            captured['clim'] = tuple(clim) if clim is not None else None
            captured['copy'] = copy

        monkeypatch.setattr(panel._labels_volume, 'set_data', fake_set_data)
        # Also patch _texture.set_data so a regression to the old code path
        # would be visible (captured['raw_texture']).
        monkeypatch.setattr(panel._labels_volume._texture, 'set_data',
                            lambda *a, **kw: captured.setdefault(
                                'raw_texture', True))

        # Re-assert _last_total_ds_axes in case anything reset it; the
        # second build must use the same block as the first so the
        # shape-match check passes and the fast path runs.
        panel._last_total_ds_axes = (1, 2, 1)
        # Capture the volume identity to verify the fast path reused it
        # rather than rebuilding (slow path replaces the visual object,
        # which would otherwise let `clim == (0.0, 2.0)` pass via the
        # rebuild branch and silently regress).
        original_volume_id = id(panel._labels_volume)
        # Add a larger id to doc and trigger another refresh — simulates a
        # paint stroke adding label 5 after the overlay was last built.
        doc.labels_3d[1, 1, 1] = 5
        doc.labels_meta[5] = {"name": "r5", "color": (0, 255, 0),
                              "opacity": 128, "visible": True,
                              "fill_mode": "solid"}
        panel._rebuild_labels_overlay()
        assert id(panel._labels_volume) == original_volume_id, (
            "fast path must reuse the existing Volume visual; if a "
            "rebuild swapped it out, set_data was patched on a stale "
            "object and the test no longer pins the in-place path"
        )

        assert captured.get('raw_texture') is not True, (
            "fast path must route through VolumeVisual.set_data — "
            "bypassing it leaves _last_data stale and a later clim widen "
            "will re-upload outdated voxels"
        )
        # ``clim`` is now ``(0, palette_size)`` — palette_size = 2 here
        # because there are two visible meta entries (id 1 and id 5).
        # The dense-palette LUT (introduced to avoid vispy's LUT_len=1024
        # resampling that scrambled colours on wide-range raw ids) maps
        # raw id 5 to slot 2 before upload, so clim must reflect the
        # palette range, not raw max_id.
        assert captured.get('clim') == (0.0, 2.0), (
            f"clim should match palette_size (=2), got {captured.get('clim')}"
        )
    finally:
        panel.close()


@pytest.mark.skipif(not VISPY_AVAILABLE, reason="vispy not installed")
def test_drag_refresh_tick_remaps_through_dense_palette_lut(qapp, monkeypatch):
    """Regression: live paint drags must remap raw label ids to palette
    slots before the sub-region texture upload. The texture stores DENSE
    palette slots after the dense-palette LUT fix, with
    ``clim=(0, palette_size)`` — uploading raw ids (e.g. 663) into it
    would sample a wrong cmap entry and bypass the slot-0 visibility
    filter. ``_on_drag_refresh_tick`` was the one path that still poked
    raw ``labels_3d`` slices into the texture; this pins the remap.
    """
    vol = np.zeros((4, 8, 9), dtype=np.uint16)
    doc = _doc_with_volume("drag_remap", vol)
    doc.labels_3d = np.zeros_like(vol)
    # Wide-range sparse ids — exactly the case the dense-palette fix
    # was introduced for.
    doc.labels_3d[0, 0, 0] = 663
    doc.labels_meta = {663: {"name": "a", "color": (255, 0, 0),
                             "opacity": 128, "visible": True,
                             "fill_mode": "solid"}}
    panel = View3DPanel(None, channels=[("c", vol, (1.0, 1.0, 1.0))],
                        documents=[doc])
    try:
        qapp.processEvents()
        # Identity downsample so the drag tick's fast path runs.
        panel._last_total_ds_axes = (1, 1, 1)
        panel._rebuild_labels_overlay()
        assert panel._labels_volume is not None
        assert panel._labels_texture_lut is not None
        # Cached LUT should remap raw 663 → slot 1 (only visible id).
        assert int(panel._labels_texture_lut[663]) == 1

        captured = {}

        def fake_set_data(sub, offset=None):
            captured['sub'] = np.array(sub)
            captured['offset'] = tuple(offset) if offset is not None else None

        monkeypatch.setattr(panel._labels_volume._texture, 'set_data',
                            fake_set_data)
        # Simulate a paint stroke writing raw id 663 into a fresh voxel.
        doc.labels_3d[1, 2, 3] = 663
        panel._dirty_bbox = (1, 2, 2, 3, 3, 4)  # (z0,z1,y0,y1,x0,x1)
        panel._drag_dirty = True
        panel._on_drag_refresh_tick()

        assert 'sub' in captured, (
            "drag tick must call Texture3D.set_data on the sub-region"
        )
        assert captured['offset'] == (1, 2, 3)
        # The uploaded sub-region must contain palette slots, NOT the
        # raw id 663. Slot 1 = the only visible meta entry.
        sub = captured['sub']
        assert int(sub.max()) == 1, (
            f"sub-region must be remapped to dense palette slots; "
            f"got max={int(sub.max())} (663 = raw upload regression)"
        )
        assert sub[0, 0, 0] == 1
    finally:
        panel.close()


@pytest.mark.skipif(not VISPY_AVAILABLE, reason="vispy not installed")
def test_rebuild_labels_overlay_reentrancy_is_suppressed(qapp, monkeypatch):
    """show_progress flushes paint events, which can let a queued signal
    re-enter refresh_labels while the outer call is still mid-build. The
    guard must drop the reentrant call so stale locals don't clobber the
    freshly-uploaded state on return.
    """
    vol = np.zeros((4, 8, 9), dtype=np.uint16)
    doc = _doc_with_volume("reentry", vol)
    doc.labels_3d = np.zeros_like(vol)
    doc.labels_3d[0, 0, 0] = 1
    doc.labels_meta = {1: {"name": "r1", "color": (255, 0, 0),
                           "opacity": 128, "visible": True,
                           "fill_mode": "solid"}}
    panel = View3DPanel(None, channels=[("c", vol, (1.0, 1.0, 1.0))],
                        documents=[doc])
    try:
        calls = {'count': 0}
        real_do = panel._do_rebuild_labels_overlay

        def counting_do():
            calls['count'] += 1
            # Inside the first build, re-enter via the public entry point.
            if calls['count'] == 1:
                panel._rebuild_labels_overlay()
            real_do()

        monkeypatch.setattr(panel, '_do_rebuild_labels_overlay', counting_do)

        panel._rebuild_labels_overlay()

        # Only the outer call ran the actual body; the reentrant inner call
        # was short-circuited by the guard.
        assert calls['count'] == 1
    finally:
        panel.close()


@pytest.mark.skipif(not VISPY_AVAILABLE, reason="vispy not installed")
def test_all_hidden_skips_pool_and_upload(qapp, monkeypatch):
    """'Hide all' should not repool/re-upload — just toggle the visual off.
    LayerPanel's toggle-all-off sets every meta['visible']=False; without
    this short-circuit the pool+id_map+upload still runs (~1s on large
    stacks).
    """
    vol = np.zeros((4, 8, 9), dtype=np.uint16)
    doc = _doc_with_volume("hidden", vol)
    doc.labels_3d = np.zeros_like(vol)
    doc.labels_3d[0, 0, 0] = 1
    doc.labels_meta = {1: {"name": "r1", "color": (255, 0, 0),
                           "opacity": 128, "visible": True,
                           "fill_mode": "solid"}}
    panel = View3DPanel(None, channels=[("c", vol, (1.0, 1.0, 1.0))],
                        documents=[doc])
    try:
        panel._last_total_ds_axes = (1, 2, 1)
        panel._rebuild_labels_overlay()
        assert panel._labels_volume is not None

        pool_calls = {'count': 0}
        import montaris.widgets.view_3d as v3d
        real_pool = v3d._max_pool_labels

        def counted_pool(*a, **kw):
            pool_calls['count'] += 1
            return real_pool(*a, **kw)

        monkeypatch.setattr(v3d, '_max_pool_labels', counted_pool)

        # Hide the only ROI and re-trigger.
        doc.labels_meta[1]['visible'] = False
        panel._rebuild_labels_overlay()

        assert pool_calls['count'] == 0, (
            "all-hidden path must skip the pool entirely"
        )
        assert panel._labels_volume.visible is False
    finally:
        panel.close()


def test_max_pool_labels_reuses_out_buffer():
    """Passing a preallocated ``out`` buffer of matching shape/dtype must
    (a) write into it in place, (b) start from zeros, (c) return the same
    buffer so the caller can cache it across refreshes.
    """
    labels = np.zeros((2, 6, 4), dtype=np.uint16)
    labels[0, 2, 2] = 5
    target_shape = (2, 3, 4)
    buf = np.full(target_shape, 99, dtype=np.uint16)  # pre-dirty
    result = _max_pool_labels(labels, (1, 2, 1), out=buf)
    # Returns the same buffer — the visibility-toggle fast path relies on
    # this to avoid re-allocating ~0.5 GB every refresh on the user's data.
    assert result is buf
    # Old values wiped; only the real label remains.
    assert int(buf.max()) == 5
    # Mismatched shape → helper allocates a fresh output (not in place).
    too_small = np.zeros((1, 1, 1), dtype=np.uint16)
    result2 = _max_pool_labels(labels, (1, 2, 1), out=too_small)
    assert result2 is not too_small
    assert result2.shape == target_shape


def test_max_pool_labels_axis_shorter_than_block_still_pools_others():
    """Block on axis A can exceed the array length on A while pooling the
    other axes correctly. Previous stride-fallback re-introduced the
    skeleton bug on axes that could have been pooled.
    """
    labels = np.zeros((1, 4, 4), dtype=np.uint16)
    labels[0, 1, 1] = 5  # odd Y, odd X
    pooled = _max_pool_labels(labels, (2, 2, 2))
    # ceil(1/2)=1, ceil(4/2)=2 — Z axis short, Y/X still pooled.
    assert pooled.shape == (1, 2, 2)
    assert int(pooled[0, 0, 0]) == 5  # voxel at (0,1,1) falls into (0,0,0)


@pytest.mark.skipif(not VISPY_AVAILABLE, reason="vispy not installed")
def test_drag_refresh_tick_uploads_with_zyx_offset(qapp, monkeypatch):
    """Sub-region upload must pass offset=(z0, y0, x0) to Texture3D.set_data."""
    vol = np.zeros((8, 16, 20), dtype=np.uint8)
    doc = _doc_with_volume("zyxdoc", vol)
    doc.ensure_labels_3d()
    panel = View3DPanel(None, channels=[("ch", vol, (1.0, 1.0, 1.0))], documents=[doc])
    try:
        captured = {}

        class FakeTex:
            def set_data(self, data, offset):
                captured['data_shape'] = data.shape
                captured['offset'] = offset

        class FakeVolume:
            def __init__(self):
                self._texture = FakeTex()

            def update(self):
                captured['updated'] = True

        panel._labels_volume = FakeVolume()
        panel._last_total_ds = 1
        # Identity LUT so the dense-palette remap is a no-op for this
        # test (which only pins the offset/shape contract). A real
        # build sets ``_labels_texture_lut`` to a dense palette LUT;
        # the drag tick refuses to upload without one (would otherwise
        # poke raw ids into a dense-palette texture).
        panel._labels_texture_lut = np.arange(256, dtype=np.uint8)
        panel._drag_dirty = True
        # bbox = (z0, z1, y0, y1, x0, x1)
        panel._dirty_bbox = (2, 5, 4, 10, 6, 14)

        panel._on_drag_refresh_tick()

        assert captured.get('offset') == (2, 4, 6)
        # The data uploaded must match the bbox shape in zyx order.
        assert captured.get('data_shape') == (3, 6, 8)
        assert captured.get('updated') is True
    finally:
        panel.close()


def test_z_slider_swaps_image_data_to_slice(app, qapp):
    """Moving the z-slider must replace image_layer.data with volume_data[z]."""
    vol = np.zeros((6, 4, 5), dtype=np.uint8)
    for z in range(vol.shape[0]):
        vol[z] = z * 10  # distinct fingerprint per slice
    mp = vol.max(axis=0)
    layer = ImageLayer("stack", mp)
    app.layer_stack.set_image(layer)
    doc = MontageDocument(
        name="stack",
        image_layer=layer,
        downsample_factor=1,
        original_shape=mp.shape,
        volume_data=vol,
        volume_axes='ZYX',
    )
    app._documents = [doc]
    app._active_doc_index = 0
    app._update_z_slider_visibility()

    # Slider exists and is wired to the doc.
    assert app._z_bar.isVisible()
    app._z_slider.setValue(3)

    assert doc.active_z == 3
    np.testing.assert_array_equal(doc.image_layer.data, vol[3])

    app._z_slider.setValue(0)
    np.testing.assert_array_equal(doc.image_layer.data, vol[0])


def test_layer_panel_3d_mode_filters_2d_rois(app, qapp):
    """In 3D mode the panel must show only VolumeROILayers."""
    h, w = 10, 12
    vol = np.zeros((4, h, w), dtype=np.uint8)
    mp = vol.max(axis=0)
    layer = ImageLayer("stack", mp)
    app.layer_stack.set_image(layer)
    doc = MontageDocument(
        name="stack",
        image_layer=layer,
        downsample_factor=1,
        original_shape=mp.shape,
        volume_data=vol,
        volume_axes='ZYX',
    )
    doc.ensure_labels_3d()
    app._documents = [doc]
    app._active_doc_index = 0

    # Mix a 2D ROI and a 3D ROI
    app.layer_stack.add_roi(ROILayer("flat", w, h))
    lid = doc.reserve_label_id()
    app.layer_stack.roi_layers.append(VolumeROILayer(doc, lid))
    app.layer_panel.refresh()

    # 2D mode — both rows visible + the image row.
    assert app.layer_panel.list_widget.count() == 3

    app.layer_panel.set_3d_mode(True)
    # 3D mode — image row + VolumeROILayer only.
    assert app.layer_panel.list_widget.count() == 2
    assert "3D ROI" in app.layer_panel.header.text()

    app.layer_panel.set_3d_mode(False)
    assert app.layer_panel.list_widget.count() == 3


def test_add_roi_while_3d_open_creates_volume_roi(app, qapp):
    """Clicking Add ROI while the 3D panel is open must allocate a 3D label."""
    h, w = 8, 9
    vol = np.zeros((3, h, w), dtype=np.uint8)
    mp = vol.max(axis=0)
    layer = ImageLayer("stack", mp)
    app.layer_stack.set_image(layer)
    doc = MontageDocument(
        name="stack",
        image_layer=layer,
        downsample_factor=1,
        original_shape=mp.shape,
        volume_data=vol,
        volume_axes='ZYX',
    )
    doc.ensure_labels_3d()
    app._documents = [doc]
    app._active_doc_index = 0

    # Simulate 3D panel being open without actually constructing vispy.
    app._view3d_panel = MagicMock()
    try:
        app._on_roi_added()
    finally:
        app._view3d_panel = None

    rois = app.layer_stack.roi_layers
    assert len(rois) == 1
    assert getattr(rois[0], 'is_volume', False)
    assert rois[0].label_id in doc.labels_meta


def test_remove_volume_roi_clears_voxels_and_meta(app, qapp):
    """Deleting a 3D ROI must zero its voxels and drop its meta entry.

    Exercises the real delete path (``clear_active_roi``) rather than the
    orphaned ``_on_roi_removed`` handler (no caller emits ``roi_removed``).
    """
    h, w = 8, 9
    vol = np.zeros((4, h, w), dtype=np.uint8)
    mp = vol.max(axis=0)
    layer = ImageLayer("stack", mp)
    app.layer_stack.set_image(layer)
    doc = MontageDocument(
        name="stack",
        image_layer=layer,
        downsample_factor=1,
        original_shape=mp.shape,
        volume_data=vol,
        volume_axes='ZYX',
    )
    doc.ensure_labels_3d()
    app._documents = [doc]
    app._active_doc_index = 0

    lid = doc.reserve_label_id()
    # Paint a few voxels into this label so we can verify they're cleared.
    doc.labels_3d[1, 2:5, 3:6] = lid
    vroi = VolumeROILayer(doc, lid)
    app.layer_stack.roi_layers.append(vroi)
    app.layer_panel.refresh()

    assert (doc.labels_3d == lid).any()
    assert lid in doc.labels_meta

    # Pretend the 3D panel is open so the refresh hook fires, and select
    # the 3D ROI so clear_active_roi sees it as the deletion target.
    app._view3d_panel = MagicMock()
    app.canvas.set_active_layer(vroi)
    try:
        app.clear_active_roi()
    finally:
        app._view3d_panel = None

    assert not (doc.labels_3d == lid).any()
    assert lid not in doc.labels_meta
    assert len(app.layer_stack.roi_layers) == 0


def _setup_3d_app(app, shape=(4, 8, 9)):
    """Build a doc with a labels volume attached and register it on app."""
    Z, H, W = shape
    vol = np.zeros(shape, dtype=np.uint8)
    mp = vol.max(axis=0)
    layer = ImageLayer("stack", mp)
    app.layer_stack.set_image(layer)
    doc = MontageDocument(
        name="stack",
        image_layer=layer,
        downsample_factor=1,
        original_shape=mp.shape,
        volume_data=vol,
        volume_axes='ZYX',
    )
    doc.ensure_labels_3d()
    app._documents = [doc]
    app._active_doc_index = 0
    return doc


def test_duplicate_3d_roi_reserves_empty_label_with_src_metadata(app, qapp):
    """3D duplicate creates an empty sibling label (source voxels untouched).

    A single labels volume can't hold two ids at the same voxel, so duplicate
    reserves an empty label carrying the source's color/opacity/fill_mode and
    a ``(copy)`` name — the user paints into it from there.
    """
    doc = _setup_3d_app(app)
    lid = doc.reserve_label_id(name="src", opacity=200, fill_mode="solid")
    doc.labels_3d[1, 2:5, 3:6] = lid
    app.layer_stack.roi_layers.append(VolumeROILayer(doc, lid))

    app.layer_stack.duplicate_roi(0)

    assert len(app.layer_stack.roi_layers) == 2
    dup = app.layer_stack.roi_layers[1]
    assert dup.is_volume
    assert dup._label_id != lid
    assert dup.name == "src (copy)"
    assert dup.opacity == 200
    assert dup.fill_mode == "solid"
    # Source voxels preserved; dup starts empty.
    assert (doc.labels_3d == lid).sum() == 9
    assert not (doc.labels_3d == dup._label_id).any()


def test_duplicate_3d_roi_promotes_dtype_past_255(app, qapp):
    """Duplicate at id boundary (255→256) must promote labels_3d to uint16."""
    doc = _setup_3d_app(app)
    doc.labels_next_id = 255
    lid = doc.reserve_label_id()  # id 255
    doc.labels_3d[0, 0, 0] = lid
    app.layer_stack.roi_layers.append(VolumeROILayer(doc, lid))
    assert doc.labels_3d.dtype == np.uint8

    app.layer_stack.duplicate_roi(0)

    # reserve_label_id inside duplicate promotes the dtype so id 256 fits.
    assert doc.labels_3d.dtype == np.uint16
    dup = app.layer_stack.roi_layers[1]
    assert dup._label_id == 256
    assert doc.labels_3d[0, 0, 0] == 255  # source preserved


def test_merge_3d_rois_relabels_voxels(app, qapp):
    """Merging two 3D ROIs: target absorbs other's voxels, other's id released."""
    doc = _setup_3d_app(app)
    lid_a = doc.reserve_label_id(name="A")
    lid_b = doc.reserve_label_id(name="B")
    doc.labels_3d[0, 0:2, 0:2] = lid_a
    doc.labels_3d[2, 5:7, 5:7] = lid_b
    app.layer_stack.roi_layers.append(VolumeROILayer(doc, lid_a))
    app.layer_stack.roi_layers.append(VolumeROILayer(doc, lid_b))

    assert app.layer_stack.merge_rois([0, 1]) is True

    assert len(app.layer_stack.roi_layers) == 1
    assert app.layer_stack.roi_layers[0]._label_id == lid_a
    assert not (doc.labels_3d == lid_b).any()
    assert lid_b not in doc.labels_meta
    # A's original voxels + B's voxels now both carry lid_a.
    assert (doc.labels_3d == lid_a).sum() == 4 + 4


def test_merge_mixed_2d_3d_rejected(app, qapp):
    """Mixing 2D and 3D ROIs in merge must no-op and return False."""
    doc = _setup_3d_app(app)
    lid = doc.reserve_label_id()
    doc.labels_3d[0, 0, 0] = lid
    app.layer_stack.roi_layers.append(VolumeROILayer(doc, lid))
    app.layer_stack.add_roi(ROILayer("flat", doc.labels_3d.shape[2], doc.labels_3d.shape[1]))

    ok = app.layer_stack.merge_rois([0, 1])

    assert ok is False
    assert len(app.layer_stack.roi_layers) == 2
    assert (doc.labels_3d == lid).any()
    assert lid in doc.labels_meta


def test_remove_3d_roi_undo_restores_voxels(app, qapp):
    """RemoveROIUndoCommand.undo() must rehydrate voxels + meta for 3D ROIs."""
    doc = _setup_3d_app(app)
    lid = doc.reserve_label_id(name="restored", opacity=180)
    doc.labels_3d[1, 2:5, 3:6] = lid
    vroi = VolumeROILayer(doc, lid)
    app.layer_stack.roi_layers.append(vroi)

    app._view3d_panel = MagicMock()
    app.canvas.set_active_layer(vroi)
    try:
        app.clear_active_roi()
        # Voxels and meta must be gone.
        assert not (doc.labels_3d == lid).any()
        assert lid not in doc.labels_meta
        app.undo()
    finally:
        app._view3d_panel = None

    assert len(app.layer_stack.roi_layers) == 1
    assert lid in doc.labels_meta
    assert doc.labels_meta[lid]["name"] == "restored"
    assert doc.labels_meta[lid]["opacity"] == 180
    # Voxels in the bbox are restored exactly.
    mask = (doc.labels_3d == lid)
    assert mask.sum() == 9
    assert mask[1, 2:5, 3:6].all()


def test_volume_stroke_undo_round_trip():
    """VolumeStrokeUndoCommand must restore before-state on undo, after on redo."""
    from montaris.core.undo import VolumeStrokeUndoCommand
    vol = np.zeros((4, 6, 6), dtype=np.uint16)
    doc = _doc_with_volume("d", vol)
    doc.ensure_labels_3d()
    # Simulate a stroke that wrote label 7 into a 2x3x3 region.
    bbox = (1, 3, 2, 5, 1, 4)
    before = np.ascontiguousarray(doc.labels_3d[1:3, 2:5, 1:4])
    doc.labels_3d[1:3, 2:5, 1:4] = 7
    after = np.ascontiguousarray(doc.labels_3d[1:3, 2:5, 1:4])

    cmd = VolumeStrokeUndoCommand(doc, bbox, before, after)
    cmd.undo()
    assert not (doc.labels_3d == 7).any()
    cmd.redo()
    assert (doc.labels_3d == 7).sum() == 2 * 3 * 3


def test_volume_fill_undo_round_trip():
    """VolumeFillUndoCommand must swap old_label/new_label on the mask voxels."""
    from montaris.core.undo import VolumeFillUndoCommand
    vol = np.zeros((3, 5, 5), dtype=np.uint16)
    doc = _doc_with_volume("d", vol)
    doc.ensure_labels_3d()
    # Pretend old_label=3 was there; fill relabeled to new_label=9.
    doc.labels_3d[1, 1:4, 1:4] = 9
    bbox = (1, 2, 1, 4, 1, 4)
    mask_crop = np.ones((1, 3, 3), dtype=bool)
    cmd = VolumeFillUndoCommand(doc, bbox, mask_crop, old_label=3, new_label=9)

    cmd.undo()
    assert (doc.labels_3d[1, 1:4, 1:4] == 3).all()
    assert not (doc.labels_3d == 9).any()
    cmd.redo()
    assert (doc.labels_3d[1, 1:4, 1:4] == 9).all()


def test_stroke_before_snapshot_grows_across_non_contiguous_stamps(app, qapp):
    """``_extend_stroke_before`` must preserve pre-stroke values as the bbox grows.

    Regression guard for the ``_dirty_bbox`` trap — the render-throttle
    bbox is cleared every tick, which previously dropped voxels touched
    across non-contiguous drags. ``_stroke_before`` must retain the
    original pre-stroke values even after the stroke expands into new
    regions.
    """
    if not VISPY_AVAILABLE:
        pytest.skip("vispy not installed")
    vol = np.zeros((6, 12, 12), dtype=np.uint8)
    doc = _doc_with_volume("d", vol)
    doc.ensure_labels_3d()
    # Seed some distinctive pre-stroke values the test can verify survived.
    doc.labels_3d[1, 1, 1] = 11
    doc.labels_3d[4, 8, 8] = 22
    panel = View3DPanel(None, channels=[("c", vol, (1.0, 1.0, 1.0))], documents=[doc])
    try:
        panel._primary_doc = doc
        # First stamp at (1,1,1) captures the "11" value.
        panel._extend_stroke_before(doc.labels_3d, (1, 2, 1, 2, 1, 2))
        doc.labels_3d[1, 1, 1] = 99  # simulate stamp writing the region
        # Second stamp at (4,8,8) extends bbox. The pre-stroke value 22
        # must be captured from labels_3d (still 22 there), and the
        # first stamp's original 11 must NOT be overwritten by the post-
        # stroke 99.
        panel._extend_stroke_before(doc.labels_3d, (4, 5, 8, 9, 8, 9))
        z0, z1, y0, y1, x0, x1 = panel._stroke_bbox
        # Union bbox covers both stamps.
        assert (z0, z1, y0, y1, x0, x1) == (1, 5, 1, 9, 1, 9)
        before = panel._stroke_before
        # (1,1,1) in absolute coords → (0,0,0) in stroke-bbox coords.
        assert before[0, 0, 0] == 11
        # (4,8,8) in absolute → (3,7,7) in stroke-bbox coords.
        assert before[3, 7, 7] == 22
    finally:
        panel.close()


def test_paint_stroke_undo_push_emits_compound_for_new_roi(app, qapp):
    """A new-ROI paint stroke must push a CompoundUndo[Add+Stroke] command."""
    if not VISPY_AVAILABLE:
        pytest.skip("vispy not installed")
    from montaris.core.undo import (
        VolumeStrokeUndoCommand, AddVolumeROIUndoCommand,
    )
    from montaris.core.multi_undo import CompoundUndoCommand

    vol = np.zeros((4, 10, 10), dtype=np.uint8)
    mp = vol.max(axis=0)
    layer = ImageLayer("stack", mp)
    app.layer_stack.set_image(layer)
    doc = MontageDocument(
        name="stack",
        image_layer=layer,
        downsample_factor=1,
        original_shape=mp.shape,
        volume_data=vol,
        volume_axes='ZYX',
    )
    doc.ensure_labels_3d()
    app._documents = [doc]
    app._active_doc_index = 0

    panel = View3DPanel(
        app._central_stack, channels=[("c", vol, (1.0, 1.0, 1.0))], documents=[doc],
    )
    try:
        panel.label_added.connect(app._on_3d_label_added)
        pushed = []
        panel.undo_pushed.connect(pushed.append)

        # Simulate a new-ROI paint: reserve a label id, pretend _stamp_brush
        # snapshotted pre-stroke + wrote voxels, then run _finish_drag(emit=True).
        lid = doc.reserve_label_id()
        panel._primary_doc = doc
        panel._drag_active = True
        panel._drag_mode = 'paint'
        panel._drag_label_id = lid
        panel._drag_extends_existing = False
        # Pretend we stamped: set stroke_before to pre-stroke crop, write
        # voxels, then finish.
        bbox = (1, 2, 2, 5, 3, 6)
        panel._stroke_bbox = bbox
        panel._stroke_before = np.ascontiguousarray(doc.labels_3d[1:2, 2:5, 3:6])
        doc.labels_3d[1, 2:5, 3:6] = lid

        panel._finish_drag(emit=True)

        assert len(pushed) == 1
        cmd = pushed[0]
        assert isinstance(cmd, CompoundUndoCommand)
        kinds = [type(c).__name__ for c in cmd.commands]
        assert "AddVolumeROIUndoCommand" in kinds
        assert "VolumeStrokeUndoCommand" in kinds

        # Ctrl+Z must reverse both voxels AND the wrapper/meta.
        cmd.undo()
        assert not (doc.labels_3d == lid).any()
        assert lid not in doc.labels_meta
        assert all(getattr(r, '_label_id', None) != lid for r in app.layer_stack.roi_layers)

        cmd.redo()
        assert (doc.labels_3d == lid).sum() == 9
        assert lid in doc.labels_meta
    finally:
        panel.close()


def test_erase_stroke_undo_restores_voxels(app, qapp):
    """Erase stroke: Ctrl+Z must put the erased voxels back (no Add)."""
    if not VISPY_AVAILABLE:
        pytest.skip("vispy not installed")
    from montaris.core.undo import VolumeStrokeUndoCommand

    vol = np.zeros((3, 8, 8), dtype=np.uint8)
    mp = vol.max(axis=0)
    layer = ImageLayer("stack", mp)
    app.layer_stack.set_image(layer)
    doc = MontageDocument(
        name="stack",
        image_layer=layer,
        downsample_factor=1,
        original_shape=mp.shape,
        volume_data=vol,
        volume_axes='ZYX',
    )
    doc.ensure_labels_3d()
    app._documents = [doc]
    app._active_doc_index = 0

    lid = doc.reserve_label_id()
    doc.labels_3d[1, 2:5, 2:5] = lid

    panel = View3DPanel(
        app._central_stack, channels=[("c", vol, (1.0, 1.0, 1.0))], documents=[doc],
    )
    try:
        pushed = []
        panel.undo_pushed.connect(pushed.append)

        # Simulate an erase stroke over those voxels.
        panel._primary_doc = doc
        panel._drag_active = True
        panel._drag_mode = 'erase'
        panel._drag_label_id = 0
        panel._drag_extends_existing = False
        bbox = (1, 2, 2, 5, 2, 5)
        panel._stroke_bbox = bbox
        panel._stroke_before = np.ascontiguousarray(doc.labels_3d[1:2, 2:5, 2:5])
        doc.labels_3d[1, 2:5, 2:5] = 0

        panel._finish_drag(emit=False)

        assert len(pushed) == 1
        assert isinstance(pushed[0], VolumeStrokeUndoCommand)
        pushed[0].undo()
        assert (doc.labels_3d == lid).sum() == 9
    finally:
        panel.close()


# ─── Phase 3: session persistence for 3D ROIs ────────────────────────


def test_save_session_skips_volume_roi_from_roi_files(app, qapp, tmp_path):
    """VolumeROILayers don't leak into the per-slice .roi file list."""
    import os
    from montaris.app import _save_session_from_snapshots

    doc = _setup_3d_app(app)
    lid = doc.reserve_label_id(name="vol3d", color=(10, 20, 30))
    doc.labels_3d[1, 2:5, 2:5] = lid
    app.layer_stack.roi_layers.append(VolumeROILayer(doc, lid))

    session_dir = tmp_path / "session"
    # Matches the real code path: VolumeROILayers are filtered out of
    # snapshots before the worker runs.
    snapshots = []
    meta = {
        'version': 1, 'timestamp': 'now', 'image_stem': 'stack',
        'image_path': '', 'downsample_factor': 1, 'original_shape': None,
        'canvas_shape': None, 'channel_names': ['stack'], 'roi_count': 1,
        'roi_names': [], 'roi_colors': [], 'roi_opacities': [],
    }
    _save_session_from_snapshots(
        str(session_dir), snapshots, meta,
        (doc.labels_3d.copy(), dict(doc.labels_meta)),
    )

    # No .roi files should have been written for the 3D ROI.
    roi_files = [f for f in os.listdir(session_dir) if f.endswith('.roi')]
    assert roi_files == []
    assert (session_dir / "labels.tif").exists()
    assert (session_dir / "labels.labels.json").exists()


def test_session_round_trip_restores_labels_and_meta(app, qapp, tmp_path):
    """Save + restore reconstructs labels_3d, labels_meta, and VolumeROILayer wrappers."""
    import os
    import json as _json
    from montaris.app import _save_session_from_snapshots

    doc = _setup_3d_app(app)
    # Promote to uint16 so we also cover the dtype-preservation path on reload.
    doc.promote_labels_dtype(np.uint16)
    lid_a = doc.reserve_label_id(name="alpha", color=(10, 20, 30), opacity=150)
    lid_b = doc.reserve_label_id(name="beta", color=(40, 50, 60), opacity=200)
    doc.labels_3d[1, 2:5, 2:5] = lid_a
    doc.labels_3d[2, 4:7, 4:7] = lid_b
    expected_a = (doc.labels_3d == lid_a).sum()
    expected_b = (doc.labels_3d == lid_b).sum()
    labels_snapshot = doc.labels_3d.copy()

    app.layer_stack.roi_layers.append(VolumeROILayer(doc, lid_a))
    app.layer_stack.roi_layers.append(VolumeROILayer(doc, lid_b))

    session_dir = tmp_path / "session_rt"
    meta = {
        'version': 1, 'timestamp': 'now', 'image_stem': 'stack',
        'image_path': '', 'downsample_factor': 1, 'original_shape': None,
        'canvas_shape': None, 'channel_names': ['stack'], 'roi_count': 2,
        'roi_names': [], 'roi_colors': [], 'roi_opacities': [],
    }
    _save_session_from_snapshots(
        str(session_dir), [], meta,
        (labels_snapshot, dict(doc.labels_meta)),
    )

    # Read back the session.json written by the worker.
    with open(session_dir / 'session.json') as f:
        saved_meta = _json.load(f)
    assert saved_meta.get('labels_file') == 'labels.tif'
    assert list(saved_meta.get('labels_shape', [])) == list(doc.labels_3d.shape)

    # Wipe live state to simulate a fresh session — but keep the doc
    # attached so _restore_session_volume_labels can target it.
    app.layer_stack.roi_layers.clear()
    doc.labels_3d = None
    doc.labels_meta = {}
    doc.labels_next_id = 1

    added = app._restore_session_volume_labels(str(session_dir), saved_meta, doc)
    assert added == 2
    assert doc.labels_3d is not None
    assert doc.labels_3d.shape == labels_snapshot.shape
    assert (doc.labels_3d == lid_a).sum() == expected_a
    assert (doc.labels_3d == lid_b).sum() == expected_b

    # labels_meta values survive round-trip (names, colors, opacities).
    assert doc.labels_meta[lid_a]['name'] == "alpha"
    assert tuple(doc.labels_meta[lid_a]['color']) == (10, 20, 30)
    assert doc.labels_meta[lid_a]['opacity'] == 150
    assert doc.labels_meta[lid_b]['name'] == "beta"

    # labels_next_id advanced past both ids so new paints don't collide.
    assert doc.labels_next_id == max(lid_a, lid_b) + 1

    # VolumeROILayer wrappers were re-created for each restored id.
    vol_layers = [r for r in app.layer_stack.roi_layers if getattr(r, 'is_volume', False)]
    assert {l.label_id for l in vol_layers} == {lid_a, lid_b}


def test_restore_session_volume_labels_noop_without_labels_file(app, qapp, tmp_path):
    """A session meta without labels_file leaves the doc untouched."""
    doc = _setup_3d_app(app)
    original = doc.labels_3d
    result = app._restore_session_volume_labels(str(tmp_path), {}, doc)
    assert result == 0
    assert doc.labels_3d is original


# ─── Phase 4: nav/properties + 2D-only tool guards ─────────────────────


def test_volume_roi_voxel_count_reflects_whole_volume(app, qapp):
    """voxel_count() counts voxels across every Z-slice, not the current one."""
    doc = _setup_3d_app(app)
    lid = doc.reserve_label_id(name="full")
    doc.labels_3d[0, 1:3, 1:3] = lid  # 4 voxels on z=0
    doc.labels_3d[2, 4:6, 4:7] = lid  # 6 voxels on z=2
    layer = VolumeROILayer(doc, lid)
    assert layer.voxel_count() == 10


def test_volume_roi_get_volume_bbox_spans_all_occupied_axes(app, qapp):
    """get_volume_bbox returns (z1,z2,y1,y2,x1,x2) over all occupied voxels."""
    doc = _setup_3d_app(app)
    lid = doc.reserve_label_id(name="bbox")
    doc.labels_3d[1, 2, 3] = lid
    doc.labels_3d[3, 5, 7] = lid
    layer = VolumeROILayer(doc, lid)
    assert layer.get_volume_bbox() == (1, 4, 2, 6, 3, 8)


def test_volume_roi_get_volume_bbox_none_for_empty_label(app, qapp):
    """Empty label returns None so callers can distinguish zero-sized ROIs."""
    doc = _setup_3d_app(app)
    lid = doc.reserve_label_id(name="empty")
    layer = VolumeROILayer(doc, lid)
    assert layer.get_volume_bbox() is None
    assert layer.voxel_count() == 0


def test_properties_panel_shows_voxels_for_volume_roi(app, qapp):
    """Properties panel swaps the label 'Pixels:' → 'Voxels:' for 3D ROIs."""
    doc = _setup_3d_app(app)
    lid = doc.reserve_label_id(name="vol")
    doc.labels_3d[1, 0:3, 0:3] = lid
    wrapper = VolumeROILayer(doc, lid)

    app.properties_panel.set_layer(wrapper)
    assert app.properties_panel.pixel_count_row_label.text() == "Voxels:"
    assert app.properties_panel.pixel_count_label.text() == "9"

    # Swap to a 2D ROI — label should flip back to 'Pixels:'.
    roi2d = ROILayer("r", 8, 9)
    roi2d.mask[0, 0] = 1
    app.properties_panel.set_layer(roi2d)
    assert app.properties_panel.pixel_count_row_label.text() == "Pixels:"


def test_move_tool_ignores_volume_layers(app, qapp):
    """Move tool shouldn't register a 3D ROI as a drag target."""
    from montaris.tools.move import MoveTool

    doc = _setup_3d_app(app)
    lid = doc.reserve_label_id(name="vol")
    doc.labels_3d[1, 0:3, 0:3] = lid
    wrapper = VolumeROILayer(doc, lid)
    app.layer_stack.roi_layers.append(wrapper)

    tool = MoveTool(app)
    # on_press is a no-op for volume layers — nothing to move.
    tool.on_press(MagicMock(x=lambda: 1, y=lambda: 1), wrapper, app.canvas)
    # No marching-ants items should have been registered for the volume layer.
    assert not getattr(tool, '_dragging', False)


def test_transform_tool_ignores_volume_layers(app, qapp):
    """Transform tool refuses to build handles for a 3D ROI."""
    from montaris.tools.transform import TransformTool

    doc = _setup_3d_app(app)
    lid = doc.reserve_label_id(name="vol")
    doc.labels_3d[1, 0:3, 0:3] = lid
    wrapper = VolumeROILayer(doc, lid)

    tool = TransformTool(app)
    tool.on_activate(wrapper, app.canvas)
    # Handles are only created for 2D ROIs; volume layers short-circuit.
    assert tool._bbox_item is None
    assert tool._target_layers == []


def test_restore_session_volume_labels_rejects_shape_mismatch(app, qapp, tmp_path):
    """Mismatched labels shape is skipped with a warning, not raised."""
    from montaris.io.volume_labels import save_volume_labels

    doc = _setup_3d_app(app)
    # Write a labels file with a deliberately wrong shape.
    bad = np.zeros((doc.volume_data.shape[0] + 1,) + doc.volume_data.shape[1:],
                    dtype=np.uint16)
    bad[0, 0, 0] = 1
    save_volume_labels(str(tmp_path / "labels.tif"), bad, {1: {
        'name': 'x', 'color': (1, 2, 3), 'opacity': 128,
        'visible': True, 'fill_mode': 'solid',
    }})

    meta = {'labels_file': 'labels.tif'}
    added = app._restore_session_volume_labels(str(tmp_path), meta, doc)
    assert added == 0
    # doc.labels_3d shape was left matching the doc volume (not the bad file).
    assert doc.labels_3d.shape == doc.volume_data.shape


# ─── Phase 5: additional regression coverage ────────────────────────


def test_merge_cross_doc_3d_rois_rejected(app, qapp):
    """Merge across distinct 3D docs must no-op and leave both docs intact."""
    doc_a = _setup_3d_app(app)
    # Spin up a second volume-bearing doc with its own labels volume.
    vol_b = np.zeros((3, 6, 7), dtype=np.uint8)
    mp_b = vol_b.max(axis=0)
    layer_b = ImageLayer("stack_b", mp_b)
    doc_b = MontageDocument(
        name="stack_b",
        image_layer=layer_b,
        downsample_factor=1,
        original_shape=mp_b.shape,
        volume_data=vol_b,
        volume_axes='ZYX',
    )
    doc_b.ensure_labels_3d()
    app._documents.append(doc_b)

    lid_a = doc_a.reserve_label_id(name="alpha")
    lid_b = doc_b.reserve_label_id(name="beta")
    doc_a.labels_3d[0, 0, 0] = lid_a
    doc_b.labels_3d[0, 0, 0] = lid_b
    app.layer_stack.roi_layers.append(VolumeROILayer(doc_a, lid_a))
    app.layer_stack.roi_layers.append(VolumeROILayer(doc_b, lid_b))

    ok = app.layer_stack.merge_rois([0, 1])

    assert ok is False
    assert len(app.layer_stack.roi_layers) == 2
    assert lid_a in doc_a.labels_meta
    assert lid_b in doc_b.labels_meta
    assert (doc_a.labels_3d == lid_a).any()
    assert (doc_b.labels_3d == lid_b).any()


def test_paint_refuses_when_no_3d_roi_selected(app, qapp, monkeypatch):
    """Paint with no active 3D ROI must not auto-reserve a new label id.

    Regression: paint used to silently allocate a fresh label id + wrapper
    whenever no ROI was selected, surprising users who expected paint
    strokes to extend the ROI they had highlighted in the sidebar.
    """
    if not VISPY_AVAILABLE:
        pytest.skip("vispy not installed")
    from PySide6.QtWidgets import QMessageBox

    doc = _setup_3d_app(app)
    panel = View3DPanel(
        None, channels=[("c", doc.volume_data, (1.0, 1.0, 1.0))], documents=[doc],
    )
    try:
        # Force the panel into paint mode with NO selected 3D ROI. Stub out
        # the ray pick so we don't need a real GL context to produce a seed.
        panel._tool_mode = 'paint'
        panel._primary_doc = doc
        panel._active_volume_roi_id = None
        panel._volumes = [MagicMock()]  # _on_canvas_mouse_press early-outs if empty
        panel._ray_pick_seed = lambda *a, **k: np.array([1, 1, 1])

        # Capture the "No 3D ROI selected" dialog call so it doesn't pop up.
        info_calls = []
        monkeypatch.setattr(
            QMessageBox, "information",
            lambda *a, **k: info_calls.append(a) or QMessageBox.Ok,
        )

        pre_next_id = doc.labels_next_id
        pre_meta = dict(doc.labels_meta)

        event = MagicMock()
        event.button = 1
        event.pos = (0, 0)
        panel._on_canvas_mouse_press(event)

        # No label id was reserved, no voxels were written, and a dialog
        # was shown guiding the user to add/select a 3D ROI first.
        assert doc.labels_next_id == pre_next_id
        assert doc.labels_meta == pre_meta
        assert not panel._drag_active
        assert len(info_calls) == 1

        # Now select a 3D ROI — the next press should go through.
        lid = doc.reserve_label_id(name="existing")
        panel._active_volume_roi_id = lid
        info_calls.clear()
        panel._on_canvas_mouse_press(event)
        assert panel._drag_active
        assert panel._drag_label_id == lid
        assert panel._drag_extends_existing is True
        assert info_calls == []  # no dialog this time
        # Teardown the drag so the panel can close cleanly.
        panel._finish_drag(emit=False)
    finally:
        panel.close()


def test_remove_selected_drops_volume_roi_voxels_and_meta(app, qapp, monkeypatch):
    """Delete button on a 3D ROI must zero voxels + drop meta, not just the wrapper.

    Regression: before the fix, _remove_selected popped the wrapper from
    roi_layers but left the voxels in labels_3d — so the ROI vanished from
    the sidebar but kept rendering in the 3D viewer and got written to
    the next session save.
    """
    from PySide6.QtWidgets import QMessageBox

    doc = _setup_3d_app(app)
    lid = doc.reserve_label_id(name="to-delete", opacity=180)
    doc.labels_3d[1, 2:4, 2:4] = lid
    wrapper = VolumeROILayer(doc, lid)
    app.layer_stack.roi_layers.append(wrapper)

    # Auto-accept the Delete confirmation dialog.
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.Yes)

    # Populate the sidebar so the Delete button can find the wrapper, then
    # select its row. Row 0 is the image layer header; row 1 is the ROI.
    app.layer_panel.refresh()
    row = 1 if app.layer_stack.image_layer else 0
    app.layer_panel.list_widget.setCurrentRow(row)

    app._view3d_panel = MagicMock()
    try:
        app.layer_panel._remove_selected()
        # Wrapper gone from the sidebar AND voxels gone from the volume AND
        # meta dropped. The 3D panel was told to refresh so vispy reuploads.
        assert app.layer_stack.roi_layers == []
        assert lid not in doc.labels_meta
        assert not (doc.labels_3d == lid).any()
        assert app._view3d_panel.refresh_labels.called

        # Undo brings voxels + meta + wrapper back.
        app.undo()
        assert lid in doc.labels_meta
        assert doc.labels_meta[lid]["opacity"] == 180
        assert (doc.labels_3d == lid).sum() == 4
        assert len(app.layer_stack.roi_layers) == 1
    finally:
        app._view3d_panel = None


def test_clear_all_removes_volume_rois_and_is_undoable(app, qapp, monkeypatch):
    """Layer panel's Clear All must drop VolumeROILayers and be reversible."""
    from PySide6.QtWidgets import QMessageBox

    doc = _setup_3d_app(app)
    lid = doc.reserve_label_id(name="go-and-come-back", opacity=180)
    doc.labels_3d[1, 2:4, 2:4] = lid
    wrapper = VolumeROILayer(doc, lid)
    app.layer_stack.roi_layers.append(wrapper)
    # Also drop a flat ROI so we verify the mixed-path clear.
    flat = ROILayer("flat", doc.labels_3d.shape[2], doc.labels_3d.shape[1])
    flat.mask[0, 0] = 1
    app.layer_stack.add_roi(flat)

    # Auto-accept the "Clear All" confirmation dialog.
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.Yes)

    app._view3d_panel = MagicMock()
    try:
        app.layer_panel._clear_all()
        assert app.layer_stack.roi_layers == []
        assert lid not in doc.labels_meta
        assert not (doc.labels_3d == lid).any()

        app.undo()
        # 3D ROI restored with its voxels + meta.
        assert lid in doc.labels_meta
        assert doc.labels_meta[lid]["opacity"] == 180
        assert (doc.labels_3d == lid).sum() == 4
        # Wrapper + flat are back.
        assert len(app.layer_stack.roi_layers) == 2
    finally:
        app._view3d_panel = None


def test_first_paint_vs_extend_undo_is_differentiated(app, qapp):
    """First paint pushes a compound; extending an existing ROI pushes just a stroke."""
    if not VISPY_AVAILABLE:
        pytest.skip("vispy not installed")
    from montaris.core.undo import VolumeStrokeUndoCommand
    from montaris.core.multi_undo import CompoundUndoCommand

    vol = np.zeros((4, 10, 10), dtype=np.uint8)
    mp = vol.max(axis=0)
    layer = ImageLayer("stack", mp)
    app.layer_stack.set_image(layer)
    doc = MontageDocument(
        name="stack",
        image_layer=layer,
        downsample_factor=1,
        original_shape=mp.shape,
        volume_data=vol,
        volume_axes='ZYX',
    )
    doc.ensure_labels_3d()
    app._documents = [doc]
    app._active_doc_index = 0

    panel = View3DPanel(
        app._central_stack, channels=[("c", vol, (1.0, 1.0, 1.0))], documents=[doc],
    )
    try:
        panel.label_added.connect(app._on_3d_label_added)
        pushed = []
        panel.undo_pushed.connect(pushed.append)

        # First paint: brand-new label, no existing wrapper.
        lid = doc.reserve_label_id()
        panel._primary_doc = doc
        panel._drag_active = True
        panel._drag_mode = 'paint'
        panel._drag_label_id = lid
        panel._drag_extends_existing = False
        panel._stroke_bbox = (1, 2, 1, 4, 1, 4)
        panel._stroke_before = doc.labels_3d[1:2, 1:4, 1:4].copy()
        doc.labels_3d[1, 1:4, 1:4] = lid
        panel._finish_drag(emit=True)
        assert isinstance(pushed[-1], CompoundUndoCommand)

        # Second paint on the SAME id: extending an existing wrapper
        # should push only a VolumeStrokeUndoCommand (no Add wrapping).
        panel._drag_active = True
        panel._drag_mode = 'paint'
        panel._drag_label_id = lid
        panel._drag_extends_existing = True
        panel._stroke_bbox = (2, 3, 5, 7, 5, 7)
        panel._stroke_before = doc.labels_3d[2:3, 5:7, 5:7].copy()
        doc.labels_3d[2, 5:7, 5:7] = lid
        panel._finish_drag(emit=True)

        assert isinstance(pushed[-1], VolumeStrokeUndoCommand)
    finally:
        panel.close()


def test_export_volume_labels_falls_back_to_sibling_doc(app, qapp, tmp_path, monkeypatch):
    """Export works when active doc is a sibling channel without labels.

    In a synced-mode batch, sibling channel docs share ``volume_data`` but
    only the 3D viewer's primary doc actually holds the labels. Export
    used to fail with "No 3D ROIs" whenever the user had switched to
    another channel — we now search all docs for one with labels.
    """
    import os
    from PySide6.QtWidgets import QFileDialog

    # doc_a is the "primary" — where 3D paint wrote labels.
    doc_a = _setup_3d_app(app)
    doc_a.name = "primary"
    lid = doc_a.reserve_label_id(name="painted", color=(5, 6, 7), opacity=128)
    doc_a.labels_3d[1, 1:3, 1:3] = lid

    # doc_b simulates a sibling slice doc with no labels attached (the one
    # the user happens to be viewing in 2D when they hit Export).
    vol_b = doc_a.volume_data  # shared volume across synced-mode channels
    layer_b = ImageLayer("slice", vol_b[0])
    doc_b = MontageDocument(
        name="slice_z0",
        image_layer=layer_b,
        downsample_factor=1,
        original_shape=vol_b[0].shape,
        volume_data=vol_b,
        volume_axes='ZYX',
    )
    app._documents = [doc_a, doc_b]
    app._active_doc_index = 1  # active doc is the empty sibling

    out_path = str(tmp_path / "exported.tif")
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (out_path, "")))

    app.export_volume_labels()
    # The labels.tif should exist — export did not bail with "No 3D ROIs".
    assert os.path.isfile(out_path)
    # Sidecar was written too.
    assert os.path.isfile(str(tmp_path / "exported.labels.json"))

    # Round-trip verifies the right doc's labels were exported.
    from montaris.io.volume_labels import load_volume_labels
    loaded, loaded_meta = load_volume_labels(out_path)
    assert (loaded == lid).sum() == 4
    assert loaded_meta[lid]["name"] == "painted"


def test_session_roundtrip_through_app_api(app, qapp, tmp_path):
    """End-to-end: app.save_session_progress + _restore_session_volume_labels."""
    import os

    doc = _setup_3d_app(app)
    doc.image_path = str(tmp_path / "stack.tif")
    # Make the session code think the image lives here — the folder is
    # the one it'll write the timestamped session into.
    lid = doc.reserve_label_id(name="persisted", color=(7, 8, 9), opacity=123)
    doc.labels_3d[1, 1:4, 2:5] = lid
    app.layer_stack.roi_layers.append(VolumeROILayer(doc, lid))

    app.save_session_progress()
    # The worker writes asynchronously; drain the pool before inspecting.
    from montaris.core.workers import get_pool
    get_pool().shutdown(wait=True)
    # Revive the pool so subsequent tests aren't starved.
    import montaris.core.workers as _w
    _w._pool = None

    session_dir = app._session_dir
    assert session_dir is not None and os.path.isdir(session_dir)

    # Wipe and restore.
    doc.labels_3d = None
    doc.labels_meta = {}
    app.layer_stack.roi_layers.clear()
    import json as _json
    with open(os.path.join(session_dir, 'session.json')) as f:
        saved = _json.load(f)
    # Session writes a multi-doc manifest; every listed file should exist.
    manifest = saved.get('volume_labels') or []
    assert manifest and all(
        os.path.isfile(os.path.join(session_dir, e['file'])) for e in manifest
    )
    added = app._restore_session_volume_labels(session_dir, saved, doc)
    assert added == 1
    assert doc.labels_meta[lid]["name"] == "persisted"
    assert (doc.labels_3d == lid).sum() == 9
