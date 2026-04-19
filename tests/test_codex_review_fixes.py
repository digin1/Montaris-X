"""Regression tests for the 2026-04-19 codex review fixes.

Covers:

- ``close_image`` tears down the embedded 3D panel before clearing docs so
  View3DPanel can't keep mutating orphaned documents.
- ``import_volume_labels`` + session restore call ``purge_for_doc`` so stale
  undo patches can't replay into the freshly rehydrated labels array.
- ``load_volume_labels`` preserves uint32 IDs and rejects negative-signed TIFFs.
- Non-cubic GPU-cap downsample keeps per-axis factors so labels align with
  the intensity visual instead of being uniformly scaled.
- Session save writes one labels TIFF per labeled doc (multi-doc manifest).
- Properties-panel opacity change triggers a 3D overlay refresh.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from montaris.core.undo import (
    UndoStack,
    VolumeStrokeUndoCommand,
    VolumeFillUndoCommand,
)
from montaris.core.multi_undo import CompoundUndoCommand
from montaris.layers import (
    ImageLayer,
    MontageDocument,
    VolumeROILayer,
)


def _doc_with_labels(name="stack", shape=(4, 8, 9)):
    vol = np.zeros(shape, dtype=np.uint8)
    layer = ImageLayer(name, vol.max(axis=0))
    doc = MontageDocument(
        name=name,
        image_layer=layer,
        downsample_factor=1,
        original_shape=layer.data.shape,
        volume_data=vol,
        volume_axes="ZYX",
    )
    doc.ensure_labels_3d()
    return doc


# ── UndoStack.purge_for_doc ────────────────────────────────────────────

def test_purge_for_doc_drops_volume_commands_for_that_doc():
    doc_a = _doc_with_labels("a")
    doc_b = _doc_with_labels("b")
    stack = UndoStack()
    stroke = VolumeStrokeUndoCommand(
        doc_a, (0, 1, 0, 2, 0, 2),
        np.zeros((1, 2, 2), dtype=np.uint8),
        np.ones((1, 2, 2), dtype=np.uint8),
    )
    fill = VolumeFillUndoCommand(
        doc_a, (0, 1, 0, 2, 0, 2),
        np.ones((1, 2, 2), dtype=bool), 0, 5,
    )
    survivor = VolumeStrokeUndoCommand(
        doc_b, (0, 1, 0, 2, 0, 2),
        np.zeros((1, 2, 2), dtype=np.uint8),
        np.ones((1, 2, 2), dtype=np.uint8),
    )
    stack.push(stroke)
    stack.push(fill)
    stack.push(survivor)
    assert stack.can_undo
    stack.purge_for_doc(doc_a)
    # Only doc_b's command remains; doc_a's are gone.
    assert stack._stack == [survivor]
    assert stack._index == 0


def test_purge_for_doc_drops_compound_if_any_child_matches():
    doc_a = _doc_with_labels("a")
    doc_b = _doc_with_labels("b")
    stack = UndoStack()
    child_a = VolumeStrokeUndoCommand(
        doc_a, (0, 1, 0, 2, 0, 2),
        np.zeros((1, 2, 2), dtype=np.uint8),
        np.ones((1, 2, 2), dtype=np.uint8),
    )
    child_b = VolumeStrokeUndoCommand(
        doc_b, (0, 1, 0, 2, 0, 2),
        np.zeros((1, 2, 2), dtype=np.uint8),
        np.ones((1, 2, 2), dtype=np.uint8),
    )
    compound = CompoundUndoCommand([child_a, child_b])
    stack.push(compound)
    stack.purge_for_doc(doc_a)
    # The compound is gone even though child_b was for doc_b; a partial
    # replay would still corrupt doc_a's voxels.
    assert stack._stack == []
    assert stack._index == -1


def test_purge_for_doc_adjusts_index_when_current_is_purged():
    doc_a = _doc_with_labels("a")
    doc_b = _doc_with_labels("b")
    stack = UndoStack()
    for _ in range(3):
        stack.push(VolumeStrokeUndoCommand(
            doc_a, (0, 1, 0, 2, 0, 2),
            np.zeros((1, 2, 2), dtype=np.uint8),
            np.ones((1, 2, 2), dtype=np.uint8),
        ))
    stack.push(VolumeStrokeUndoCommand(
        doc_b, (0, 1, 0, 2, 0, 2),
        np.zeros((1, 2, 2), dtype=np.uint8),
        np.ones((1, 2, 2), dtype=np.uint8),
    ))
    stack.undo()  # index now 2 (last doc_a)
    stack.purge_for_doc(doc_a)
    # After dropping every doc_a command, only the doc_b command remains.
    # index was 2 -> minus 3 removed before it -> -1.
    assert len(stack._stack) == 1
    assert stack._index == -1


# ── close_image tears down 3D ─────────────────────────────────────────

def test_close_image_tears_down_mounted_3d_panel(app, qapp):
    from montaris.widgets.view_3d import VISPY_AVAILABLE
    if not VISPY_AVAILABLE:
        pytest.skip("vispy not installed")
    vol = np.zeros((4, 8, 9), dtype=np.uint8)
    mp = vol.max(axis=0)
    layer = ImageLayer("s", mp)
    app.layer_stack.set_image(layer)
    doc = MontageDocument(
        name="s", image_layer=layer, downsample_factor=1,
        original_shape=mp.shape, volume_data=vol, volume_axes="ZYX",
    )
    doc.ensure_labels_3d()
    app._documents = [doc]
    app._active_doc_index = 0

    app._open_view_3d()
    assert app._view3d_panel is not None
    # Suspend should have cleared the tool shortcuts.
    assert any(
        btn.shortcut().isEmpty()
        for btn in app.tool_panel._tool_buttons.values()
    )

    # Closing the image while 3D is mounted MUST release the panel.
    from montaris.widgets.alert_modal import AlertModal
    # Close path takes the "no ROIs" branch (no dialog). Ensure no confirm
    # popup is launched by confirming the stack is empty before close.
    app.layer_stack.roi_layers.clear()
    app.close_image()
    assert app._view3d_panel is None
    # Tool shortcuts must be restored so 2D keys work again.
    assert any(
        not btn.shortcut().isEmpty()
        for btn in app.tool_panel._tool_buttons.values()
    )


# ── load_volume_labels dtype handling ─────────────────────────────────

def test_load_volume_labels_preserves_uint32(tmp_path):
    import tifffile
    from montaris.io.volume_labels import load_volume_labels
    big = np.zeros((2, 4, 4), dtype=np.uint32)
    big[0, 0, 0] = 70000  # > uint16 max
    path = tmp_path / "big.tif"
    tifffile.imwrite(str(path), big, photometric="minisblack")
    arr, meta = load_volume_labels(str(path))
    # The id must NOT have been wrapped into uint16.
    assert arr.dtype == np.uint32
    assert int(arr.max()) == 70000


def test_load_volume_labels_rejects_negative_signed(tmp_path):
    import tifffile
    from montaris.io.volume_labels import load_volume_labels
    bad = np.array([[[-1, 0, 0, 0]] * 4] * 2, dtype=np.int32)
    path = tmp_path / "bad.tif"
    tifffile.imwrite(str(path), bad, photometric="minisblack")
    with pytest.raises(ValueError, match="negative"):
        load_volume_labels(str(path))


def test_load_volume_labels_rejects_float_dtype(tmp_path):
    import tifffile
    from montaris.io.volume_labels import load_volume_labels
    bad = np.zeros((2, 3, 3), dtype=np.float32)
    path = tmp_path / "bad.tif"
    tifffile.imwrite(str(path), bad, photometric="minisblack")
    with pytest.raises(ValueError, match="integer dtype"):
        load_volume_labels(str(path))


def test_load_volume_labels_accepts_unsigned_int32_positive(tmp_path):
    import tifffile
    from montaris.io.volume_labels import load_volume_labels
    arr = np.zeros((2, 3, 3), dtype=np.int32)
    arr[0, 0, 0] = 300  # > uint8 max
    path = tmp_path / "fits16.tif"
    tifffile.imwrite(str(path), arr, photometric="minisblack")
    got, _ = load_volume_labels(str(path))
    # Positive int32 with max 300 collapses to uint16 — not uint8 or uint32.
    assert got.dtype == np.uint16
    assert int(got.max()) == 300


def test_save_volume_labels_promotes_to_uint32_for_large_ids(tmp_path):
    import tifffile
    from montaris.io.volume_labels import save_volume_labels, load_volume_labels
    vol = np.zeros((2, 3, 3), dtype=np.uint32)
    vol[0, 0, 0] = 70000
    meta = {70000: {"name": "big", "color": (1, 2, 3), "opacity": 128,
                    "visible": True, "fill_mode": "solid"}}
    path, _ = save_volume_labels(str(tmp_path / "big.tif"), vol, meta)
    # uint32 bytes on disk. Re-read and verify.
    got, got_meta = load_volume_labels(str(path))
    assert got.dtype in (np.uint32,)
    assert int(got.max()) == 70000
    assert 70000 in got_meta


# ── Anisotropic GPU downsample ────────────────────────────────────────

def test_last_total_ds_axes_initialised():
    from montaris.widgets.view_3d import VISPY_AVAILABLE, View3DPanel
    if not VISPY_AVAILABLE:
        pytest.skip("vispy not installed")
    vol = np.zeros((4, 8, 9), dtype=np.uint8)
    doc = _doc_with_labels("s", shape=vol.shape)
    panel = View3DPanel(None, channels=[("c", vol, (1.0, 1.0, 1.0))], documents=[doc])
    try:
        assert panel._last_total_ds_axes == (1, 1, 1)
        assert panel._last_total_ds == 1
    finally:
        panel.close()


def test_fit_to_gpu_returns_per_axis_factors():
    from montaris.widgets.view_3d import _fit_to_gpu
    # Shape that only exceeds the cap on Y and X, not Z.
    vol = np.zeros((8, 256, 256), dtype=np.uint8)
    _, factors = _fit_to_gpu(vol, max_dim=100)
    # Z axis fits (8 <= 100) → factor 1; Y/X need striding.
    assert factors[0] == 1
    assert factors[1] >= 3
    assert factors[2] >= 3


# ── Session multi-doc manifest ────────────────────────────────────────

def test_session_save_writes_manifest_for_multiple_labeled_docs(tmp_path):
    from montaris.app import _save_session_from_snapshots

    doc_a = _doc_with_labels("docA")
    lid_a = doc_a.reserve_label_id(name="alpha", color=(1, 2, 3), opacity=100)
    doc_a.labels_3d[0, 0, 0] = lid_a

    doc_b = _doc_with_labels("docB")
    lid_b = doc_b.reserve_label_id(name="beta", color=(4, 5, 6), opacity=200)
    doc_b.labels_3d[1, 2, 3] = lid_b

    session_dir = tmp_path / "session"
    meta = {
        'version': 1, 'timestamp': 'now', 'image_stem': 'stack',
        'image_path': '', 'downsample_factor': 1, 'original_shape': None,
        'canvas_shape': None, 'channel_names': ['docA', 'docB'],
        'roi_count': 2, 'roi_names': [], 'roi_colors': [], 'roi_opacities': [],
    }
    _save_session_from_snapshots(
        str(session_dir), [], meta,
        [
            ("docA", doc_a.labels_3d.copy(), dict(doc_a.labels_meta)),
            ("docB", doc_b.labels_3d.copy(), dict(doc_b.labels_meta)),
        ],
    )

    import json
    with open(session_dir / "session.json") as f:
        saved = json.load(f)
    manifest = saved.get("volume_labels") or []
    assert len(manifest) == 2
    assert {e['doc_name'] for e in manifest} == {"docA", "docB"}
    for entry in manifest:
        assert (session_dir / entry['file']).exists()


def test_restore_session_rehydrates_per_doc_from_manifest(app, qapp, tmp_path):
    from montaris.app import _save_session_from_snapshots

    doc_a = _doc_with_labels("docA")
    lid_a = doc_a.reserve_label_id(name="alpha", color=(1, 2, 3), opacity=100)
    doc_a.labels_3d[0, 0, 0] = lid_a

    doc_b = _doc_with_labels("docB")
    lid_b = doc_b.reserve_label_id(name="beta", color=(4, 5, 6), opacity=200)
    doc_b.labels_3d[1, 2, 3] = lid_b

    # Attach both docs to the app.
    app.layer_stack.set_image(doc_a.image_layer)
    app._documents = [doc_a, doc_b]
    app._active_doc_index = 0

    session_dir = tmp_path / "sess"
    meta_in = {
        'version': 1, 'timestamp': 'now', 'image_stem': 'stack',
        'image_path': '', 'downsample_factor': 1, 'original_shape': None,
        'canvas_shape': None, 'channel_names': ['docA', 'docB'],
        'roi_count': 2, 'roi_names': [], 'roi_colors': [], 'roi_opacities': [],
    }
    _save_session_from_snapshots(
        str(session_dir), [], meta_in,
        [
            ("docA", doc_a.labels_3d.copy(), dict(doc_a.labels_meta)),
            ("docB", doc_b.labels_3d.copy(), dict(doc_b.labels_meta)),
        ],
    )

    # Wipe state and restore.
    doc_a.labels_3d = None
    doc_a.labels_meta = {}
    doc_b.labels_3d = None
    doc_b.labels_meta = {}

    import json
    with open(session_dir / "session.json") as f:
        saved = json.load(f)
    added = app._restore_session_volume_labels(str(session_dir), saved, doc_a)
    # Both docs should be rehydrated — not just the "target" doc_a.
    assert added == 2
    assert doc_a.labels_meta[lid_a]['name'] == 'alpha'
    assert doc_b.labels_meta[lid_b]['name'] == 'beta'


# ── Properties-panel opacity → 3D refresh ─────────────────────────────

def test_opacity_change_refreshes_3d_overlay(app, qapp):
    """Changing a VolumeROILayer's opacity should kick refresh_labels."""
    doc = _doc_with_labels()
    app.layer_stack.set_image(doc.image_layer)
    app._documents = [doc]
    app._active_doc_index = 0
    lid = doc.reserve_label_id(name="roi", color=(10, 20, 30), opacity=100)
    wrapper = VolumeROILayer(doc, lid)
    app.layer_stack.roi_layers.append(wrapper)

    # Fake 3D panel so we can observe refresh_labels being called.
    panel = MagicMock()
    app._view3d_panel = panel
    try:
        app.properties_panel.set_layer(wrapper)
        app.properties_panel._on_opacity_changed(200)
        assert wrapper.opacity == 200
        panel.refresh_labels.assert_called()
    finally:
        app._view3d_panel = None


# ── Undo invalidation on labels replacement ───────────────────────────

def test_import_volume_labels_purges_undo_for_target_doc(app, qapp, tmp_path,
                                                         monkeypatch):
    """Undo commands tied to the pre-import labels must not replay after import."""
    from PySide6.QtWidgets import QFileDialog
    from montaris.io.volume_labels import save_volume_labels

    doc = _doc_with_labels()
    app.layer_stack.set_image(doc.image_layer)
    app._documents = [doc]
    app._active_doc_index = 0

    # Seed a volume undo entry for this doc — something that would blow up
    # if replayed after labels_3d is replaced.
    pre = VolumeStrokeUndoCommand(
        doc, (0, 1, 0, 2, 0, 2),
        np.zeros((1, 2, 2), dtype=np.uint8),
        np.ones((1, 2, 2), dtype=np.uint8),
    )
    app.undo_stack.push(pre)
    assert app.undo_stack.can_undo

    # Write a labels TIFF to import.
    labels = np.zeros(doc.volume_data.shape, dtype=np.uint8)
    labels[0, 0, 0] = 1
    meta = {1: {"name": "imported", "color": (9, 9, 9), "opacity": 128,
                "visible": True, "fill_mode": "solid"}}
    out, _ = save_volume_labels(str(tmp_path / "in.tif"), labels, meta)
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(out), "")))

    app.import_volume_labels()
    # The pre-import command must have been purged.
    assert not app.undo_stack.can_undo
    # And the imported labels landed on the doc.
    assert 1 in doc.labels_meta
