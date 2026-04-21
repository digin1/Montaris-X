"""Regression: editing after 'Import 3D ROIs from labels TIFF'.

After importing a labels TIFF while the 3D viewer is open, the wand/paint
tools refused to write with "Wand needs a 3D ROI to write into" because
``View3DPanel._active_volume_roi_id`` was never set — ``_add_volume_roi``
auto-selects the new row but ``import_volume_labels`` didn't.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from montaris.io.volume_labels import save_volume_labels
from montaris.layers import (
    ImageLayer,
    MontageDocument,
    ROILayer,
    VolumeROILayer,
)
from montaris.widgets.view_3d import VISPY_AVAILABLE, View3DPanel


def _attach_volume_doc(app, vol):
    mp = vol.max(axis=0)
    layer = ImageLayer("stack", mp)
    app.layer_stack.set_image(layer)
    doc = MontageDocument(
        name="stack",
        image_layer=layer,
        downsample_factor=1,
        original_shape=mp.shape,
        volume_data=vol,
        volume_axes="ZYX",
    )
    app._documents = [doc]
    app._active_doc_index = 0
    return doc


def _write_labels_tiff(tmp_path: Path) -> Path:
    """Build a tiny labels TIFF + sidecar on disk. Returns the TIFF path."""
    labels = np.zeros((4, 16, 16), dtype=np.uint16)
    labels[1, 2:6, 2:6] = 1
    labels[2, 8:12, 8:12] = 2
    meta = {
        1: {"name": "lbl1", "color": (255, 0, 0), "opacity": 128,
            "visible": True, "fill_mode": "solid"},
        2: {"name": "lbl2", "color": (0, 255, 0), "opacity": 128,
            "visible": True, "fill_mode": "solid"},
    }
    out = tmp_path / "labels.tif"
    save_volume_labels(out, labels, meta)
    return out


@pytest.mark.skipif(not VISPY_AVAILABLE, reason="vispy not installed")
def test_import_labels_auto_selects_first_roi(app, qapp, tmp_path):
    """After import, the 3D panel must have an active volume ROI id set.

    Without it, paint/wand immediately refuse with
    'Wand needs a 3D ROI to write into'.
    """
    vol = np.zeros((4, 16, 16), dtype=np.uint16)
    doc = _attach_volume_doc(app, vol)

    # Attach a real View3DPanel so import_volume_labels can call refresh_labels
    # and, after the fix, can push selection through _on_layer_selected.
    panel = View3DPanel(None, channels=[("ch", vol, (1.0, 1.0, 1.0))],
                        documents=[doc])
    try:
        app._view3d_panel = panel
        app.layer_panel.set_3d_mode(True)

        assert panel._active_volume_roi_id is None

        tiff = _write_labels_tiff(tmp_path)
        with patch(
            "montaris.app.QFileDialog.getOpenFileName",
            return_value=(str(tiff), ""),
        ):
            app.import_volume_labels()

        # Meta populated and VolumeROILayers mirrored.
        assert set(doc.labels_meta.keys()) == {1, 2}
        vol_rois = [r for r in app.layer_stack.roi_layers
                    if isinstance(r, VolumeROILayer) and r._doc is doc]
        assert len(vol_rois) == 2

        # Core assertion: the panel knows which ROI a subsequent wand/paint
        # stroke should extend. Without this, all edit tools refuse.
        active = panel._active_volume_roi_id
        assert active is not None, (
            "import_volume_labels did not auto-select any 3D ROI; paint/wand "
            "will fail with 'Wand needs a 3D ROI to write into'"
        )
        assert active in doc.labels_meta
    finally:
        app._view3d_panel = None
        panel.close()


@pytest.mark.skipif(not VISPY_AVAILABLE, reason="vispy not installed")
def test_import_auto_select_handles_2d_roi_filter(app, qapp, tmp_path):
    """In 3D mode the panel filters 2D ROIs out of the list widget, so a
    roi_layers-index-to-row mapping would be off by the number of 2D ROIs.
    The selection logic must find the correct list row regardless.
    """
    vol = np.zeros((4, 16, 16), dtype=np.uint16)
    doc = _attach_volume_doc(app, vol)

    # Pre-seed a 2D ROI that will NOT show in 3D mode's filtered list.
    app.layer_stack.add_roi(ROILayer("flat", 16, 16))

    panel = View3DPanel(None, channels=[("ch", vol, (1.0, 1.0, 1.0))],
                        documents=[doc])
    try:
        app._view3d_panel = panel
        app.layer_panel.set_3d_mode(True)

        tiff = _write_labels_tiff(tmp_path)
        with patch(
            "montaris.app.QFileDialog.getOpenFileName",
            return_value=(str(tiff), ""),
        ):
            app.import_volume_labels()

        # Active id must correspond to the FIRST imported label, not the one
        # that would be picked by a naive row-index calculation.
        vol_rois = [r for r in app.layer_stack.roi_layers
                    if isinstance(r, VolumeROILayer) and r._doc is doc]
        assert vol_rois, "imported VolumeROILayers missing"
        first_imported_lid = vol_rois[0]._label_id
        assert panel._active_volume_roi_id == first_imported_lid
    finally:
        app._view3d_panel = None
        panel.close()


@pytest.mark.skipif(not VISPY_AVAILABLE, reason="vispy not installed")
def test_second_import_over_same_row_still_syncs(app, qapp, tmp_path):
    """Second import where the first-imported row number matches the already-
    selected row number. Qt's setCurrentRow is a no-op in that case, so the
    force-sync path must kick in or _active_volume_roi_id points at a
    labels_meta entry that's just been replaced.
    """
    vol = np.zeros((4, 16, 16), dtype=np.uint16)
    doc = _attach_volume_doc(app, vol)
    panel = View3DPanel(None, channels=[("ch", vol, (1.0, 1.0, 1.0))],
                        documents=[doc])
    try:
        app._view3d_panel = panel
        app.layer_panel.set_3d_mode(True)

        tiff = _write_labels_tiff(tmp_path)
        with patch(
            "montaris.app.QFileDialog.getOpenFileName",
            return_value=(str(tiff), ""),
        ):
            app.import_volume_labels()
        # Selection lands on first imported label; capture its current row.
        first_row = app.layer_panel.list_widget.currentRow()

        # Re-import the same TIFF — meta is rebuilt fresh. The roi_layers
        # length after drop+append is identical, so first-imported row will
        # match first_row again. Without force-sync, setCurrentRow no-ops.
        with patch(
            "montaris.app.QFileDialog.getOpenFileName",
            return_value=(str(tiff), ""),
        ):
            app.import_volume_labels()

        assert app.layer_panel.list_widget.currentRow() == first_row
        active = panel._active_volume_roi_id
        assert active is not None
        assert active in doc.labels_meta
        # Make sure canvas._active_layer points at a live wrapper, not a
        # dangling one from the pre-replacement list.
        live = app.canvas._active_layer
        assert live is not None and live in app.layer_stack.roi_layers
    finally:
        app._view3d_panel = None
        panel.close()


@pytest.mark.skipif(not VISPY_AVAILABLE, reason="vispy not installed")
def test_on_3d_label_added_selects_under_2d_filter(app, qapp):
    """_on_3d_label_added (the ``+`` button path) must also select the new
    VolumeROILayer correctly when 2D ROIs are hidden by the 3D-mode filter.
    Previously the raw row-index math was off by the number of hidden rows.
    """
    vol = np.zeros((4, 16, 16), dtype=np.uint16)
    doc = _attach_volume_doc(app, vol)
    # 2D ROI pre-seeded — filtered out of the list widget in 3D mode, so
    # roi_layers indexing no longer matches widget row indexing.
    app.layer_stack.add_roi(ROILayer("flat", 16, 16))

    panel = View3DPanel(None, channels=[("ch", vol, (1.0, 1.0, 1.0))],
                        documents=[doc])
    try:
        app._view3d_panel = panel
        app.layer_panel.set_3d_mode(True)

        # Drive the ``+`` path directly.
        app._add_volume_roi()

        active = panel._active_volume_roi_id
        assert active is not None
        assert active in doc.labels_meta
    finally:
        app._view3d_panel = None
        panel.close()


@pytest.mark.skipif(not VISPY_AVAILABLE, reason="vispy not installed")
def test_import_targets_primary_doc_when_3d_open(app, qapp, tmp_path):
    """Multi-channel session: active combo doc != View3DPanel._primary_doc.

    The panel picks _primary_doc once at construction (first volume doc)
    and never changes. If import honored the combo's active doc instead,
    labels would land on a doc the wand/paint guard doesn't read, so every
    edit tool would keep refusing. Labels must follow the 3D editor, not
    the combo.
    """
    shape = (4, 16, 16)
    vol_a = np.zeros(shape, dtype=np.uint16)
    vol_b = np.zeros(shape, dtype=np.uint16)
    mp_a = vol_a.max(axis=0)
    mp_b = vol_b.max(axis=0)
    doc_a = MontageDocument(
        name="A", image_layer=ImageLayer("A", mp_a),
        downsample_factor=1, original_shape=mp_a.shape,
        volume_data=vol_a, volume_axes="ZYX",
    )
    doc_b = MontageDocument(
        name="B", image_layer=ImageLayer("B", mp_b),
        downsample_factor=1, original_shape=mp_b.shape,
        volume_data=vol_b, volume_axes="ZYX",
    )
    app.layer_stack.set_image(doc_a.image_layer)
    app._documents = [doc_a, doc_b]
    # Active combo = B (second), but panel's _primary_doc will be A (first).
    app._active_doc_index = 1

    panel = View3DPanel(None,
                        channels=[("A", vol_a, (1.0, 1.0, 1.0)),
                                  ("B", vol_b, (1.0, 1.0, 1.0))],
                        documents=[doc_a, doc_b])
    try:
        app._view3d_panel = panel
        app.layer_panel.set_3d_mode(True)
        assert panel._primary_doc is doc_a

        tiff = _write_labels_tiff(tmp_path)
        with patch(
            "montaris.app.QFileDialog.getOpenFileName",
            return_value=(str(tiff), ""),
        ):
            app.import_volume_labels()

        # Labels must have landed on primary (A), not the active combo (B).
        assert doc_a.labels_3d is not None
        assert set(doc_a.labels_meta.keys()) == {1, 2}
        assert doc_b.labels_3d is None
        assert not doc_b.labels_meta

        # Wand guard reads from panel._primary_doc — must pass.
        primary = panel._primary_doc
        assert primary is not None
        active = panel._active_volume_roi_id
        assert active is not None
        assert active in primary.labels_meta
    finally:
        app._view3d_panel = None
        panel.close()


@pytest.mark.skipif(not VISPY_AVAILABLE, reason="vispy not installed")
def test_import_clears_synced_sibling_stale_labels(app, qapp, tmp_path):
    """Synced channels share the same volume_data ndarray. If a sibling doc
    already holds labels_3d (e.g. from an older session), importing into
    _primary_doc must purge those siblings or export/save leaks stale rows
    and the LayerPanel shows ghost entries.
    """
    shape = (4, 16, 16)
    shared_vol = np.zeros(shape, dtype=np.uint16)  # same ndarray, aliased
    mp = shared_vol.max(axis=0)
    doc_a = MontageDocument(
        name="A", image_layer=ImageLayer("A", mp),
        downsample_factor=1, original_shape=mp.shape,
        volume_data=shared_vol, volume_axes="ZYX",
    )
    doc_b = MontageDocument(
        name="B", image_layer=ImageLayer("B", mp),
        downsample_factor=1, original_shape=mp.shape,
        volume_data=shared_vol, volume_axes="ZYX",
    )
    # Pre-existing stale labels on sibling B (simulating an older import/session).
    doc_b.labels_3d = np.zeros(shape, dtype=np.uint16)
    doc_b.labels_3d[0, 0:3, 0:3] = 99
    doc_b.labels_meta = {99: {"name": "stale",
                              "color": (128, 128, 128),
                              "opacity": 128,
                              "visible": True,
                              "fill_mode": "solid"}}
    doc_b.labels_next_id = 100
    app.layer_stack.set_image(doc_a.image_layer)
    app.layer_stack.roi_layers.append(VolumeROILayer(doc_b, 99))
    app._documents = [doc_a, doc_b]
    app._active_doc_index = 0

    panel = View3DPanel(None,
                        channels=[("A", shared_vol, (1.0, 1.0, 1.0))],
                        documents=[doc_a, doc_b])
    try:
        app._view3d_panel = panel
        app.layer_panel.set_3d_mode(True)
        assert panel._primary_doc is doc_a

        tiff = _write_labels_tiff(tmp_path)
        with patch(
            "montaris.app.QFileDialog.getOpenFileName",
            return_value=(str(tiff), ""),
        ):
            app.import_volume_labels()

        # A has the fresh imports.
        assert doc_a.labels_3d is not None
        assert set(doc_a.labels_meta.keys()) == {1, 2}
        # B's stale labels must be cleared since it aliases A's volume_data.
        assert doc_b.labels_3d is None
        assert doc_b.labels_meta == {}
        # No ghost VolumeROILayer wrappers for B remain.
        b_wrappers = [r for r in app.layer_stack.roi_layers
                      if isinstance(r, VolumeROILayer) and r._doc is doc_b]
        assert not b_wrappers
    finally:
        app._view3d_panel = None
        panel.close()


@pytest.mark.skipif(not VISPY_AVAILABLE, reason="vispy not installed")
def test_import_labels_enables_wand_guard(app, qapp, tmp_path):
    """The exact guard in _on_canvas_mouse_press / wand must pass post-import."""
    vol = np.zeros((4, 16, 16), dtype=np.uint16)
    doc = _attach_volume_doc(app, vol)
    panel = View3DPanel(None, channels=[("ch", vol, (1.0, 1.0, 1.0))],
                        documents=[doc])
    try:
        app._view3d_panel = panel
        app.layer_panel.set_3d_mode(True)
        tiff = _write_labels_tiff(tmp_path)
        with patch(
            "montaris.app.QFileDialog.getOpenFileName",
            return_value=(str(tiff), ""),
        ):
            app.import_volume_labels()

        # Replicate the wand/paint guard literally.
        active = panel._active_volume_roi_id
        refused = active is None or active not in doc.labels_meta
        assert not refused, "wand/paint guard still refuses after import"
    finally:
        app._view3d_panel = None
        panel.close()
