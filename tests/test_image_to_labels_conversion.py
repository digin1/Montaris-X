"""Tests for converting the displayed image row into ROI label layers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from PySide6.QtWidgets import QApplication, QMessageBox

from montaris.layers import ImageLayer, MontageDocument, VolumeROILayer


def _set_image(app, data, name="segmentation"):
    app.layer_stack.set_image(ImageLayer(name, data))
    app.canvas.refresh_image()
    app.layer_panel.refresh()
    QApplication.processEvents()


def _attach_volume_doc(app, vol, name="stack"):
    mp = vol.max(axis=0)
    layer = ImageLayer(name, mp)
    app.layer_stack.set_image(layer)
    app.canvas.refresh_image()
    doc = MontageDocument(
        name=name,
        image_layer=layer,
        downsample_factor=1,
        original_shape=mp.shape,
        volume_data=vol,
        volume_axes="ZYX",
    )
    app._documents = [doc]
    app._active_doc_index = 0
    app.layer_panel.refresh()
    QApplication.processEvents()
    return doc


def test_convert_integer_image_creates_roi_per_nonzero_value(app):
    data = np.array(
        [
            [0, 1, 1, 0],
            [2, 2, 0, 0],
            [0, 2, 1, 0],
        ],
        dtype=np.uint16,
    )
    _set_image(app, data)

    with patch.object(app.toast, "show") as toast:
        app.convert_image_to_label_layers()

    assert [roi.name for roi in app.layer_stack.roi_layers] == [
        "Label 1",
        "Label 2",
    ]
    np.testing.assert_array_equal(
        app.layer_stack.roi_layers[0].mask > 0, data == 1
    )
    np.testing.assert_array_equal(
        app.layer_stack.roi_layers[1].mask > 0, data == 2
    )
    assert app.canvas._active_layer is app.layer_stack.roi_layers[0]
    toast.assert_called_once()


def test_convert_float_image_truncates_values_like_napari(app):
    data = np.array(
        [
            [0.2, 1.9, 2.1],
            [2.8, 0.0, 1.1],
        ],
        dtype=np.float32,
    )
    _set_image(app, data)

    with patch.object(app.toast, "show") as toast:
        app.convert_image_to_label_layers()

    assert [roi.name for roi in app.layer_stack.roi_layers] == [
        "Label 1",
        "Label 2",
    ]
    np.testing.assert_array_equal(
        app.layer_stack.roi_layers[0].mask > 0, data.astype(np.int64) == 1
    )
    np.testing.assert_array_equal(
        app.layer_stack.roi_layers[1].mask > 0, data.astype(np.int64) == 2
    )
    assert "truncated" in toast.call_args[0][0]


def test_convert_to_labels_rejects_rgb_images(app):
    data = np.zeros((8, 8, 3), dtype=np.uint8)
    _set_image(app, data, name="rgb")

    with patch("montaris.app.QMessageBox.information") as info:
        app.convert_image_to_label_layers()

    assert app.layer_stack.roi_layers == []
    info.assert_called_once()
    assert "single-channel 2D" in info.call_args[0][2]


def test_large_conversion_can_be_cancelled(app):
    data = np.arange(23 * 23, dtype=np.uint16).reshape(23, 23)
    _set_image(app, data, name="many-labels")

    with patch(
        "montaris.app.QMessageBox.warning",
        return_value=QMessageBox.No,
    ) as warning:
        app.convert_image_to_label_layers()

    assert app.layer_stack.roi_layers == []
    warning.assert_called_once()


def test_convert_volume_image_creates_3d_labels_when_3d_open(app):
    vol = np.zeros((3, 4, 5), dtype=np.uint16)
    vol[0, 0:2, 0:2] = 1
    vol[2, 1:4, 2:5] = 2
    doc = _attach_volume_doc(app, vol)
    panel = MagicMock()
    panel._primary_doc = doc
    app._view3d_panel = panel
    app.layer_panel.set_3d_mode(True)

    with patch.object(app.toast, "show") as toast:
        app.convert_image_to_label_layers()

    wrappers = [
        roi for roi in app.layer_stack.roi_layers
        if isinstance(roi, VolumeROILayer) and roi._doc is doc
    ]
    assert [roi.name for roi in wrappers] == ["Label 1", "Label 2"]
    assert set(doc.labels_meta.keys()) == {1, 2}
    np.testing.assert_array_equal(doc.labels_3d == 1, vol == 1)
    np.testing.assert_array_equal(doc.labels_3d == 2, vol == 2)
    assert app.canvas._active_layer is wrappers[0]
    panel.refresh_labels.assert_called()
    panel.set_active_volume_roi_id.assert_called_with(wrappers[0].label_id)
    toast.assert_called_once()


def test_convert_volume_targets_primary_doc_when_3d_open(app):
    vol_a = np.zeros((3, 4, 4), dtype=np.uint16)
    vol_a[1, 1:3, 1:3] = 7
    vol_b = np.zeros((3, 4, 4), dtype=np.uint16)
    vol_b[0, 0:2, 0:2] = 9

    doc_a = _attach_volume_doc(app, vol_a, name="A")
    layer_b = ImageLayer("B", vol_b.max(axis=0))
    doc_b = MontageDocument(
        name="B",
        image_layer=layer_b,
        downsample_factor=1,
        original_shape=layer_b.data.shape,
        volume_data=vol_b,
        volume_axes="ZYX",
    )
    app._documents = [doc_a, doc_b]
    app._active_doc_index = 1
    app.layer_stack.image_layer = layer_b
    panel = MagicMock()
    panel._primary_doc = doc_a
    app._view3d_panel = panel
    app.layer_panel.set_3d_mode(True)

    with patch.object(app.toast, "show"):
        app.convert_image_to_label_layers()

    a_wrappers = [
        roi for roi in app.layer_stack.roi_layers
        if isinstance(roi, VolumeROILayer) and roi._doc is doc_a
    ]
    b_wrappers = [
        roi for roi in app.layer_stack.roi_layers
        if isinstance(roi, VolumeROILayer) and roi._doc is doc_b
    ]
    assert [roi.name for roi in a_wrappers] == ["Label 7"]
    assert not b_wrappers
    assert doc_a.labels_3d is not None
    assert set(doc_a.labels_meta.keys()) == {1}
    assert doc_b.labels_3d is None
    assert doc_b.labels_meta == {}


def test_convert_volume_preserves_existing_3d_labels(app):
    vol = np.zeros((2, 4, 4), dtype=np.uint16)
    vol[0, 0:2, 0:2] = 5
    vol[1, 2:4, 2:4] = 9
    doc = _attach_volume_doc(app, vol)
    doc.ensure_labels_3d()
    existing = doc.reserve_label_id(name="existing", color=(1, 2, 3))
    doc.labels_3d[0, 0:2, 0:2] = existing
    app.layer_stack.roi_layers.append(VolumeROILayer(doc, existing))
    panel = MagicMock()
    panel._primary_doc = doc
    app._view3d_panel = panel
    app.layer_panel.set_3d_mode(True)

    with patch.object(app.toast, "show"):
        app.convert_image_to_label_layers()

    wrappers = [
        roi for roi in app.layer_stack.roi_layers
        if isinstance(roi, VolumeROILayer) and roi._doc is doc
    ]
    assert [roi.name for roi in wrappers] == ["existing", "Label 9"]
    assert doc.labels_3d[0, 0, 0] == existing
    assert (doc.labels_3d[1, 2:4, 2:4] == 2).all()


def test_convert_volume_promotes_dtype_for_high_label_ids(app):
    vol = np.array([[[1, 2, 3]]], dtype=np.uint16)
    doc = _attach_volume_doc(app, vol)
    doc.labels_next_id = 65535
    panel = MagicMock()
    panel._primary_doc = doc
    app._view3d_panel = panel
    app.layer_panel.set_3d_mode(True)

    with patch.object(app.toast, "show"):
        app.convert_image_to_label_layers()

    wrappers = [
        roi for roi in app.layer_stack.roi_layers
        if isinstance(roi, VolumeROILayer) and roi._doc is doc
    ]
    assert doc.labels_3d.dtype == np.uint32
    assert [roi.label_id for roi in wrappers] == [65535, 65536, 65537]
    np.testing.assert_array_equal(
        doc.labels_3d[0, 0], np.array([65535, 65536, 65537], dtype=np.uint32)
    )


def test_convert_volume_avoids_full_volume_delta_buffer(app, monkeypatch):
    vol = np.zeros((2, 4, 4), dtype=np.uint16)
    vol[0, 0:2, 0:2] = 1
    vol[1, 2:4, 2:4] = 2
    doc = _attach_volume_doc(app, vol)
    doc.ensure_labels_3d()
    panel = MagicMock()
    panel._primary_doc = doc
    app._view3d_panel = panel
    app.layer_panel.set_3d_mode(True)

    real_zeros = np.zeros

    def guarded_zeros(shape, *args, **kwargs):
        norm = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        if norm == vol.shape:
            raise AssertionError("unexpected full-volume scratch buffer")
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr("montaris.app.np.zeros", guarded_zeros)

    with patch.object(app.toast, "show"):
        app.convert_image_to_label_layers()



def test_convert_volume_rejects_shape_mismatch(app):
    """If ``doc.labels_3d`` is a stale shape (e.g. user swapped
    ``volume_data`` mid-session), the conversion must surface a clear
    error rather than broadcast-failing or silently scrambling indices.

    Mirrors the wand path's shape-match guard in view_3d.py.
    """
    vol = np.zeros((3, 4, 4), dtype=np.uint16)
    vol[1, 1:3, 1:3] = 1
    doc = _attach_volume_doc(app, vol)
    # Pre-allocate a labels_3d at the WRONG shape — different Z extent.
    doc.labels_3d = np.zeros((5, 4, 4), dtype=np.uint8)
    doc.labels_meta = {}
    panel = MagicMock()
    panel._primary_doc = doc
    app._view3d_panel = panel
    app.layer_panel.set_3d_mode(True)

    with patch.object(QMessageBox, "warning") as warn:
        app.convert_image_to_label_layers()

    warn.assert_called_once()
    args = warn.call_args.args
    assert "Shape mismatch" in args[1] or "shape" in args[2].lower(), (
        f"warning dialog should mention shape mismatch; got {args}"
    )
    # No ROI was created.
    assert not [
        roi for roi in app.layer_stack.roi_layers
        if isinstance(roi, VolumeROILayer)
    ]
    # Stale labels_3d shape is preserved (not overwritten).
    assert doc.labels_3d.shape == (5, 4, 4)


def test_convert_volume_single_z_slice(app):
    """Edge case: a depth-1 volume (one Z slice) — verifies the per-Z
    loop doesn't choke when ``labels.shape[0] == 1``."""
    vol = np.zeros((1, 4, 4), dtype=np.uint16)
    vol[0, 1:3, 1:3] = 5
    vol[0, 0, 3] = 9
    doc = _attach_volume_doc(app, vol)
    doc.ensure_labels_3d()
    panel = MagicMock()
    panel._primary_doc = doc
    app._view3d_panel = panel
    app.layer_panel.set_3d_mode(True)

    with patch.object(app.toast, "show"):
        app.convert_image_to_label_layers()

    wrappers = [
        roi for roi in app.layer_stack.roi_layers
        if isinstance(roi, VolumeROILayer) and roi._doc is doc
    ]
    assert {roi.name for roi in wrappers} == {"Label 5", "Label 9"}
    assert set(doc.labels_meta.keys()) == {1, 2}


def test_convert_volume_same_value_across_disconnected_z_slices(app):
    """Same source value appearing on multiple Z slices must collapse
    into ONE ROI id (one VolumeROILayer covering all matching voxels),
    not N — the conversion is "one ROI per unique value", not per
    connected component.
    """
    vol = np.zeros((4, 4, 4), dtype=np.uint16)
    vol[0, 0:2, 0:2] = 7   # value 7 on slice 0
    vol[3, 2:4, 2:4] = 7   # same value 7 on slice 3 (disconnected)
    doc = _attach_volume_doc(app, vol)
    doc.ensure_labels_3d()
    panel = MagicMock()
    panel._primary_doc = doc
    app._view3d_panel = panel
    app.layer_panel.set_3d_mode(True)

    with patch.object(app.toast, "show"):
        app.convert_image_to_label_layers()

    wrappers = [
        roi for roi in app.layer_stack.roi_layers
        if isinstance(roi, VolumeROILayer) and roi._doc is doc
    ]
    assert [roi.name for roi in wrappers] == ["Label 7"], (
        "value 7 across two disconnected slices must collapse to ONE ROI"
    )
    # Both slabs got the same id.
    lid = wrappers[0].label_id
    np.testing.assert_array_equal(doc.labels_3d == lid, vol == 7)


def test_convert_volume_handles_nan_inf_via_napari_coercion(app):
    """Float volumes with NaN / +Inf / -Inf must be coerced to 0 (napari
    rule) before unique-value extraction. ``_coerce_labels_source`` runs
    ``np.nan_to_num(...)`` so these become 0 and don't appear as ROIs.
    """
    vol = np.zeros((2, 3, 3), dtype=np.float32)
    vol[0, 0, 0] = 3.7              # truncates to 3
    vol[0, 1, 1] = np.nan           # → 0
    vol[1, 0, 1] = np.inf           # → 0
    vol[1, 1, 0] = -np.inf          # → 0
    vol[1, 2, 2] = 5.0              # → 5
    doc = _attach_volume_doc(app, vol)
    doc.ensure_labels_3d()
    panel = MagicMock()
    panel._primary_doc = doc
    app._view3d_panel = panel
    app.layer_panel.set_3d_mode(True)

    with patch.object(app.toast, "show") as toast:
        app.convert_image_to_label_layers()

    wrappers = [
        roi for roi in app.layer_stack.roi_layers
        if isinstance(roi, VolumeROILayer) and roi._doc is doc
    ]
    # Only 3 and 5 survived as labels (NaN/Inf collapsed to 0 → background).
    assert {roi.name for roi in wrappers} == {"Label 3", "Label 5"}
    # The toast must say "non-integer values truncated".
    msg = toast.call_args.args[0]
    assert "truncated" in msg


def test_convert_volume_undo_restores_meta_and_voxels(app):
    """Conversion's CompoundUndoCommand must, on undo, both remove the
    new VolumeROILayer wrappers AND zero out the voxels they wrote into
    ``labels_3d``. Redo must reapply both.
    """
    vol = np.zeros((2, 4, 4), dtype=np.uint16)
    vol[0, 1:3, 1:3] = 1
    vol[1, 2:4, 0:2] = 2
    doc = _attach_volume_doc(app, vol)
    doc.ensure_labels_3d()
    panel = MagicMock()
    panel._primary_doc = doc
    app._view3d_panel = panel
    app.layer_panel.set_3d_mode(True)

    with patch.object(app.toast, "show"):
        app.convert_image_to_label_layers()

    # After conversion: 2 ROIs, voxels written.
    wrappers = [
        roi for roi in app.layer_stack.roi_layers
        if isinstance(roi, VolumeROILayer) and roi._doc is doc
    ]
    assert len(wrappers) == 2
    assert int(doc.labels_3d.sum()) > 0
    pre_undo_label_ids = {w.label_id for w in wrappers}

    # Undo — ROIs must be gone AND voxels must be cleared.
    app.undo_stack.undo()
    remaining = [
        roi for roi in app.layer_stack.roi_layers
        if isinstance(roi, VolumeROILayer) and roi._doc is doc
    ]
    assert remaining == [], "undo must remove the converted VolumeROILayers"
    # All voxels must be zero again.
    assert int((doc.labels_3d != 0).sum()) == 0, (
        "undo must clear the voxels written by the conversion"
    )

    # Redo — restore both meta and voxels.
    app.undo_stack.redo()
    restored = [
        roi for roi in app.layer_stack.roi_layers
        if isinstance(roi, VolumeROILayer) and roi._doc is doc
    ]
    assert len(restored) == 2
    assert {w.label_id for w in restored} == pre_undo_label_ids
    assert int(doc.labels_3d.sum()) > 0


def test_convert_volume_handles_empty_z_slices(app):
    """Volumes where some Z slices have NO labels (the ``not plane_mask.any():
    continue`` branch) must still produce correct bboxes for the slices
    that do contain labels.
    """
    vol = np.zeros((5, 4, 4), dtype=np.uint16)
    # Label only on z=2; z=0,1,3,4 are empty.
    vol[2, 1:3, 1:3] = 4
    doc = _attach_volume_doc(app, vol)
    doc.ensure_labels_3d()
    panel = MagicMock()
    panel._primary_doc = doc
    app._view3d_panel = panel
    app.layer_panel.set_3d_mode(True)

    with patch.object(app.toast, "show"):
        app.convert_image_to_label_layers()

    wrappers = [
        roi for roi in app.layer_stack.roi_layers
        if isinstance(roi, VolumeROILayer) and roi._doc is doc
    ]
    assert [roi.name for roi in wrappers] == ["Label 4"]
    # Voxels written only on z=2.
    np.testing.assert_array_equal(doc.labels_3d == wrappers[0].label_id,
                                   vol == 4)
    # Volume bbox is restricted to z=2 (not the full 0..5).
    bbox = wrappers[0].get_volume_bbox()
    assert bbox is not None
    z1, z2, _, _, _, _ = bbox
    assert z1 == 2 and z2 == 3
