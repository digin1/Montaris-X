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
