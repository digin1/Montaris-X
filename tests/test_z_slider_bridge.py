"""Phase 4: 2D bridge — Z slider surfaces labels_3d cross-sections.

Headless smoke tests for the Z-slider wiring added to MontarisApp. Builds a
synthetic 3D document with a labels volume that occupies different regions on
different Z-slices, then asserts that moving the slider:

  1. Updates ``doc.active_z``.
  2. Causes ``VolumeROILayer.mask`` to return the new slice.
  3. Invalidates the cached bbox so the 2D compositor redraws correctly.
"""
from __future__ import annotations

import numpy as np
import pytest

from PySide6.QtWidgets import QApplication

from montaris.layers import ImageLayer, VolumeROILayer


def _attach_volume_doc(app, shape=(5, 32, 40)):
    """Create a doc backed by a synthetic z-stack and push it through the app."""
    z, y, x = shape
    vol = np.zeros(shape, dtype=np.uint8)
    # Give each z-slice a distinctive intensity pattern so max-projection is
    # well-defined and the image layer shape matches volume YX.
    for i in range(z):
        vol[i] = (i + 1) * 20
    mp = vol.max(axis=0)

    # Go through the normal single-channel image loader so the doc/image layer
    # plumbing matches what real loads produce, then attach the volume.
    app.layer_stack.set_image(ImageLayer("zstack", mp))
    doc = app._documents[app._active_doc_index] if app._documents else None
    if doc is None:
        # _load_single_channel is the real path; simulate minimally if tests
        # skip it. Use what already exists: the first (and only) document.
        from montaris.layers import MontageDocument
        doc = MontageDocument(name="zstack", image_layer=app.layer_stack.image_layer)
        app._documents = [doc]
        app._active_doc_index = 0

    doc.volume_data = vol
    doc.volume_axes = "ZYX"
    doc.active_z = 0

    # Now allocate labels_3d and paint a single label that only appears on z=2.
    doc.ensure_labels_3d()
    label_id = doc.reserve_label_id(name="3D ROI test", color=(255, 0, 0))
    doc.labels_3d[2, 5:15, 5:15] = label_id
    # And a disjoint region on z=4 so slice changes produce visible diffs.
    doc.labels_3d[4, 20:25, 20:25] = label_id

    wrapper = VolumeROILayer(doc, label_id)
    app.layer_stack.roi_layers.append(wrapper)
    app.layer_stack.changed.emit()
    app._update_z_slider_visibility()
    QApplication.processEvents()
    return doc, wrapper, label_id


def test_z_slider_visible_only_for_zstack(app):
    """With no volume_data attached the Z bar stays hidden."""
    assert app._z_bar.isVisible() is False


def test_z_slider_shown_and_ranged_on_volume_attach(app):
    doc, wrapper, _ = _attach_volume_doc(app, shape=(7, 20, 30))
    assert app._z_bar.isVisible() is True
    assert app._z_slider.minimum() == 0
    assert app._z_slider.maximum() == 6  # n - 1


def test_z_slider_drives_volume_roi_mask_slice(app):
    """Moving the slider swaps which slice the VolumeROILayer exposes."""
    doc, wrapper, label_id = _attach_volume_doc(app, shape=(5, 32, 40))

    # z=0: wrapper.mask should be empty (no labels on that slice).
    app._z_slider.setValue(0)
    QApplication.processEvents()
    assert doc.active_z == 0
    assert wrapper.mask.sum() == 0

    # z=2: the 10x10 patch should light up.
    app._z_slider.setValue(2)
    QApplication.processEvents()
    assert doc.active_z == 2
    assert wrapper.mask.sum() == 100  # 10 * 10

    # z=4: the 5x5 patch.
    app._z_slider.setValue(4)
    QApplication.processEvents()
    assert doc.active_z == 4
    assert wrapper.mask.sum() == 25  # 5 * 5


def test_z_slider_invalidates_cached_bbox(app):
    """If bbox were cached, a slice change would return stale coords."""
    doc, wrapper, _ = _attach_volume_doc(app, shape=(5, 32, 40))

    # Prime the cache on z=2 — bbox = (y1=5, y2=15, x1=5, x2=15).
    app._z_slider.setValue(2)
    QApplication.processEvents()
    bbox_z2 = wrapper.get_bbox()
    assert bbox_z2 == (5, 15, 5, 15)

    # Move to z=4 and confirm the bbox reflects the new patch, not the cached one.
    app._z_slider.setValue(4)
    QApplication.processEvents()
    bbox_z4 = wrapper.get_bbox()
    assert bbox_z4 == (20, 25, 20, 25), f"bbox cache was not invalidated: {bbox_z4}"


def test_z_slider_label_text(app):
    """The trailing label reads '<current>/<total>' 1-indexed for humans."""
    doc, _, _ = _attach_volume_doc(app, shape=(10, 16, 16))
    app._z_slider.setValue(0)
    QApplication.processEvents()
    assert app._z_label.text() == "1 / 10"
    app._z_slider.setValue(9)
    QApplication.processEvents()
    assert app._z_label.text() == "10 / 10"
