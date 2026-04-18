"""Phase 5: Paint/Erase brush tests for View3DPanel.

Headless smoke tests that exercise ``_stamp_brush`` geometry and the
drag-paint / drag-erase flow without requiring a real GL context. Construct
a View3DPanel with a synthetic channel and primary document, then drive the
drag state machine directly (skipping the ray-pick which needs GL).
"""
from __future__ import annotations

import numpy as np
import pytest

from PySide6.QtWidgets import QApplication, QMessageBox

from montaris.layers import MontageDocument, ImageLayer
from montaris.widgets.view_3d import View3DPanel, VISPY_AVAILABLE


pytestmark = pytest.mark.skipif(not VISPY_AVAILABLE, reason="vispy not installed")


def _make_doc_with_volume(shape=(10, 24, 28)):
    z, y, x = shape
    vol = np.zeros(shape, dtype=np.uint8)
    # Simple intensity ramp along z — gives the panel a valid channel to
    # show while we poke at labels_3d directly.
    for i in range(z):
        vol[i] = (i + 1) * 10
    mp = vol.max(axis=0)
    doc = MontageDocument(name="paintdoc", image_layer=ImageLayer("paintdoc", mp))
    doc.volume_data = vol
    doc.volume_axes = "ZYX"
    doc.active_z = 0
    doc.ensure_labels_3d()
    return doc, vol


def _make_panel(qapp, doc, vol):
    panel = View3DPanel(None, channels=[("ch0", vol, (1.0, 1.0, 1.0))], documents=[doc])
    # Don't show()/render() — ray-pick requires GL. We drive _stamp_brush
    # with a synthetic seed instead.
    return panel


def test_stamp_brush_writes_sphere(qapp):
    """A single stamp at the volume's center fills a spherical region only."""
    doc, vol = _make_doc_with_volume((20, 40, 40))
    panel = _make_panel(qapp, doc, vol)
    try:
        # Simulate a paint-stroke setup: reserved label, paint mode.
        panel._drag_active = True
        panel._drag_mode = 'paint'
        panel._drag_label_id = doc.reserve_label_id()
        panel._brush_radius = 4

        panel._stamp_brush((10, 20, 20))

        # Painted voxels: exactly those inside radius 4 of (10, 20, 20).
        painted = (doc.labels_3d == panel._drag_label_id)
        z, y, x = np.indices(doc.labels_3d.shape)
        expected = ((z - 10) ** 2 + (y - 20) ** 2 + (x - 20) ** 2) <= 16
        np.testing.assert_array_equal(painted, expected)
        # Nothing else should be labeled.
        assert doc.labels_3d[~painted].sum() == 0
        # Overlay dirty flag set so the throttle timer will rebuild.
        assert panel._drag_dirty is True
    finally:
        panel.close()


def test_stamp_brush_clips_to_volume_bounds(qapp):
    """Stamping near an edge doesn't write out of bounds or wrap around."""
    doc, vol = _make_doc_with_volume((10, 20, 20))
    panel = _make_panel(qapp, doc, vol)
    try:
        panel._drag_active = True
        panel._drag_mode = 'paint'
        panel._drag_label_id = doc.reserve_label_id()
        panel._brush_radius = 5

        # Seed at a corner — much of the sphere falls outside.
        panel._stamp_brush((0, 0, 0))

        # Label must be present near the corner but zero elsewhere.
        assert doc.labels_3d[0, 0, 0] == panel._drag_label_id
        # Voxel at (0, 6, 0) is outside radius 5 → still 0.
        assert doc.labels_3d[0, 6, 0] == 0
        # Volume shape was preserved (no accidental resize).
        assert doc.labels_3d.shape == (10, 20, 20)
    finally:
        panel.close()


def test_erase_stamp_writes_zero_over_existing_label(qapp):
    """Erase mode zeros voxels inside the sphere regardless of owner."""
    doc, vol = _make_doc_with_volume((12, 24, 24))
    panel = _make_panel(qapp, doc, vol)
    try:
        # Pre-populate a slab with label_id 1 so we have something to erase.
        lid_a = doc.reserve_label_id()
        doc.labels_3d[5:10, 10:20, 10:20] = lid_a

        panel._drag_active = True
        panel._drag_mode = 'erase'
        panel._drag_label_id = 0
        panel._brush_radius = 3

        panel._stamp_brush((7, 15, 15))

        # Center of the sphere is now 0, outside the sphere still holds lid_a.
        assert doc.labels_3d[7, 15, 15] == 0
        # A voxel well outside radius 3 is untouched.
        assert doc.labels_3d[5, 10, 10] == lid_a
    finally:
        panel.close()


def test_paint_drag_release_emits_label_added(qapp):
    """Drag-paint lifecycle: press stamps, release emits label_added."""
    doc, vol = _make_doc_with_volume((10, 20, 20))
    panel = _make_panel(qapp, doc, vol)
    try:
        received = []
        panel.label_added.connect(lambda d, lid: received.append((d, lid)))

        panel._drag_active = True
        panel._drag_mode = 'paint'
        panel._drag_label_id = doc.reserve_label_id()
        panel._brush_radius = 2
        panel._stamp_brush((5, 10, 10))

        panel._finish_drag(emit=True)

        assert received == [(doc, panel_captured_lid := received[0][1])]
        # After finish, drag state is cleared.
        assert panel._drag_active is False
        assert panel._drag_label_id is None
        assert panel._drag_mode is None
        # The stamped voxels remain in labels_3d under the old id.
        assert (doc.labels_3d == panel_captured_lid).sum() > 0
    finally:
        panel.close()


def test_erase_finish_does_not_emit_label_added(qapp):
    """Erase strokes shouldn't spawn a new VolumeROILayer on the panel."""
    doc, vol = _make_doc_with_volume((10, 20, 20))
    panel = _make_panel(qapp, doc, vol)
    try:
        received = []
        panel.label_added.connect(lambda d, lid: received.append((d, lid)))

        panel._drag_active = True
        panel._drag_mode = 'erase'
        panel._drag_label_id = 0
        panel._brush_radius = 2
        panel._stamp_brush((5, 10, 10))
        panel._finish_drag(emit=False)

        assert received == []
    finally:
        panel.close()


def test_tool_combo_has_paint_and_erase(qapp):
    """UI contract: the tool dropdown exposes all four modes in order."""
    doc, vol = _make_doc_with_volume((4, 8, 8))
    panel = _make_panel(qapp, doc, vol)
    try:
        items = [panel._tool_combo.itemText(i)
                 for i in range(panel._tool_combo.count())]
        assert items == ["Navigate", "Fill", "Paint", "Erase"]
    finally:
        panel.close()


def test_brush_spin_enabled_only_in_paint_or_erase(qapp):
    """Brush radius control gates on/off with the tool mode."""
    doc, vol = _make_doc_with_volume((4, 8, 8))
    panel = _make_panel(qapp, doc, vol)
    try:
        # Navigate → disabled
        panel._tool_combo.setCurrentText("Navigate")
        QApplication.processEvents()
        assert panel._brush_spin.isEnabled() is False

        # Fill → disabled (napari-style Fill is label-based, no tolerance)
        panel._tool_combo.setCurrentText("Fill")
        QApplication.processEvents()
        assert panel._brush_spin.isEnabled() is False
        assert panel._tol_slider.isEnabled() is False

        # Paint → enabled
        panel._tool_combo.setCurrentText("Paint")
        QApplication.processEvents()
        assert panel._brush_spin.isEnabled() is True
        assert panel._tol_slider.isEnabled() is False

        # Erase → enabled
        panel._tool_combo.setCurrentText("Erase")
        QApplication.processEvents()
        assert panel._brush_spin.isEnabled() is True
    finally:
        panel.close()


def test_tool_switch_midstroke_aborts_without_emitting(qapp):
    """Switching the tool dropdown during a drag rolls back the fresh id."""
    doc, vol = _make_doc_with_volume((10, 20, 20))
    panel = _make_panel(qapp, doc, vol)
    try:
        received = []
        panel.label_added.connect(lambda d, lid: received.append((d, lid)))

        panel._tool_combo.setCurrentText("Paint")
        panel._drag_active = True
        panel._drag_mode = 'paint'
        lid = doc.reserve_label_id()
        panel._drag_label_id = lid
        panel._drag_extends_existing = False
        panel._brush_radius = 2
        panel._stamp_brush((5, 10, 10))

        # User flips to Navigate mid-stroke — tool change should rollback.
        panel._tool_combo.setCurrentText("Navigate")
        QApplication.processEvents()

        assert panel._drag_active is False
        assert received == []
        # Rollback drops the meta entry and zeros the voxels so no orphan
        # appears in the LayerPanel after the abort.
        assert lid not in doc.labels_meta
        assert (doc.labels_3d == lid).sum() == 0
    finally:
        panel.close()


def test_paint_extends_selected_volume_roi(qapp):
    """Paint with an active id extends that ROI — no new reservation."""
    doc, vol = _make_doc_with_volume((10, 20, 20))
    panel = _make_panel(qapp, doc, vol)
    try:
        # Pre-reserve an id as if the user had painted once already and
        # the LayerPanel selected its VolumeROILayer.
        existing = doc.reserve_label_id(name="my_dendrite")
        doc.labels_3d[5, 10, 10] = existing
        panel.set_active_volume_roi_id(existing)

        received = []
        panel.label_added.connect(lambda d, lid: received.append((d, lid)))

        panel._tool_combo.setCurrentText("Paint")
        # Simulate the press-stamp-release lifecycle.
        panel._drag_active = True
        panel._drag_mode = 'paint'
        panel._drag_extends_existing = True
        panel._drag_label_id = existing
        panel._brush_radius = 2
        panel._stamp_brush((6, 12, 12))
        panel._finish_drag(emit=False)

        # No new id reserved → label count unchanged.
        assert len(doc.labels_meta) == 1
        assert existing in doc.labels_meta
        # Extending an existing ROI must not emit label_added (would
        # create a duplicate VolumeROILayer on MontarisApp's side).
        assert received == []
        # The stamp actually wrote with the existing id.
        assert (doc.labels_3d == existing).sum() > 1
    finally:
        panel.close()


def test_paint_without_selection_reserves_new_id(qapp):
    """With no selected 3D ROI, paint stroke allocates a fresh id."""
    doc, vol = _make_doc_with_volume((10, 20, 20))
    panel = _make_panel(qapp, doc, vol)
    try:
        panel.set_active_volume_roi_id(None)
        pre_count = len(doc.labels_meta)

        received = []
        panel.label_added.connect(lambda d, lid: received.append((d, lid)))

        panel._tool_combo.setCurrentText("Paint")
        panel._drag_active = True
        panel._drag_mode = 'paint'
        panel._drag_extends_existing = False
        new_lid = doc.reserve_label_id()
        panel._drag_label_id = new_lid
        panel._brush_radius = 2
        panel._stamp_brush((5, 10, 10))
        panel._finish_drag(emit=True)

        assert len(doc.labels_meta) == pre_count + 1
        assert received == [(doc, new_lid)]
    finally:
        panel.close()


def test_tool_switch_midstroke_preserves_extended_id(qapp):
    """Aborting an extension stroke must NOT release the pre-existing ROI."""
    doc, vol = _make_doc_with_volume((10, 20, 20))
    panel = _make_panel(qapp, doc, vol)
    try:
        # Pre-existing ROI with a painted voxel the user wants to keep.
        existing = doc.reserve_label_id()
        doc.labels_3d[5, 10, 10] = existing
        panel.set_active_volume_roi_id(existing)

        panel._tool_combo.setCurrentText("Paint")
        panel._drag_active = True
        panel._drag_mode = 'paint'
        panel._drag_extends_existing = True
        panel._drag_label_id = existing
        panel._brush_radius = 1
        panel._stamp_brush((3, 3, 3))

        panel._tool_combo.setCurrentText("Navigate")
        QApplication.processEvents()

        # The id is still alive — the extension's partial stamps stay in
        # place (consistent with fire-and-forget paint semantics).
        assert existing in doc.labels_meta
        assert doc.labels_3d[5, 10, 10] == existing
    finally:
        panel.close()


def test_fill_on_background_no_selection_guides_user(qapp, monkeypatch):
    """napari Fill can't segment raw voxels — without a selected ROI on an
    unpainted voxel, inform the user instead of flooding background."""
    doc, _vol = _make_doc_with_volume((6, 12, 12))
    panel = _make_panel(qapp, doc, _vol)
    try:
        panel.set_active_volume_roi_id(None)
        infos = []
        monkeypatch.setattr(
            QMessageBox, "information",
            staticmethod(lambda *a, **k: infos.append(a)),
        )
        panel._tool_combo.setCurrentText("Fill")
        panel._run_fill((3, 6, 6))
        assert infos, "expected guidance popup"
        # No ROI reserved.
        assert len(doc.labels_meta) == 0
    finally:
        panel.close()


def test_fill_relabels_connected_component_to_selected_roi(qapp):
    """Click Fill on painted ROI A with B selected → A's connected region
    becomes B. Matches napari's selected_label-driven Fill."""
    doc, vol = _make_doc_with_volume((6, 12, 12))
    panel = _make_panel(qapp, doc, vol)
    try:
        lid_a = doc.reserve_label_id(name="a")
        lid_b = doc.reserve_label_id(name="b")
        doc.labels_3d[2:5, 4:8, 4:8] = lid_a
        panel.set_active_volume_roi_id(lid_b)

        received = []
        panel.label_added.connect(lambda d, lid: received.append((d, lid)))

        panel._tool_combo.setCurrentText("Fill")
        panel._run_fill((3, 6, 6))

        assert (doc.labels_3d == lid_a).sum() == 0
        assert (doc.labels_3d == lid_b).sum() > 0
        # Extension — no new VolumeROILayer signal.
        assert received == []
    finally:
        panel.close()


def test_fill_on_painted_region_without_selection_assigns_new_id(qapp):
    """No selection + click on painted region → allocate fresh id and
    relabel the connected component to it. label_added fires."""
    doc, vol = _make_doc_with_volume((6, 12, 12))
    panel = _make_panel(qapp, doc, vol)
    try:
        lid_a = doc.reserve_label_id(name="a")
        doc.labels_3d[2:5, 4:8, 4:8] = lid_a
        panel.set_active_volume_roi_id(None)

        received = []
        panel.label_added.connect(lambda d, lid: received.append((d, lid)))

        panel._tool_combo.setCurrentText("Fill")
        panel._run_fill((3, 6, 6))

        assert (doc.labels_3d == lid_a).sum() == 0
        assert len(received) == 1
        new_lid = received[0][1]
        assert (doc.labels_3d == new_lid).sum() > 0
    finally:
        panel.close()


def test_fill_on_own_label_is_noop(qapp):
    """napari early-out: clicking Fill on a voxel already of the target id
    leaves labels_3d unchanged and emits nothing."""
    doc, vol = _make_doc_with_volume((6, 12, 12))
    panel = _make_panel(qapp, doc, vol)
    try:
        lid = doc.reserve_label_id()
        doc.labels_3d[2:5, 4:8, 4:8] = lid
        panel.set_active_volume_roi_id(lid)
        before = doc.labels_3d.copy()

        received = []
        panel.label_added.connect(lambda d, l: received.append((d, l)))

        panel._tool_combo.setCurrentText("Fill")
        panel._run_fill((3, 6, 6))

        np.testing.assert_array_equal(doc.labels_3d, before)
        assert received == []
    finally:
        panel.close()


def test_fill_only_affects_connected_component(qapp):
    """Two disjoint blobs of label A + click on blob 1 with B selected →
    only blob 1 flips to B. Blob 2 stays A. (Contiguous semantics.)"""
    doc, vol = _make_doc_with_volume((6, 12, 20))
    panel = _make_panel(qapp, doc, vol)
    try:
        lid_a = doc.reserve_label_id(name="a")
        lid_b = doc.reserve_label_id(name="b")
        doc.labels_3d[1:3, 2:5, 2:5] = lid_a      # blob 1
        doc.labels_3d[3:5, 7:10, 14:18] = lid_a   # blob 2 — disjoint
        panel.set_active_volume_roi_id(lid_b)

        panel._tool_combo.setCurrentText("Fill")
        panel._run_fill((2, 3, 3))  # seed inside blob 1

        assert doc.labels_3d[2, 3, 3] == lid_b
        # Blob 2 untouched.
        assert doc.labels_3d[4, 8, 16] == lid_a
    finally:
        panel.close()


def test_set_active_volume_roi_id_accepts_none(qapp):
    """None/0 clears the active id so subsequent strokes allocate fresh."""
    doc, vol = _make_doc_with_volume((4, 8, 8))
    panel = _make_panel(qapp, doc, vol)
    try:
        panel.set_active_volume_roi_id(7)
        assert panel._active_volume_roi_id == 7
        panel.set_active_volume_roi_id(None)
        assert panel._active_volume_roi_id is None
        panel.set_active_volume_roi_id(0)
        assert panel._active_volume_roi_id is None
    finally:
        panel.close()
