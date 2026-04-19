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
        assert items == ["Navigate", "Fill", "Wand", "Paint", "Erase"]
    finally:
        panel.close()


def test_brush_spin_enabled_only_in_paint_or_erase(qapp):
    """Brush + Wand controls gate on/off with the tool mode."""
    doc, vol = _make_doc_with_volume((4, 8, 8))
    panel = _make_panel(qapp, doc, vol)
    try:
        # Navigate → everything off
        panel._tool_combo.setCurrentText("Navigate")
        QApplication.processEvents()
        assert panel._brush_spin.isEnabled() is False
        assert panel._tol_slider.isEnabled() is False
        assert panel._fill_channel_combo.isEnabled() is False

        # Fill → everything off (napari-style Fill is label-based, no tolerance)
        panel._tool_combo.setCurrentText("Fill")
        QApplication.processEvents()
        assert panel._brush_spin.isEnabled() is False
        assert panel._tol_slider.isEnabled() is False
        assert panel._fill_channel_combo.isEnabled() is False

        # Wand → Channel + Tolerance on, Brush off
        panel._tool_combo.setCurrentText("Wand")
        QApplication.processEvents()
        assert panel._brush_spin.isEnabled() is False
        assert panel._tol_slider.isEnabled() is True
        assert panel._fill_channel_combo.isEnabled() is True

        # Paint → Brush on, Wand off
        panel._tool_combo.setCurrentText("Paint")
        QApplication.processEvents()
        assert panel._brush_spin.isEnabled() is True
        assert panel._tol_slider.isEnabled() is False
        assert panel._fill_channel_combo.isEnabled() is False

        # Erase → Brush on
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


def test_wand_refuses_without_selected_roi(qapp, monkeypatch):
    """Wand requires a selected 3D ROI — no auto-reserve."""
    doc, vol = _make_doc_with_volume((6, 12, 12))
    panel = _make_panel(qapp, doc, vol)
    try:
        panel.set_active_volume_roi_id(None)
        infos = []
        monkeypatch.setattr(
            QMessageBox, "information",
            staticmethod(lambda *a, **k: infos.append(a)),
        )
        panel._tool_combo.setCurrentText("Wand")
        panel._run_wand((3, 6, 6))
        assert infos, "expected guidance popup when nothing is selected"
        assert len(doc.labels_meta) == 0
    finally:
        panel.close()


def test_wand_flood_fills_by_intensity_into_selected_roi(qapp):
    """Wand flood-fills voxels within tolerance of the seed intensity,
    writing the selected ROI's label id into currently-empty voxels."""
    # Build a volume with a bright dendrite-like region surrounded by dark.
    shape = (8, 16, 16)
    vol = np.zeros(shape, dtype=np.uint8)
    vol[3:6, 5:10, 5:10] = 200  # "dendrite" blob
    vol[3:6, 5:10, 10:14] = 205  # continuation, within tolerance
    # A disconnected bright blob — wand should NOT reach it.
    vol[0:2, 0:3, 0:3] = 210
    mp = vol.max(axis=0)
    doc = MontageDocument(name="wand", image_layer=ImageLayer("w", mp))
    doc.volume_data = vol
    doc.volume_axes = "ZYX"
    doc.ensure_labels_3d()

    panel = View3DPanel(
        None, channels=[("ch0", vol, (1.0, 1.0, 1.0))], documents=[doc],
    )
    try:
        lid = doc.reserve_label_id(name="dendrite")
        panel.set_active_volume_roi_id(lid)
        panel._tool_combo.setCurrentText("Wand")
        panel._fill_tolerance = 20  # 200±20 covers 200 and 205

        panel._run_wand((4, 7, 7))

        # Both connected bright slabs got labeled.
        assert doc.labels_3d[4, 7, 7] == lid
        assert doc.labels_3d[4, 7, 12] == lid
        # Disconnected blob wasn't reached.
        assert doc.labels_3d[1, 1, 1] == 0
        # Background stayed 0.
        assert doc.labels_3d[0, 15, 15] == 0
    finally:
        panel.close()


def test_wand_does_not_clobber_other_rois(qapp):
    """Wand writes only into background (0) voxels — existing ROIs survive."""
    shape = (6, 12, 12)
    vol = np.zeros(shape, dtype=np.uint8)
    vol[2:5, 3:9, 3:9] = 180  # uniform bright region
    doc = MontageDocument(
        name="w", image_layer=ImageLayer("w", vol.max(axis=0)),
    )
    doc.volume_data = vol
    doc.volume_axes = "ZYX"
    doc.ensure_labels_3d()

    panel = View3DPanel(
        None, channels=[("ch0", vol, (1.0, 1.0, 1.0))], documents=[doc],
    )
    try:
        lid_other = doc.reserve_label_id(name="other")
        lid_target = doc.reserve_label_id(name="target")
        doc.labels_3d[3, 5, 5] = lid_other  # one voxel already owned
        panel.set_active_volume_roi_id(lid_target)

        panel._tool_combo.setCurrentText("Wand")
        panel._fill_tolerance = 5
        panel._run_wand((3, 4, 4))

        # Target ROI got voxels in the bright region...
        assert (doc.labels_3d == lid_target).sum() > 0
        # ...but the other ROI's claimed voxel was preserved.
        assert doc.labels_3d[3, 5, 5] == lid_other
    finally:
        panel.close()


def test_wand_aborts_when_tolerance_floods_half_the_volume(qapp, monkeypatch):
    """Runaway guard: tolerance so high that >= 50% of voxels flood gets refused."""
    shape = (6, 10, 10)
    vol = np.full(shape, 100, dtype=np.uint8)  # uniform → any seed floods everything
    doc = MontageDocument(
        name="w", image_layer=ImageLayer("w", vol.max(axis=0)),
    )
    doc.volume_data = vol
    doc.volume_axes = "ZYX"
    doc.ensure_labels_3d()

    panel = View3DPanel(
        None, channels=[("ch0", vol, (1.0, 1.0, 1.0))], documents=[doc],
    )
    try:
        lid = doc.reserve_label_id()
        panel.set_active_volume_roi_id(lid)
        warnings = []
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda *a, **k: warnings.append(a)),
        )
        panel._tool_combo.setCurrentText("Wand")
        panel._fill_tolerance = 100

        panel._run_wand((3, 5, 5))

        assert warnings, "expected runaway-guard warning"
        # No voxels were written.
        assert (doc.labels_3d == lid).sum() == 0
    finally:
        panel.close()


def test_tolerance_slider_updates_label_and_value(qapp):
    """Moving the slider changes _fill_tolerance and the numeric readout."""
    doc, vol = _make_doc_with_volume((4, 8, 8))
    panel = _make_panel(qapp, doc, vol)
    try:
        panel._tool_combo.setCurrentText("Wand")
        QApplication.processEvents()
        panel._tol_slider.setValue(47)
        QApplication.processEvents()
        assert panel._fill_tolerance == 47
        assert panel._tol_label.text() == "47"
        panel._tol_slider.setValue(0)
        QApplication.processEvents()
        assert panel._fill_tolerance == 0
        assert panel._tol_label.text() == "0"
    finally:
        panel.close()


def test_channel_combo_selection_updates_active_channel(qapp):
    """Changing Wand Channel flips which volume the Wand reads intensity from."""
    # Two channels with different values at the same seed — picking channel
    # 1 vs channel 0 should change what _fill_tolerance=0 can reach.
    shape = (4, 8, 8)
    ch0 = np.full(shape, 50, dtype=np.uint8)
    ch1 = np.full(shape, 200, dtype=np.uint8)
    ch0[2, 4, 4] = 51  # tiny deviation — within tol 0 only for ch1 path
    ch1[2, 4, 4] = 200
    mp = ch0.max(axis=0)
    doc = MontageDocument(name="c", image_layer=ImageLayer("c", mp))
    doc.volume_data = ch0
    doc.volume_axes = "ZYX"
    doc.ensure_labels_3d()
    panel = View3DPanel(
        None,
        channels=[("c0", ch0, (1.0, 1.0, 1.0)), ("c1", ch1, (1.0, 0.0, 0.0))],
        documents=[doc],
    )
    try:
        panel._tool_combo.setCurrentText("Wand")
        panel._fill_channel_combo.setCurrentIndex(0)
        QApplication.processEvents()
        assert panel._fill_channel_idx == 0
        assert panel._active_channel_volume() is ch0

        panel._fill_channel_combo.setCurrentIndex(1)
        QApplication.processEvents()
        assert panel._fill_channel_idx == 1
        assert panel._active_channel_volume() is ch1
    finally:
        panel.close()


def test_tolerance_change_affects_wand_flood_size(qapp):
    """Tightening tolerance shrinks the wand's reach; widening grows it.

    Wand uses asymmetric flood — accepts voxels ``>= seed - tol``. Build a
    graded slab that descends from the seed value so each tolerance step
    pulls in additional columns. Seed at x=5 (val=130). tol=5 catches
    ≥125 (2 cols), tol=30 catches ≥100 (all 7 cols).
    """
    shape = (4, 8, 20)
    vol = np.zeros(shape, dtype=np.uint8)
    # Graded slab along x: 130, 125, 120, 115, 110, 105, 100 (seed is the
    # bright end; flood descends toward dimmer voxels).
    for i, v in enumerate([130, 125, 120, 115, 110, 105, 100]):
        vol[1:3, 2:6, 5 + i] = v
    doc = MontageDocument(
        name="g", image_layer=ImageLayer("g", vol.max(axis=0)),
    )
    doc.volume_data = vol
    doc.volume_axes = "ZYX"
    doc.ensure_labels_3d()
    panel = View3DPanel(
        None, channels=[("c0", vol, (1.0, 1.0, 1.0))], documents=[doc],
    )
    try:
        lid = doc.reserve_label_id()
        panel.set_active_volume_roi_id(lid)
        panel._tool_combo.setCurrentText("Wand")

        # Tight tolerance via the slider: 130 − 5 = 125 → 2 columns.
        panel._tol_slider.setValue(5)
        QApplication.processEvents()
        panel._run_wand((2, 3, 5))  # seed at x=5 (value 130)
        tight_count = int((doc.labels_3d == lid).sum())

        # Reset and widen: 130 − 30 = 100 → all 7 columns.
        doc.labels_3d[...] = 0
        panel._tol_slider.setValue(30)
        QApplication.processEvents()
        panel._run_wand((2, 3, 5))
        wide_count = int((doc.labels_3d == lid).sum())

        assert tight_count > 0, "tight tolerance should still fill at least the seed"
        assert wide_count > tight_count, (
            f"widening tolerance must expand the flood "
            f"(tight={tight_count}, wide={wide_count})"
        )
    finally:
        panel.close()


def test_wand_asymmetric_reaches_dim_boundary(qapp):
    """Wand's asymmetric tolerance grows toward dimmer voxels.

    Build a bright core surrounded by a dim boundary shell. Seed on the
    core (255); the boundary (80) is 175 units dimmer. A properly
    asymmetric flood with tol=175 must include both, even though the
    boundary is far outside a symmetric ±tol window that clamps at 255.
    """
    shape = (6, 14, 14)
    vol = np.zeros(shape, dtype=np.uint8)
    # 3×3×3 dim shell wrapping the bright 1×1×1 core.
    vol[2:5, 5:10, 5:10] = 80    # shell
    vol[3, 7, 7] = 255            # bright core
    doc = MontageDocument(
        name="a", image_layer=ImageLayer("a", vol.max(axis=0)),
    )
    doc.volume_data = vol
    doc.volume_axes = "ZYX"
    doc.ensure_labels_3d()
    panel = View3DPanel(
        None, channels=[("c0", vol, (1.0, 1.0, 1.0))], documents=[doc],
    )
    try:
        lid = doc.reserve_label_id()
        panel.set_active_volume_roi_id(lid)
        panel._tool_combo.setCurrentText("Wand")
        # Seed on the bright core; tol = 175 → accepts val >= 255 - 175 = 80.
        panel._tol_slider.setValue(175)
        QApplication.processEvents()
        panel._run_wand((3, 7, 7))

        # Every shell voxel got claimed.
        shell_mask = (vol == 80)
        shell_filled = int(((doc.labels_3d == lid) & shell_mask).sum())
        assert shell_filled == int(shell_mask.sum()), (
            f"asymmetric flood must reach the dim shell: "
            f"{shell_filled}/{int(shell_mask.sum())}"
        )
        # Nothing outside shell + core got labeled (background stays 0).
        assert (doc.labels_3d[vol == 0] == 0).all()
    finally:
        panel.close()


def test_keyboard_shortcuts_switch_tool(qapp):
    """Tool-switch shortcuts (V/F/W/P/E) are wired to the combo.

    We check the wiring by firing each QShortcut.activated signal directly
    — QTest.keyClick doesn't reliably deliver to a panel that isn't shown,
    so we exercise the slot wiring instead of the platform key path.
    """
    from PySide6.QtGui import QShortcut

    doc, vol = _make_doc_with_volume((4, 8, 8))
    panel = _make_panel(qapp, doc, vol)
    try:
        shortcuts = {
            sc.key().toString(): sc
            for sc in panel.findChildren(QShortcut)
        }
        # Confirm all 5 tool keys are registered.
        for key in ("V", "F", "W", "P", "E"):
            assert key in shortcuts, f"missing shortcut for {key}; have {list(shortcuts)}"

        mapping = [("F", "Fill"), ("W", "Wand"), ("P", "Paint"),
                   ("E", "Erase"), ("V", "Navigate")]
        for key, expected in mapping:
            shortcuts[key].activated.emit()
            QApplication.processEvents()
            assert panel._tool_combo.currentText() == expected
    finally:
        panel.close()


def test_space_hold_temporarily_navigates(qapp):
    """Holding Space in Paint switches to Navigate; releasing restores Paint."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    doc, vol = _make_doc_with_volume((4, 8, 8))
    panel = _make_panel(qapp, doc, vol)
    try:
        panel.setFocus(Qt.OtherFocusReason)
        panel._tool_combo.setCurrentText("Paint")
        QApplication.processEvents()
        assert panel._tool_mode == 'paint'

        QTest.keyPress(panel, Qt.Key_Space)
        QApplication.processEvents()
        assert panel._tool_mode == 'navigate', "Space should swap to Navigate"
        assert panel._space_prev_tool == "Paint"

        QTest.keyRelease(panel, Qt.Key_Space)
        QApplication.processEvents()
        assert panel._tool_mode == 'paint', "Release should restore Paint"
        assert panel._space_prev_tool is None
    finally:
        panel.close()


def test_space_hold_is_noop_when_already_navigating(qapp):
    """If tool is already Navigate, Space does not snapshot/restore."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    doc, vol = _make_doc_with_volume((4, 8, 8))
    panel = _make_panel(qapp, doc, vol)
    try:
        panel.setFocus(Qt.OtherFocusReason)
        panel._tool_combo.setCurrentText("Navigate")
        QApplication.processEvents()
        QTest.keyPress(panel, Qt.Key_Space)
        QTest.keyRelease(panel, Qt.Key_Space)
        QApplication.processEvents()
        assert panel._tool_mode == 'navigate'
        assert panel._space_prev_tool is None
    finally:
        panel.close()


def test_camera_zoom_button_changes_scale_factor(qapp):
    """Zoom-in / Zoom-out buttons scale the camera's scale_factor."""
    doc, vol = _make_doc_with_volume((4, 8, 8))
    panel = _make_panel(qapp, doc, vol)
    try:
        cam = panel._view.camera
        cam._scale_factor = 100.0
        panel._camera_zoom(0.5)
        assert abs(cam._scale_factor - 50.0) < 1e-6
        panel._camera_zoom(2.0)
        assert abs(cam._scale_factor - 100.0) < 1e-6
    finally:
        panel.close()


def test_camera_rotate_button_changes_orientation(qapp):
    """Rotate-right button changes the camera's orientation quaternion
    (or azimuth on turntable cameras). The exact change depends on the
    camera type — we just assert that the orientation moved."""
    doc, vol = _make_doc_with_volume((4, 8, 8))
    panel = _make_panel(qapp, doc, vol)
    try:
        cam = panel._view.camera
        if hasattr(cam, 'azimuth'):
            before = float(cam.azimuth)
            panel._camera_rotate(15, 0)
            assert abs(float(cam.azimuth) - (before + 15)) < 1e-6
        elif hasattr(cam, '_quaternion'):
            q0 = cam._quaternion
            before = (q0.w, q0.x, q0.y, q0.z)
            panel._camera_rotate(15, 0)
            q1 = cam._quaternion
            after = (q1.w, q1.x, q1.y, q1.z)
            assert before != after, "quaternion should change after rotate"
        else:
            pytest.skip("camera exposes no rotation API we recognize")
    finally:
        panel.close()


def test_cursor_readout_updates_with_position(qapp, monkeypatch):
    """Moving the mouse (no drag) updates the readout with (z,y,x) + intensity."""
    doc, vol = _make_doc_with_volume((4, 8, 8))
    panel = _make_panel(qapp, doc, vol)
    try:
        # _update_cursor_readout bails when _volumes is empty. In the
        # offscreen test the deferred GPU upload hasn't run, so seed a
        # sentinel so the path past the early-return actually executes.
        panel._volumes = [object()]
        monkeypatch.setattr(
            panel, '_ray_pick_seed',
            lambda pos, require_bright=True: (2, 3, 4),
        )
        panel._update_cursor_readout((0, 0))
        text = panel._cursor_readout_label.text()
        assert "z=2" in text and "y=3" in text and "x=4" in text
        assert "intensity=" in text
    finally:
        panel._volumes = []
        panel.close()


def test_tool_cursor_changes_per_tool(qapp):
    """Each tool sets a distinctive Qt stock cursor on the canvas.

    Navigate = OpenHandCursor (grab metaphor), Fill = PointingHandCursor
    (click this), Paint/Wand/Erase = CrossCursor (precision point).
    """
    from PySide6.QtCore import Qt

    doc, vol = _make_doc_with_volume((4, 8, 8))
    panel = _make_panel(qapp, doc, vol)
    try:
        expected = {
            "Navigate": Qt.OpenHandCursor,
            "Fill": Qt.PointingHandCursor,
            "Wand": Qt.CrossCursor,
            "Paint": Qt.CrossCursor,
            "Erase": Qt.CrossCursor,
        }
        for label, shape in expected.items():
            panel._tool_combo.setCurrentText(label)
            QApplication.processEvents()
            got = panel._canvas.native.cursor().shape()
            assert got == shape, (
                f"{label}: expected cursor {shape}, got {got}"
            )
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
