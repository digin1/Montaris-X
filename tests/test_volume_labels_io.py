"""Phase 6: round-trip tests for 3D-ROI export/import and flatten-to-2D.

Covers:
- ``save_volume_labels`` / ``load_volume_labels`` round-trip (array + meta).
- Sidecar missing → meta auto-regenerated from unique IDs in the TIFF.
- Shape mismatch on import is rejected.
- ``flatten_volume_labels_to_2d`` emits one ROILayer per non-empty slice.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from montaris.io.volume_labels import (
    save_volume_labels, load_volume_labels, _sidecar_path,
)
from montaris.layers import ImageLayer, MontageDocument, VolumeROILayer


def _make_labels_volume():
    labels = np.zeros((6, 16, 20), dtype=np.uint16)
    labels[1:4, 2:8, 3:10] = 1
    labels[2:5, 10:14, 12:18] = 2
    meta = {
        1: {"name": "soma", "color": (255, 0, 0), "opacity": 200,
            "visible": True, "fill_mode": "solid"},
        2: {"name": "dendrite", "color": (0, 255, 0), "opacity": 150,
            "visible": False, "fill_mode": "outline"},
    }
    return labels, meta


def test_save_and_load_roundtrip(tmp_path):
    """Array content and every per-ID meta field survives a save/load cycle."""
    labels, meta = _make_labels_volume()
    out = tmp_path / "export.tif"
    tiff_path, sidecar_path = save_volume_labels(out, labels, meta)
    assert tiff_path.exists()
    assert sidecar_path.exists()

    labels_loaded, meta_loaded = load_volume_labels(out)
    np.testing.assert_array_equal(labels_loaded, labels)
    assert set(meta_loaded.keys()) == {1, 2}
    assert meta_loaded[1]["name"] == "soma"
    assert tuple(meta_loaded[1]["color"]) == (255, 0, 0)
    assert meta_loaded[1]["opacity"] == 200
    assert meta_loaded[2]["visible"] is False
    assert meta_loaded[2]["fill_mode"] == "outline"


def test_sidecar_path_derivation(tmp_path):
    """Sidecar lives next to the TIFF with a .labels.json suffix."""
    assert _sidecar_path(tmp_path / "a.tif") == tmp_path / "a.labels.json"
    assert _sidecar_path(tmp_path / "a.tiff") == tmp_path / "a.labels.json"
    # Uppercase extension also maps to the lowercase sidecar name.
    assert _sidecar_path(tmp_path / "a.TIF") == tmp_path / "a.labels.json"
    # No tiff extension — sidecar gets appended, original name kept.
    assert _sidecar_path(tmp_path / "x") == tmp_path / "x.labels.json"


def test_missing_sidecar_regenerates_meta(tmp_path):
    """When JSON is missing, meta is reconstructed from unique label IDs."""
    labels, meta = _make_labels_volume()
    tiff_path, sidecar_path = save_volume_labels(
        tmp_path / "export.tif", labels, meta,
    )
    sidecar_path.unlink()

    labels_loaded, meta_loaded = load_volume_labels(tiff_path)
    np.testing.assert_array_equal(labels_loaded, labels)
    # Both IDs present with default names + palette colors (not the originals).
    assert set(meta_loaded.keys()) == {1, 2}
    assert meta_loaded[1]["name"] == "3D ROI 1"
    assert meta_loaded[2]["name"] == "3D ROI 2"


def test_corrupt_sidecar_falls_back_to_auto_meta(tmp_path):
    """A truncated sidecar shouldn't crash the import — fall back to auto."""
    labels, meta = _make_labels_volume()
    tiff_path, sidecar_path = save_volume_labels(
        tmp_path / "export.tif", labels, meta,
    )
    sidecar_path.write_text("{not json")

    _, meta_loaded = load_volume_labels(tiff_path)
    # IDs recovered from the TIFF even though sidecar was unreadable.
    assert set(meta_loaded.keys()) == {1, 2}


def test_load_2d_labels_promotes_to_3d(tmp_path):
    """A single-slice label TIFF loads as a (1, Y, X) volume."""
    import tifffile
    arr = np.zeros((16, 20), dtype=np.uint8)
    arr[2:6, 3:9] = 1
    path = tmp_path / "single.tif"
    tifffile.imwrite(str(path), arr)
    loaded, meta = load_volume_labels(path)
    assert loaded.shape == (1, 16, 20)
    assert 1 in meta


def test_load_downcasts_to_uint8_when_possible(tmp_path):
    """Labels whose max fits in uint8 come back as uint8 to save RAM."""
    labels, meta = _make_labels_volume()
    out = tmp_path / "small.tif"
    save_volume_labels(out, labels, meta)
    loaded, _ = load_volume_labels(out)
    # Max id is 2 → comfortably fits in uint8.
    assert loaded.dtype == np.uint8


# ── MontarisApp-level (export / import / flatten actions) ──────────────────


def _attach_labels_doc(app, shape=(5, 16, 20)):
    z, y, x = shape
    vol = np.zeros(shape, dtype=np.uint8)
    vol[:] = np.arange(1, z + 1)[:, None, None] * 30
    mp = vol.max(axis=0)
    app.layer_stack.set_image(ImageLayer("zlabels", mp))
    if not app._documents:
        doc = MontageDocument(name="zlabels", image_layer=app.layer_stack.image_layer)
        app._documents = [doc]
        app._active_doc_index = 0
    doc = app._documents[app._active_doc_index]
    doc.volume_data = vol
    doc.volume_axes = "ZYX"
    doc.active_z = 0
    doc.ensure_labels_3d()
    # Two labels, each spanning multiple z-slices.
    lid_a = doc.reserve_label_id(name="roi_a", color=(255, 0, 0))
    lid_b = doc.reserve_label_id(name="roi_b", color=(0, 0, 255))
    doc.labels_3d[1:3, 2:6, 2:6] = lid_a
    doc.labels_3d[3:5, 10:14, 10:14] = lid_b
    app.layer_stack.roi_layers.extend([
        VolumeROILayer(doc, lid_a),
        VolumeROILayer(doc, lid_b),
    ])
    app.layer_stack.changed.emit()
    return doc, lid_a, lid_b


def test_export_import_via_app_roundtrip(app, tmp_path, monkeypatch):
    """Export through the menu action, wipe, reimport — labels match."""
    doc, lid_a, lid_b = _attach_labels_doc(app)

    out_path = tmp_path / "through_app.tif"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(out_path), "")),
    )
    app.export_volume_labels()
    assert out_path.exists()
    sidecar = _sidecar_path(out_path)
    assert sidecar.exists()

    original = doc.labels_3d.copy()
    original_meta_names = {lid: m["name"] for lid, m in doc.labels_meta.items()}

    # Wipe the document's labels and simulate an import.
    doc.labels_3d = np.zeros_like(doc.labels_3d)
    doc.labels_meta = {}
    app.layer_stack.roi_layers = []
    app.layer_stack.changed.emit()

    monkeypatch.setattr(
        QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **k: (str(out_path), "")),
    )
    app.import_volume_labels()
    np.testing.assert_array_equal(doc.labels_3d, original)
    assert {lid: m["name"] for lid, m in doc.labels_meta.items()} == original_meta_names
    # Each label was mirrored into a VolumeROILayer on the layer stack.
    vol_rois = [r for r in app.layer_stack.roi_layers
                if getattr(r, 'is_volume', False)]
    assert len(vol_rois) == 2


def test_import_rejects_shape_mismatch(app, tmp_path, monkeypatch):
    """Import refuses a labels TIFF whose shape ≠ the active volume."""
    doc, _, _ = _attach_labels_doc(app, shape=(5, 16, 20))

    # Build a labels TIFF with different shape.
    import tifffile
    bad = tmp_path / "bad.tif"
    tifffile.imwrite(str(bad), np.zeros((4, 10, 10), dtype=np.uint16))

    monkeypatch.setattr(
        QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **k: (str(bad), "")),
    )
    warnings = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: warnings.append(a)),
    )
    app.import_volume_labels()
    # Shape mismatch should warn and leave the doc's labels untouched.
    assert warnings, "expected a shape-mismatch warning"
    # Original labels still in place — the "2 labels" from the fixture.
    assert doc.labels_meta, "existing labels should not have been cleared"


def test_flatten_creates_one_roi_per_nonempty_slice(app):
    """Each label's Z-range becomes one 2D ROILayer per occupied slice."""
    doc, lid_a, lid_b = _attach_labels_doc(app, shape=(5, 16, 20))
    # lid_a covers z=1..2 (2 slices), lid_b covers z=3..4 (2 slices) → 4 ROIs.

    pre_2d = sum(
        1 for r in app.layer_stack.roi_layers if not getattr(r, 'is_volume', False)
    )
    app.flatten_volume_labels_to_2d()
    post_2d = [r for r in app.layer_stack.roi_layers
               if not getattr(r, 'is_volume', False)]
    assert len(post_2d) - pre_2d == 4
    # Names embed the z-index so the per-slice ordering is obvious.
    flat_names = {r.name for r in post_2d}
    assert "roi_a_z001" in flat_names
    assert "roi_a_z002" in flat_names
    assert "roi_b_z003" in flat_names
    assert "roi_b_z004" in flat_names
    # The 3D wrappers are still present — flatten is additive.
    vol_rois = [r for r in app.layer_stack.roi_layers
                if getattr(r, 'is_volume', False)]
    assert len(vol_rois) == 2


def test_export_no_labels_shows_info(app, monkeypatch):
    """With nothing painted the menu action bails out politely."""
    # No doc, no labels.
    infos = []
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **k: infos.append(a)),
    )
    app.export_volume_labels()
    assert infos, "expected an info popup when no labels exist"
