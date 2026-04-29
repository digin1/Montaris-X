"""Headed measurement of three user-reported slow paths in the 3D view:
1. Delete-all (Clear All) on 195 skeletons — was ~30 s freeze
2. Duplicate ROI propagation to the 3D viewer
3. Erase stroke release — was ~2.5 s full rebuild on every release

Run on a real display:
    DISPLAY=:1 LD_LIBRARY_PATH="" /usr/bin/python3 tests/headed_3d_delete_dup_erase_perf.py
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

INTENSITY_PATH = os.path.join(
    REPO,
    "GQR183_s02_eGFP_cellfill_PSD93SiRHalo_PSD93JF552Halo_eGFPfill_bottom_"
    "100x_100nmstep_10mthick_CrotexVIS_27072025-1.tif",
)
LABELS_PATH = os.path.join(REPO, "neuron_1_branch_labels_full.tif")


@contextmanager
def _t(label):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    flag = "🚨" if dt * 1000 > 1000 else ("⚠ " if dt * 1000 > 200 else "  ")
    print(f"  {flag} [{dt*1000:8.1f} ms]  {label}")


def main():
    if not (os.path.exists(INTENSITY_PATH) and os.path.exists(LABELS_PATH)):
        print("FATAL: assets missing")
        return 2

    import tifffile
    print("Loading neuron skeletons...")
    labels_arr = tifffile.imread(LABELS_PATH)
    intensity = tifffile.imread(INTENSITY_PATH)
    print(f"  labels {labels_arr.shape} {labels_arr.dtype}  "
          f"unique={len(np.unique(labels_arr))}")

    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import Qt
    from unittest.mock import patch
    qapp = QApplication.instance() or QApplication(sys.argv)

    from montaris.app import MontarisApp
    from montaris.layers import (
        ImageLayer, MontageDocument, ROI_COLORS, VolumeROILayer,
    )

    app = MontarisApp()
    app.show()
    mp = labels_arr.max(axis=0).astype(np.uint8)
    img_layer = ImageLayer("neuron_labels", mp)
    app.layer_stack.set_image(img_layer)
    doc = MontageDocument(
        name="GQR183", image_layer=img_layer,
        volume_data=intensity, volume_axes="ZYX",
    )
    app._documents = [doc]
    doc.labels_3d = labels_arr.astype(np.uint16, copy=True)
    ids = np.unique(labels_arr)
    meta = {}
    for lid in ids:
        if lid == 0:
            continue
        lid = int(lid)
        meta[lid] = {
            "name": f"3D ROI {lid}",
            "color": ROI_COLORS[(lid - 1) % len(ROI_COLORS)],
            "opacity": 128, "visible": True, "fill_mode": "solid",
        }
    doc.labels_meta = meta
    doc.labels_next_id = max(meta.keys()) + 1
    for lid in sorted(meta.keys()):
        app.layer_stack.roi_layers.append(VolumeROILayer(doc, lid))
    qapp.processEvents()

    print("\n=== A. Open 3D ===")
    with _t("open 3D + drain"):
        app._open_view_3d()
        deadline = time.time() + 60
        while time.time() < deadline:
            qapp.processEvents()
            panel = app._view3d_panel
            if panel is not None and panel._volumes:
                break
            time.sleep(0.005)
    panel = app._view3d_panel
    panel._rebuild_labels_overlay()
    for _ in range(8):
        qapp.processEvents()

    n_rois_pre = len(app.layer_stack.roi_layers)
    print(f"\n  initial: {n_rois_pre} ROIs, {len(doc.labels_meta)} meta keys, "
          f"{int((doc.labels_3d != 0).sum()):,} labelled voxels")

    print("\n=== B. Duplicate ROI (was: 3D canvas didn't pick up new id) ===")
    keys_before = set(doc.labels_meta.keys())
    panel_keys_before = (frozenset(panel._labels_meta_keys)
                        if panel._labels_meta_keys else frozenset())
    with _t("duplicate row 1 → expect new id appears in panel + canvas LUT"):
        app.layer_panel.list_widget.setCurrentRow(1)
        app.layer_panel._duplicate_selected()
        for _ in range(10):
            qapp.processEvents()
    keys_after = set(doc.labels_meta.keys())
    panel_keys_after = (frozenset(panel._labels_meta_keys)
                       if panel._labels_meta_keys else frozenset())
    new_ids_doc = keys_after - keys_before
    new_ids_panel = panel_keys_after - panel_keys_before
    print(f"  new id added to doc.labels_meta: {new_ids_doc}")
    print(f"  panel cache picked up the new id: {new_ids_panel}")
    if new_ids_doc and new_ids_doc == new_ids_panel:
        print("  ✓ duplicate now propagates to 3D — fast path won't render new id as background")
    else:
        print("  ✘ panel cache didn't pick up the duplicate — still buggy")

    print("\n=== C. Erase stroke release (was: 2.5 s full rebuild every release) ===")
    # Simulate an erase stroke directly by setting drag state then calling
    # _finish_drag — we measure the release-time work, not the GL drag tick.
    panel._drag_active = True
    panel._drag_mode = 'erase'
    panel._drag_label_id = 0
    panel._drag_extends_existing = False
    panel._stroke_bbox = (10, 12, 100, 110, 200, 220)
    z0, z1, y0, y1, x0, x1 = panel._stroke_bbox
    panel._stroke_before = doc.labels_3d[z0:z1, y0:y1, x0:x1].copy()
    panel._stroke_dtype = doc.labels_3d.dtype
    # Pretend the drag wrote 0 into a chunk of voxels.
    doc.labels_3d[z0:z1, y0:y1, x0:x1] = 0
    with _t("_finish_drag for ERASE (cmap-only fast path)"):
        panel._finish_drag(emit=False)
        for _ in range(8):
            qapp.processEvents()

    print("\n=== D. Delete All (was: ~30 s freeze on 195 ROIs) ===")
    # Patch the QMessageBox confirm to auto-accept.
    with patch.object(QMessageBox, 'question', return_value=QMessageBox.Yes):
        with _t("Clear All on 195+ ROIs"):
            app.layer_panel._clear_all()
            for _ in range(15):
                qapp.processEvents()
    n_rois_post = len(app.layer_stack.roi_layers)
    voxels_after = int((doc.labels_3d != 0).sum())
    print(f"  ROIs remaining: {n_rois_post} (expected 0)")
    print(f"  labelled voxels remaining: {voxels_after:,} (expected 0)")
    print(f"  meta keys: {len(doc.labels_meta)} (expected 0)")

    print("\n=== E. Undo Clear All — restores everything ===")
    with _t("undo Clear All"):
        app.undo_stack.undo()
        for _ in range(15):
            qapp.processEvents()
    n_rois_undo = len(app.layer_stack.roi_layers)
    voxels_undo = int((doc.labels_3d != 0).sum())
    print(f"  ROIs after undo: {n_rois_undo} (expected ~{n_rois_pre + 1})")
    print(f"  labelled voxels: {voxels_undo:,}")

    app.close()
    qapp.processEvents()
    return 0


if __name__ == "__main__":
    sys.exit(main())
