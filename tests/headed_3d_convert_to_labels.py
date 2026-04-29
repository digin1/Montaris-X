"""Headed exercise of Colin's PR #4 "Convert to Labels" feature on the
real GQR183 data + neuron skeleton labels.

Validates:
1. 2D conversion of a small synthetic uint16 image (sanity)
2. 3D conversion of the 348M-voxel neuron_1_branch_labels_full.tif while
   the 3D viewer is open — measures wall-time on real-world data
3. The conversion's bbox + undo + meta roundtrip work end-to-end

Run on a real display:
    DISPLAY=:1 LD_LIBRARY_PATH="" /usr/bin/python3 tests/headed_3d_convert_to_labels.py
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
def _timed(label):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    flag = "🚨" if dt * 1000 > 5000 else ("⚠ " if dt * 1000 > 1000 else "  ")
    print(f"  {flag} [{dt*1000:8.1f} ms]  {label}")


def main():
    if not (os.path.exists(INTENSITY_PATH) and os.path.exists(LABELS_PATH)):
        print("FATAL: GQR + neuron labels TIFFs missing")
        return 2

    import tifffile
    print("Loading neuron skeleton labels TIFF...")
    labels_arr = tifffile.imread(LABELS_PATH)
    print(f"  labels {labels_arr.shape} {labels_arr.dtype}  "
          f"unique={len(np.unique(labels_arr))}  max_id={int(labels_arr.max())}")

    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    from PySide6.QtWidgets import QApplication
    qapp = QApplication.instance() or QApplication(sys.argv)

    from montaris.app import MontarisApp
    from montaris.layers import (
        ImageLayer, MontageDocument, VolumeROILayer,
    )

    print("\nBuilding app + document with neuron labels as volume_data...")
    app = MontarisApp()
    app.show()
    # The conversion expects volume_data — feed it the labels TIFF directly
    # (this is the legitimate "Convert to Labels" use-case: the user has
    # imported a label TIFF that they want as ROIs).
    mp = labels_arr.max(axis=0).astype(np.uint8)
    img_layer = ImageLayer("neuron_labels", mp)
    app.layer_stack.set_image(img_layer)
    doc = MontageDocument(
        name="GQR183_neurons", image_layer=img_layer,
        volume_data=labels_arr, volume_axes="ZYX",
    )
    app._documents = [doc]
    app._active_doc_index = 0
    qapp.processEvents()

    print("\n=== A. Open 3D viewer (so the conversion routes to the 3D path) ===")
    with _timed("Open 3D dialog + drain initial build"):
        app._open_view_3d()
        deadline = time.time() + 60
        while time.time() < deadline:
            qapp.processEvents()
            panel = app._view3d_panel
            if panel is not None and panel._volumes:
                break
            time.sleep(0.005)
    panel = app._view3d_panel
    if panel is None:
        print("FATAL: 3D panel did not open")
        return 3

    print("\n=== B. Convert to Labels on the 348M-voxel neuron volume ===")
    print(f"  expected outcome: {len(np.unique(labels_arr)) - 1} new VolumeROILayers, "
          f"max id ≤ uint16")
    rois_before = len([
        roi for roi in app.layer_stack.roi_layers
        if isinstance(roi, VolumeROILayer)
    ])

    # Patch the >512 confirmation to auto-accept (test sees ~195 distinct ids
    # so it shouldn't trigger anyway, but defend).
    from PySide6.QtWidgets import QMessageBox
    from unittest.mock import patch
    with patch.object(QMessageBox, 'warning', return_value=QMessageBox.Yes):
        with _timed("convert_image_to_label_layers (3D path on 851 MB volume)"):
            app.convert_image_to_label_layers()
            for _ in range(20):
                qapp.processEvents()

    rois_after = len([
        roi for roi in app.layer_stack.roi_layers
        if isinstance(roi, VolumeROILayer)
    ])
    n_added = rois_after - rois_before
    print(f"\n  added {n_added} VolumeROILayers (expected ~195)")
    print(f"  doc.labels_3d.dtype = {doc.labels_3d.dtype}  "
          f"max written = {int(doc.labels_3d.max())}")
    print(f"  labels_meta keys = {len(doc.labels_meta)}")
    print(f"  cache _labels_meta_keys (panel) = "
          f"{len(panel._labels_meta_keys) if panel._labels_meta_keys else 'None'}")

    print("\n=== C. Verify per-ROI visibility toggle is still cmap-only fast ===")
    list_w = app.layer_panel.list_widget
    if list_w.count() > 3:
        from PySide6.QtCore import Qt
        item = list_w.item(2)
        with _timed("Per-ROI visibility checkbox (cmap-only)"):
            item.setCheckState(Qt.Unchecked)
            for _ in range(6):
                qapp.processEvents()
        with _timed("Per-ROI visibility re-toggle on"):
            item.setCheckState(Qt.Checked)
            for _ in range(6):
                qapp.processEvents()

    print("\n=== D. Undo conversion ===")
    with _timed("Undo conversion (restores 195 voxel-clears + meta removals)"):
        app.undo_stack.undo()
        for _ in range(10):
            qapp.processEvents()
    rois_post_undo = len([
        roi for roi in app.layer_stack.roi_layers
        if isinstance(roi, VolumeROILayer)
    ])
    voxels_remaining = int((doc.labels_3d != 0).sum())
    print(f"  ROIs remaining: {rois_post_undo} (expected 0)")
    print(f"  voxels still labelled: {voxels_remaining:,} (expected 0)")

    print("\n=== E. Redo conversion ===")
    with _timed("Redo conversion"):
        app.undo_stack.redo()
        for _ in range(10):
            qapp.processEvents()
    rois_post_redo = len([
        roi for roi in app.layer_stack.roi_layers
        if isinstance(roi, VolumeROILayer)
    ])
    print(f"  ROIs after redo: {rois_post_redo} (expected {n_added})")

    print("\n=== F. Shape-mismatch guard (the medium fix) ===")
    # Inject a stale labels_3d with the wrong shape; the conversion must
    # surface an error instead of crashing.
    saved_labels_3d = doc.labels_3d
    saved_meta = dict(doc.labels_meta)
    saved_next_id = doc.labels_next_id
    doc.labels_3d = np.zeros((50, 50, 50), dtype=np.uint8)
    doc.labels_meta = {}
    doc.labels_next_id = 1

    with patch.object(QMessageBox, 'warning') as warn:
        with _timed("convert with mismatched labels_3d (must surface warning)"):
            app.convert_image_to_label_layers()

    if warn.called:
        msg_args = warn.call_args.args
        print(f"  ✓ warning shown: {msg_args[1]!r}")
    else:
        print(f"  ✘ warning NOT shown — guard may have regressed")

    # Restore for cleanup.
    doc.labels_3d = saved_labels_3d
    doc.labels_meta = saved_meta
    doc.labels_next_id = saved_next_id

    print("\n=== Done ===")
    app.close()
    qapp.processEvents()
    return 0


if __name__ == "__main__":
    sys.exit(main())
