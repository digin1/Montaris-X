"""Measure ``Convert to Labels`` end-to-end on the user's actual TIF
(``neuron_1_branch_labels_full.tif``) — once via the 2D max-projection
path (no 3D viewer open) and once via the 3D path (3D viewer open).

The user reported "force quit / keep waiting" dialogs while running
this from the layer-panel context menu, so we want a step-by-step
breakdown of where the wall-time goes.

Run on a real display:
    DISPLAY=:1 LD_LIBRARY_PATH="" /usr/bin/python3 tests/headed_3d_convert_perf.py
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

LABELS_PATH = os.path.join(REPO, "neuron_1_branch_labels_full.tif")
INTENSITY_PATH = os.path.join(
    REPO,
    "GQR183_s02_eGFP_cellfill_PSD93SiRHalo_PSD93JF552Halo_eGFPfill_bottom_"
    "100x_100nmstep_10mthick_CrotexVIS_27072025-1.tif",
)


@contextmanager
def _t(name):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    flag = "🚨" if dt * 1000 > 1000 else ("⚠ " if dt * 1000 > 200 else "  ")
    print(f"  {flag} [{dt*1000:8.1f} ms]  {name}")


def main():
    if not os.path.exists(LABELS_PATH):
        print(f"FATAL: missing {LABELS_PATH}")
        return 2

    import tifffile
    print("Loading neuron skeletons TIF (the user's actual scenario)...")
    labels_arr = tifffile.imread(LABELS_PATH)
    n_unique = len(np.unique(labels_arr))
    print(f"  shape={labels_arr.shape} dtype={labels_arr.dtype} "
          f"unique={n_unique} max={int(labels_arr.max())}")

    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from unittest.mock import patch
    qapp = QApplication.instance() or QApplication(sys.argv)

    from montaris.app import MontarisApp
    from montaris.layers import ImageLayer, MontageDocument

    print("\n=== Setup ===")
    app = MontarisApp()
    app.show()
    # Mirror what File→Open does for a 3D TIF: max-project as the 2D
    # image layer, store the full volume on the doc.
    with _t("max-project + load image_layer"):
        mp = labels_arr.max(axis=0)
        # Coerce to uint8 the same way MontarisApp does.
        if mp.dtype != np.uint8:
            mp_max = int(mp.max()) or 1
            mp_8 = (mp.astype(np.float32) * (255.0 / mp_max)).clip(0, 255).astype(np.uint8)
        else:
            mp_8 = mp
        img_layer = ImageLayer("neuron_labels", mp_8)
        app.layer_stack.set_image(img_layer)
        doc = MontageDocument(
            name="GQR_neurons", image_layer=img_layer,
            volume_data=labels_arr, volume_axes="ZYX",
        )
        app._documents = [doc]
    qapp.processEvents()

    # CASE A — 3D viewer NOT open. Convert routes to the 2D path on the
    # max-projection. This is the typical "user just opened the TIF and
    # right-clicks Convert to Labels" scenario.
    print(f"\n=== A. 2D path (3D viewer NOT open) — image is "
          f"{mp_8.shape} uint8 with {n_unique} unique values in projection ===")

    # Patch the >512 confirm dialog to auto-accept (test sees ~195 ids
    # in the projection so it shouldn't trigger, but defend).
    with patch.object(QMessageBox, 'warning', return_value=QMessageBox.Yes):
        with _t("convert_image_to_label_layers (2D path, 195 ROIs)"):
            app.convert_image_to_label_layers()
            for _ in range(15):
                qapp.processEvents()

    n_2d = len([r for r in app.layer_stack.roi_layers if not getattr(r, 'is_volume', False)])
    print(f"  created {n_2d} 2D ROIs")

    # Reset for case B.
    app.layer_stack.roi_layers.clear()
    doc.labels_3d = None
    doc.labels_meta = {}
    doc.labels_next_id = 1
    qapp.processEvents()

    # CASE B — open the 3D viewer first, then convert. Routes to the 3D
    # path on the full 348M-voxel volume.
    print(f"\n=== B. 3D path (3D viewer open) — full {labels_arr.shape} "
          f"uint16 volume with {n_unique} unique ids ===")
    with _t("open 3D viewer + drain initial build"):
        app._open_view_3d()
        deadline = time.time() + 60
        while time.time() < deadline:
            qapp.processEvents()
            panel = app._view3d_panel
            if panel is not None and panel._volumes:
                break
            time.sleep(0.005)

    with patch.object(QMessageBox, 'warning', return_value=QMessageBox.Yes):
        with _t("convert_image_to_label_layers (3D path, 195 ROIs)"):
            app.convert_image_to_label_layers()
            for _ in range(20):
                qapp.processEvents()

    vols = [r for r in app.layer_stack.roi_layers if getattr(r, 'is_volume', False)]
    print(f"  created {len(vols)} 3D ROIs (VolumeROILayer)")

    # CASE C — granular breakdown of the 2D path so we can see exactly
    # where the time goes when the user clicks Convert with no 3D open.
    print(f"\n=== C. 2D path step-by-step on the max-projection ===")
    print(f"  image shape = {mp_8.shape}, n_unique = {n_unique}")

    from montaris.core.rle import rle_encode
    from montaris.layers import ROILayer, generate_unique_roi_name

    with _t("Step 1: np.unique on (2599 x 1291) uint8"):
        values = np.unique(mp_8)
        values = values[values != 0]
    print(f"     n_values = {values.size}")

    # Per-value loop is the body of the 2D path — measure 1, then full.
    with _t("Step 2: ONE iteration (mask + rle_encode + ROILayer construct)"):
        v = int(values[0])
        mask = (mp_8 == v)
        rle_bytes, rle_shape = rle_encode(mask.astype(np.uint8, copy=False) * 255)
        roi = ROILayer.__new__(ROILayer)
        roi.name = generate_unique_roi_name(f"Label {v}", [])
        roi._mask = None
        roi._rle_data = rle_bytes
        roi._mask_shape = rle_shape
        roi.color = (255, 0, 0)
        roi.opacity = 128
        roi.visible = True
        roi.fill_mode = "solid"
        roi._dirty_rect = None
        roi.offset_x = 0
        roi.offset_y = 0
        roi._cached_bbox = None
        roi._bbox_valid = False

    # Now the full loop without the surrounding bookkeeping.
    n = int(values.size)
    with _t(f"Step 3: full loop body × {n} (mask + rle + ROI construct)"):
        added = []
        for value in values.tolist():
            mask = (mp_8 == value)
            if not mask.any():
                continue
            rle_bytes, rle_shape = rle_encode(mask.astype(np.uint8, copy=False) * 255)
            roi = ROILayer.__new__(ROILayer)
            roi.name = generate_unique_roi_name(
                f"Label {int(value)}", added,
            )
            roi._mask = None
            roi._rle_data = rle_bytes
            roi._mask_shape = rle_shape
            roi.color = (255, 0, 0)
            roi.opacity = 128
            roi.visible = True
            roi.fill_mode = "solid"
            roi._dirty_rect = None
            roi.offset_x = 0
            roi.offset_y = 0
            roi._cached_bbox = None
            roi._bbox_valid = False
            added.append(roi)
    print(f"     final added: {len(added)}")

    # Drill into the per-iteration cost: which sub-step dominates?
    print(f"\n=== D. Per-step micro-timings on a single iteration ===")
    v = int(values[100])  # a typical value with non-trivial extent
    with _t("(a) mask = (mp_8 == value)  — full image scan"):
        mask = (mp_8 == v)
    with _t("(b) mask.any()"):
        _ = mask.any()
    with _t("(c) mask.astype(uint8) * 255"):
        m255 = mask.astype(np.uint8, copy=False) * 255
    with _t("(d) rle_encode"):
        rle_bytes, rle_shape = rle_encode(m255)
    with _t("(e) generate_unique_roi_name (vs an existing list of 200)"):
        fake_existing = added[:200] if len(added) >= 200 else added
        _ = generate_unique_roi_name(f"Label {v}", fake_existing)

    print("\n=== Summary ===")
    print(f"  Image: {mp_8.shape} uint8 max-projection")
    print(f"  Unique non-zero values: {values.size}")
    print(f"  TIF: {labels_arr.shape} {labels_arr.dtype} "
          f"(volume_data on the doc)")

    app.close()
    qapp.processEvents()
    return 0


if __name__ == "__main__":
    sys.exit(main())
