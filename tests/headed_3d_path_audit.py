"""Headed audit — exercise every major 3D-view interaction path on the
GQR183 file (851 MB intensity + 851 MB neuron skeleton labels) and
measure where the UI thread blocks.

Each measurement reports max-ms (worst single emission) and total-ms
(across N samples). A path is "responsive" if max-ms < 100; "laggy"
if 100-500; "blocking" if > 500 ms.

Run on a real display:
    DISPLAY=:1 LD_LIBRARY_PATH="" /usr/bin/python3 tests/headed_3d_path_audit.py
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
def _timed(name, results, n=1):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    results.append((name, dt, n))
    per_ms = dt * 1000 / n
    flag = "  "
    if per_ms > 500:
        flag = "🚨"
    elif per_ms > 100:
        flag = "⚠ "
    extra = f"  ({per_ms:.1f} ms/each over {n})" if n > 1 else ""
    print(f"  {flag} [{dt*1000:8.1f} ms]  {name}{extra}")


def main():
    if not (os.path.exists(INTENSITY_PATH) and os.path.exists(LABELS_PATH)):
        print("FATAL: TIFF assets missing")
        return 2

    import tifffile
    print("Loading TIFFs...")
    intensity = tifffile.imread(INTENSITY_PATH)
    labels_arr = tifffile.imread(LABELS_PATH)
    print(f"  intensity {intensity.shape} {intensity.dtype}")
    print(f"  labels    {labels_arr.shape} {labels_arr.dtype}")

    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    qapp = QApplication.instance() or QApplication(sys.argv)

    from montaris.app import MontarisApp
    from montaris.layers import (
        ImageLayer, MontageDocument, ROI_COLORS, VolumeROILayer,
    )

    print("\nBuilding app...")
    app = MontarisApp()
    app.show()
    mp_8 = (intensity.max(axis=0).astype(np.float32) * (255.0 / max(1, int(intensity.max())))).clip(0, 255).astype(np.uint8)
    img_layer = ImageLayer("intensity", mp_8)
    app.layer_stack.set_image(img_layer)
    doc = MontageDocument(name="GQR183", image_layer=img_layer,
                         volume_data=intensity, volume_axes="ZYX")
    app._documents = [doc]
    doc.labels_3d = labels_arr
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
    for lid in sorted(meta.keys()):
        app.layer_stack.roi_layers.append(VolumeROILayer(doc, lid))
    qapp.processEvents()

    results: list[tuple[str, float, int]] = []

    print("\n=== A. Cold-open the 3D dialog ===")
    # Wait for the deferred rebuild to actually finish, not just for
    # the first 600 ms tick. A condition-driven drain measures real time.
    with _timed("Open 3D dialog + initial intensity volume upload", results):
        app._open_view_3d()
        deadline = time.time() + 60
        while time.time() < deadline:
            qapp.processEvents()
            panel = app._view3d_panel
            if panel is not None and panel._volumes:
                break
            time.sleep(0.005)

    panel = app._view3d_panel
    assert panel is not None and panel._volumes, "panel never finished initial build"

    with _timed("First labels overlay rebuild (851 MB → uint8 dense palette)", results):
        panel._rebuild_labels_overlay()
        for _ in range(8):
            qapp.processEvents()

    print(f"\n  GPU budget: {panel._vram_budget // (1024*1024)} MB  "
          f"downsample axes: {panel._last_total_ds_axes}  "
          f"palette_size: {panel._labels_palette_size}")

    print("\n=== B. Layer-panel signals (visibility / opacity / colour) ===")
    list_w = app.layer_panel.list_widget
    if list_w.count() > 3:
        item = list_w.item(2)
        with _timed("Per-ROI visibility checkbox toggle (cmap-only fast path)", results):
            item.setCheckState(Qt.Unchecked)
            for _ in range(6):
                qapp.processEvents()
        with _timed("Per-ROI visibility re-toggle on", results):
            item.setCheckState(Qt.Checked)
            for _ in range(6):
                qapp.processEvents()

    sl = panel._labels_opacity_slider
    sl.setValue(100)
    qapp.processEvents()
    with _timed("Opacity slider 1 valueChanged (cmap-only)", results):
        sl.setValue(85)
        qapp.processEvents()
    with _timed("Opacity slider 10 sequential changes (drag sim)", results, n=10):
        for v in range(80, 70, -1):
            sl.setValue(v)
            qapp.processEvents()
    sl.setValue(100)
    qapp.processEvents()

    print("\n=== C. Master toggles ===")
    cb = panel._labels_cb
    with _timed("Master '3D ROIs' checkbox off", results):
        cb.setChecked(False)
        for _ in range(6):
            qapp.processEvents()
    with _timed("Master '3D ROIs' checkbox on", results):
        cb.setChecked(True)
        for _ in range(6):
            qapp.processEvents()

    if panel._channel_toggles:
        ch = panel._channel_toggles[0]
        with _timed("Channel visibility toggle off", results):
            ch.setChecked(False)
            for _ in range(6):
                qapp.processEvents()
        with _timed("Channel visibility toggle on", results):
            ch.setChecked(True)
            for _ in range(6):
                qapp.processEvents()

    print("\n=== D. Render mode dropdown ===")
    if panel._mode_combo:
        with _timed("Render mode change: mip → translucent", results):
            panel._mode_combo.setCurrentText('Translucent')
            for _ in range(6):
                qapp.processEvents()
        with _timed("Render mode change: translucent → mip", results):
            panel._mode_combo.setCurrentText('MIP')
            for _ in range(6):
                qapp.processEvents()

    print("\n=== E. Tool switching ===")
    for tool_name in ['paint', 'erase', 'fill', 'wand', 'navigate']:
        btn = panel._tool_buttons.get(tool_name)
        if btn is None:
            continue
        with _timed(f"Tool switch → {tool_name}", results):
            btn.setChecked(True)
            for _ in range(4):
                qapp.processEvents()

    print("\n=== F. Camera (rotate / zoom) ===")
    cam = panel._view.camera
    with _timed("Camera scale_factor change (zoom in 5×)", results):
        cam._scale_factor *= 0.2
        cam.view_changed()
        for _ in range(8):
            qapp.processEvents()
    with _timed("Camera scale_factor change (zoom out 5×)", results):
        cam._scale_factor *= 5.0
        cam.view_changed()
        for _ in range(8):
            qapp.processEvents()

    print("\n=== G. Downsample slider (heaviest path) ===")
    if panel._ds_slider:
        cur = panel._ds_slider.value()
        with _timed("Downsample slider 1 step (full intensity rebuild)", results):
            panel._ds_slider.setValue(cur + 1 if cur < panel._ds_slider.maximum() else cur - 1)
            deadline = time.time() + 30
            while time.time() < deadline:
                qapp.processEvents()
                if panel._volumes:
                    break
                time.sleep(0.005)
        with _timed("Downsample slider RAPID 5 sequential ticks (drag sim)", results, n=5):
            base = panel._ds_slider.value()
            mn, mx = panel._ds_slider.minimum(), panel._ds_slider.maximum()
            for d in range(5):
                target = max(mn, min(mx, base + (1 if d % 2 == 0 else -1)))
                panel._ds_slider.setValue(target)
                # NOTE: deliberately NOT draining — this simulates a real
                # drag where slider events fire faster than rebuild completes.
                qapp.processEvents()
        # Drain whatever's queued.
        deadline = time.time() + 30
        while time.time() < deadline:
            qapp.processEvents()
            if panel._volumes:
                break
            time.sleep(0.005)
        # Reset to original.
        panel._ds_slider.setValue(cur)
        deadline = time.time() + 30
        while time.time() < deadline:
            qapp.processEvents()
            if panel._volumes:
                break
            time.sleep(0.005)

    print("\n=== H. Tolerance slider (wand setting) ===")
    tol_slider = getattr(panel, '_tol_slider', None) or getattr(panel, '_tolerance_spin', None)
    if tol_slider is not None:
        with _timed("Tolerance setting 1 change (no rebuild expected)", results):
            tol_slider.setValue(60)
            for _ in range(4):
                qapp.processEvents()

    print("\n=== I. Unhappy paths ===")

    # Stale uint32 id in labels_3d (defensive code path).
    print("  Probing stale-id defence...")
    saved = doc.labels_3d[0, 0, 0]
    doc.labels_3d[0, 0, 0] = np.uint16(60000)  # outside meta; uint16 fits
    with _timed("Refresh after stale id injected (uint16)", results):
        panel._rebuild_labels_overlay()
        for _ in range(8):
            qapp.processEvents()
    doc.labels_3d[0, 0, 0] = saved

    # All-hidden short-circuit then re-enable (regression coverage).
    print("  Probing all-hidden → re-enable one ROI...")
    saved_visibility = {lid: m['visible'] for lid, m in doc.labels_meta.items()}
    for lid in doc.labels_meta:
        doc.labels_meta[lid]['visible'] = False
    with _timed("Hide all (bulk, 1 rebuild)", results):
        panel._rebuild_labels_overlay()
        for _ in range(6):
            qapp.processEvents()
    first = sorted(doc.labels_meta.keys())[0]
    doc.labels_meta[first]['visible'] = True
    with _timed("Re-enable one ROI (cmap-only after all_hidden)", results):
        panel.refresh_labels_meta_only()
        for _ in range(6):
            qapp.processEvents()
    # Restore.
    for lid, vis in saved_visibility.items():
        doc.labels_meta[lid]['visible'] = vis

    # Empty labels_meta (corner case).
    print("  Probing empty-meta short-circuit...")
    saved_meta = doc.labels_meta
    doc.labels_meta = {}
    with _timed("Refresh with empty labels_meta", results):
        panel._rebuild_labels_overlay()
        for _ in range(6):
            qapp.processEvents()
    doc.labels_meta = saved_meta
    panel._rebuild_labels_overlay()
    for _ in range(6):
        qapp.processEvents()

    print("\n=== J. Closing the panel ===")
    with _timed("Teardown 3D panel (back to 2D)", results):
        app._open_view_3d()  # toggles back
        for _ in range(8):
            qapp.processEvents()

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY (🚨 > 500ms blocking, ⚠ 100-500ms laggy, blank = responsive)")
    print("=" * 72)
    for name, dt, n in results:
        per_ms = dt * 1000 / n
        flag = "  "
        if per_ms > 500:
            flag = "🚨"
        elif per_ms > 100:
            flag = "⚠ "
        suffix = f"  ({per_ms:.1f} ms/each, ×{n})" if n > 1 else ""
        print(f"  {flag} {dt*1000:9.0f} ms   {name}{suffix}")

    app.close()
    qapp.processEvents()
    return 0


if __name__ == "__main__":
    sys.exit(main())
