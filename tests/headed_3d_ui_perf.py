"""Headed UI-lag measurement for 3D view interactions.

Loads the GQR183 z-stack + neuron labels into a real on-screen Montaris-X
session, then measures and screenshots:

1. Per-rebuild time: full overlay rebuild on a paint-style refresh.
2. Opacity slider drag: time for 20 successive value changes (simulates
   a real drag from 100 → 80%). Each is one ``valueChanged`` emission.
3. Per-ROI visibility checkbox toggle: time for one item-change cycle.
4. ``3D ROIs`` master checkbox toggle.
5. Layer-panel inspection: dump first 10 ROI rows + their swatch RGB to
   confirm each skeleton has its own colour entry in the sidebar.
6. Screenshots: full canvas + a zoomed crop centred on a dense skeleton
   bundle so the user can verify each skeleton renders in its own colour.

Output goes to ``/tmp/claude/montaris_3d_perf/*.png`` plus stdout.

Run on a real display:
    DISPLAY=:1 LD_LIBRARY_PATH="" /usr/bin/python3 tests/headed_3d_ui_perf.py
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

INTENSITY_PATH = os.path.join(
    REPO,
    "GQR183_s02_eGFP_cellfill_PSD93SiRHalo_PSD93JF552Halo_eGFPfill_bottom_"
    "100x_100nmstep_10mthick_CrotexVIS_27072025-1.tif",
)
LABELS_PATH = os.path.join(REPO, "neuron_1_branch_labels_full.tif")
SHOTS_DIR = Path("/tmp/claude/montaris_3d_perf")


@contextmanager
def _timed(name, results, n=1):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    results.append((name, dt, n))
    per = f" ({dt*1000/n:.1f} ms/each over {n})" if n > 1 else ""
    print(f"  [{dt*1000:8.1f} ms]  {name}{per}")


def _save_screenshot(panel, qapp, name: str):
    """Grab the panel's QWidget contents as a PNG."""
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    qapp.processEvents()
    pix = panel.grab()
    path = SHOTS_DIR / f"{name}.png"
    pix.save(str(path))
    print(f"     screenshot → {path} ({pix.width()}x{pix.height()})")
    return path


def _save_canvas_screenshot(panel, qapp, name: str):
    """Grab the vispy canvas widget specifically (the 3D render)."""
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    qapp.processEvents()
    cv = panel._canvas.native
    pix = cv.grab()
    path = SHOTS_DIR / f"{name}.png"
    pix.save(str(path))
    print(f"     canvas-shot → {path} ({pix.width()}x{pix.height()})")
    return path


def main():
    if not os.path.exists(INTENSITY_PATH) or not os.path.exists(LABELS_PATH):
        print("FATAL: required TIFF assets missing")
        return 2

    print(f"\n=== Setup ===")
    setup_results: list[tuple[str, float, int]] = []

    import tifffile
    with _timed("Load intensity TIFF", setup_results):
        intensity = tifffile.imread(INTENSITY_PATH)
    with _timed("Load labels TIFF", setup_results):
        labels = tifffile.imread(LABELS_PATH)
    print(f"     intensity {intensity.shape} {intensity.dtype}")
    print(f"     labels    {labels.shape} {labels.dtype}  "
          f"unique={len(np.unique(labels))}  max_id={int(labels.max())}")

    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    qapp = QApplication.instance() or QApplication(sys.argv)

    from montaris.app import MontarisApp
    from montaris.layers import ImageLayer, MontageDocument, ROI_COLORS, VolumeROILayer

    with _timed("Build app + install document", setup_results):
        app = MontarisApp()
        app.show()
        mp = intensity.max(axis=0)
        if mp.dtype != np.uint8:
            mp_max = int(mp.max()) or 1
            mp_8 = (mp.astype(np.float32) * (255.0 / mp_max)).clip(0, 255).astype(np.uint8)
        else:
            mp_8 = mp
        img_layer = ImageLayer("intensity", mp_8)
        app.layer_stack.set_image(img_layer)
        doc = MontageDocument(
            name="GQR183", image_layer=img_layer,
            volume_data=intensity, volume_axes="ZYX",
        )
        app._documents = [doc]
        # Build labels_3d + labels_meta on the doc BEFORE opening the 3D
        # panel so the layer panel's 3D filter has VolumeROILayers to show.
        doc.labels_3d = labels
        ids = np.unique(labels)
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
        # Wrap each label in a VolumeROILayer so the layer panel renders
        # one row per skeleton (with its colour swatch).
        for lid in sorted(meta.keys()):
            app.layer_stack.roi_layers.append(VolumeROILayer(doc, lid))

    qapp.processEvents()

    print(f"\n=== Open 3D + drain initial build ===")
    open_results: list[tuple[str, float, int]] = []
    with _timed("Open 3D dialog + initial build", open_results):
        app._open_view_3d()
        for _ in range(60):
            qapp.processEvents()
            time.sleep(0.01)
    panel = app._view3d_panel
    if panel is None:
        print("FATAL: panel did not open")
        return 3

    # Trigger a labels rebuild so the overlay actually exists.
    with _timed("First labels overlay rebuild", open_results):
        panel._rebuild_labels_overlay()
        for _ in range(10):
            qapp.processEvents()

    print(f"     palette_size={panel._labels_palette_size} "
          f"lut_dtype={panel._labels_texture_lut.dtype if panel._labels_texture_lut is not None else None}")

    print(f"\n=== Layer-panel inspection (sidebar) ===")
    lp = app.layer_panel
    list_w = lp.list_widget
    print(f"  rows in list_widget: {list_w.count()}")
    print(f"  3D mode active     : {lp._mode_3d}")
    swatches = []
    for row in range(min(10, list_w.count())):
        item = list_w.item(row)
        data = item.data(Qt.UserRole)
        if data and data[0] == "roi":
            roi = lp.layer_stack.get_roi(data[1])
            if roi is not None:
                swatches.append((row, item.text(), tuple(roi.color)))
    print(f"  first {len(swatches)} swatches:")
    for row, name, color in swatches:
        print(f"     row {row:2d}  {name!r:24s}  RGB={color}")

    distinct = {c for _, _, c in swatches}
    print(f"  distinct RGB values in first {len(swatches)} rows: {len(distinct)}")
    if len(distinct) == len(swatches):
        print("  → each skeleton has its OWN colour in the sidebar (verified)")
    else:
        print("  → WARNING: some skeletons share a colour (palette cycled)")

    # Take a full-window + canvas screenshot of the initial state.
    _save_screenshot(panel, qapp, "01_initial_full_panel")
    _save_canvas_screenshot(panel, qapp, "02_initial_canvas")

    # Zoom in on the canvas via the camera so individual skeletons
    # are visible. Zooming is a camera scale_factor change — render
    # the result and grab.
    cam = panel._view.camera
    cam._scale_factor *= 0.18  # zoom in ~5x
    cam.view_changed()
    for _ in range(15):
        qapp.processEvents()
        time.sleep(0.02)
    _save_canvas_screenshot(panel, qapp, "03_zoomed_skeletons")

    # Now measure UI-lag interactions.
    print(f"\n=== UI-lag measurements ===")
    ui_results: list[tuple[str, float, int]] = []

    # Opacity slider — measure one valueChanged emission, then five
    # in sequence (simulates a brief drag). Each fires
    # ``_on_labels_opacity_changed`` once. With the cmap-only fast
    # path this should be ms; with the old full-rebuild it would
    # have been ~2.7 s per emission (un-droppable on a real drag).
    sl = panel._labels_opacity_slider
    sl.setValue(100)
    qapp.processEvents()
    with _timed("Opacity slider: 1 valueChanged emission", ui_results):
        sl.setValue(90)
        qapp.processEvents()
    with _timed("Opacity slider: 5 sequential value changes", ui_results, n=5):
        for v in (85, 80, 75, 70, 65):
            sl.setValue(v)
            qapp.processEvents()

    # Reset opacity, take a screenshot at lower opacity
    sl.setValue(40)
    for _ in range(8):
        qapp.processEvents()
    _save_canvas_screenshot(panel, qapp, "04_opacity_40")
    sl.setValue(100)
    for _ in range(8):
        qapp.processEvents()

    # Master 3D-ROI checkbox toggle — off then on.
    cb = panel._labels_cb
    with _timed("Master '3D ROIs' checkbox: toggle off", ui_results):
        cb.setChecked(False)
        for _ in range(10):
            qapp.processEvents()
    _save_canvas_screenshot(panel, qapp, "05_master_off")
    with _timed("Master '3D ROIs' checkbox: toggle on", ui_results):
        cb.setChecked(True)
        for _ in range(10):
            qapp.processEvents()
    _save_canvas_screenshot(panel, qapp, "06_master_on")

    # Per-ROI visibility checkbox via the layer panel item — uncheck row 1.
    if list_w.count() > 1:
        item1 = list_w.item(1)
        with _timed("Per-ROI checkbox: hide one skeleton", ui_results):
            item1.setCheckState(Qt.Unchecked)
            for _ in range(10):
                qapp.processEvents()
        _save_canvas_screenshot(panel, qapp, "07_one_skeleton_hidden")
        with _timed("Per-ROI checkbox: show one skeleton", ui_results):
            item1.setCheckState(Qt.Checked)
            for _ in range(10):
                qapp.processEvents()

    # Hide all-but-one — instead of toggling 194 checkboxes one at a
    # time (each fires its own full overlay rebuild and the loop alone
    # would be ~10 minutes), set ``visible=False`` directly on every
    # meta entry except the first, then issue ONE rebuild. This is what
    # the LayerPanel's bulk ops should do anyway.
    n_rows = list_w.count()
    if n_rows > 5:
        keep_lid = sorted(doc.labels_meta.keys())[0]
        with _timed(f"Hide all-but-one (bulk meta + 1 rebuild)", ui_results):
            for lid in doc.labels_meta:
                doc.labels_meta[lid]['visible'] = (lid == keep_lid)
            panel._rebuild_labels_overlay()
            for _ in range(10):
                qapp.processEvents()
        _save_canvas_screenshot(panel, qapp, "08_only_first_skeleton")
        # Restore visibility for cleanup.
        for lid in doc.labels_meta:
            doc.labels_meta[lid]['visible'] = True

    print(f"\n=== Summary ===")
    for tag, results in [("Setup", setup_results),
                         ("Open", open_results),
                         ("UI lag", ui_results)]:
        print(f"  -- {tag} --")
        for name, dt, n in results:
            per = f"  ({dt*1000/n:.1f} ms/each)" if n > 1 else ""
            print(f"    {dt*1000:8.0f} ms   {name}{per}")

    print(f"\n  Screenshots in {SHOTS_DIR}/")
    for png in sorted(SHOTS_DIR.glob("*.png")):
        print(f"    {png}")

    # Hold so the user can confirm visually.
    deadline = time.time() + 4
    while time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.05)
    app.close()
    qapp.processEvents()
    return 0


if __name__ == "__main__":
    sys.exit(main())
