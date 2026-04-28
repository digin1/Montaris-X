"""Headed performance measurement for 3D rendering pipeline.

Loads ``GQR183_*.tif`` (intensity, 851 MB, 133×2599×1291 uint16) and
``neuron_1_branch_labels_full.tif`` (labels, 851 MB, 195 sparse ids in
[0..663]) into a real on-screen Montaris-X session and reports:

1. Per-stage wall time: TIFF load → ImageLayer prep → 3D dialog open
   (intensity volume build) → labels import → labels overlay rebuild.
2. GPU memory auto-detect path: which branch fired (NVX, ATI, heuristic),
   what raw value the GL query returned, what budget the panel chose,
   and what nvidia-smi reports as actually free for comparison.
3. Downsample factor the panel picked given the detected budget vs the
   intensity volume size, plus the GL_MAX_3D_TEXTURE_SIZE cap.

Run on a machine with a real display:

    DISPLAY=:1 LD_LIBRARY_PATH="" /usr/bin/python3 tests/headed_3d_perf.py

This is a diagnostic script, not part of the pytest suite.
"""

from __future__ import annotations

import os
import subprocess
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
def _timed(name, results):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    results.append((name, dt))
    print(f"  [{dt*1000:8.1f} ms]  {name}")


def _nvidia_smi_free_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,nounits,noheader"],
            text=True, timeout=2,
        )
        free_mb, total_mb = (int(x.strip()) for x in out.strip().split(","))
        return free_mb, total_mb
    except Exception as e:
        return None, f"nvidia-smi failed: {e}"


def _raw_gl_query(canvas, enum):
    """Return raw GL query result without coercion — for diagnosis."""
    from vispy.gloo import gl
    try:
        with canvas.context:
            return gl.glGetParameter(enum)
    except Exception as e:
        return f"<query failed: {e}>"


def main():
    if not os.path.exists(INTENSITY_PATH):
        print(f"FATAL: intensity TIF missing: {INTENSITY_PATH}")
        return 2
    if not os.path.exists(LABELS_PATH):
        print(f"FATAL: labels TIF missing: {LABELS_PATH}")
        return 2

    free_before_mb, total_mb = _nvidia_smi_free_mb()
    print(f"\n=== nvidia-smi BEFORE app start ===")
    print(f"  Free VRAM:  {free_before_mb} MB / {total_mb} MB total")

    print(f"\n=== Pipeline timings ===")
    results: list[tuple[str, float]] = []

    # --- Stage 1: TIFF load ---
    with _timed("TIFF: load intensity (851 MB uint16, 133x2599x1291)", results):
        import tifffile
        intensity = tifffile.imread(INTENSITY_PATH)
    print(f"     intensity.shape={intensity.shape} dtype={intensity.dtype}")

    with _timed("TIFF: load labels (851 MB uint16)", results):
        labels = tifffile.imread(LABELS_PATH)
    n_unique = len(np.unique(labels))
    print(f"     labels.shape={labels.shape} dtype={labels.dtype} "
          f"unique_ids={n_unique} max_id={int(labels.max())}")

    # --- Stage 2: Qt + Montaris app ---
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    with _timed("Qt: import + QApplication construction", results):
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QTimer
        qapp = QApplication.instance() or QApplication(sys.argv)

    with _timed("Montaris: import app module", results):
        from montaris.app import MontarisApp
        from montaris.layers import ImageLayer, MontageDocument

    with _timed("Montaris: MontarisApp() construction", results):
        app = MontarisApp()
        app.show()

    # Build a 2D max-projection ImageLayer (what MontarisApp expects in 2D
    # before opening the 3D dialog).
    with _timed("ImageLayer: build max-projection (uint8)", results):
        mp = intensity.max(axis=0)
        if mp.dtype != np.uint8:
            mp_max = int(mp.max()) or 1
            mp_8 = (mp.astype(np.float32) * (255.0 / mp_max)).clip(0, 255).astype(np.uint8)
        else:
            mp_8 = mp
        img_layer = ImageLayer("intensity", mp_8)

    with _timed("LayerStack: install image layer", results):
        app.layer_stack.set_image(img_layer)
        # Wire a montage document so MontarisApp's 3D dialog can find the
        # volume under doc.image_layer (it pulls the original Z-stack from
        # there for the 3D upload).
        doc = MontageDocument(
            name="GQR183",
            image_layer=img_layer,
            volume_data=intensity,
            volume_axes="ZYX",
        )
        app._documents = [doc]

    # Process events so the window paints before opening the dialog
    qapp.processEvents()

    # --- Stage 3: open the 3D dialog ---
    with _timed("View3D: dialog open + intensity volume build", results):
        app._open_view_3d()
        # Drain deferred timers (constructor schedules QTimer.singleShot(0)
        # → _rebuild_volumes; that's the actual GPU upload).
        for _ in range(40):
            qapp.processEvents()
            time.sleep(0.01)
    panel = app._view3d_panel

    if panel is None:
        print("FATAL: View3DPanel not found after open_3d_view")
        return 3

    # --- Stage 4: GPU detect snapshot ---
    print(f"\n=== GPU auto-detect snapshot ===")
    print(f"  GPU description       : {panel._gpu_desc}")
    print(f"  GL_MAX_3D_TEXTURE_SIZE: {panel._max_3d_dim}")
    print(f"  VRAM budget chosen    : "
          f"{panel._vram_budget // (1024*1024)} MB "
          f"({panel._vram_budget:,} bytes)")
    print(f"  Last total downsample : axes={panel._last_total_ds_axes} "
          f"scalar={panel._last_total_ds}")

    # Probe the raw GL queries to see what each path returned.
    from montaris.widgets.view_3d import (
        _GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX,
        _GPU_MEMORY_INFO_TOTAL_AVAILABLE_VIDMEM_NVX,
        _TEXTURE_FREE_MEMORY_ATI,
        _GL_MAX_3D_TEXTURE_SIZE,
        _query_gl,
        _coerce_gl_int,
    )
    canvas = panel._canvas
    raw_nvx_free = _query_gl(canvas, _GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, 0)
    raw_nvx_total = _query_gl(canvas, _GPU_MEMORY_INFO_TOTAL_AVAILABLE_VIDMEM_NVX, 0)
    raw_ati = _query_gl(canvas, _TEXTURE_FREE_MEMORY_ATI, 0)
    raw_max3d = _query_gl(canvas, _GL_MAX_3D_TEXTURE_SIZE, 0)
    print(f"\n  Raw GL queries (no coercion):")
    print(f"    NVX_AVAILABLE_VIDMEM (free, KB)   : {raw_nvx_free!r}")
    print(f"    NVX_TOTAL_AVAILABLE  (total, KB)  : {raw_nvx_total!r}")
    print(f"    ATI_TEXTURE_FREE_MEMORY           : {raw_ati!r}")
    print(f"    GL_MAX_3D_TEXTURE_SIZE            : {raw_max3d!r}")
    print(f"  After coerce_gl_int:")
    print(f"    NVX free   : {_coerce_gl_int(raw_nvx_free, 0):,} KB "
          f"= {_coerce_gl_int(raw_nvx_free, 0)/1024:.0f} MB")
    print(f"    NVX total  : {_coerce_gl_int(raw_nvx_total, 0):,} KB "
          f"= {_coerce_gl_int(raw_nvx_total, 0)/1024:.0f} MB")

    free_after_mb, _ = _nvidia_smi_free_mb()
    print(f"\n  nvidia-smi after intensity upload: {free_after_mb} MB free")
    print(f"    delta vs before       : "
          f"{(free_before_mb - free_after_mb) if free_after_mb else 'n/a'} MB consumed")

    # --- Stage 5: import labels ---
    print(f"\n=== Labels import + overlay rebuild ===")
    with _timed("Doc: assign labels_3d + auto-build labels_meta", results):
        from montaris.layers import ROI_COLORS
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
                "opacity": 128,
                "visible": True,
                "fill_mode": "solid",
            }
        doc.labels_meta = meta
        panel._primary_doc = doc

    print(f"     labels_meta entries: {len(doc.labels_meta)}, "
          f"max_id={max(doc.labels_meta.keys())}")

    with _timed("View3D: full labels overlay rebuild (slow path)", results):
        panel._rebuild_labels_overlay()
        for _ in range(5):
            qapp.processEvents()

    palette_size = panel._labels_palette_size
    lut_size = panel._labels_texture_lut.size if panel._labels_texture_lut is not None else None
    print(f"     dense palette size: {palette_size} slots "
          f"(LUT entries: {lut_size})")

    free_final_mb, _ = _nvidia_smi_free_mb()
    print(f"  nvidia-smi after labels overlay : {free_final_mb} MB free")

    # --- Summary ---
    print(f"\n=== Summary ===")
    total = sum(dt for _, dt in results)
    print(f"  Total measured wall time: {total:.2f} s")
    for name, dt in results:
        print(f"    {dt*1000:8.0f} ms   {name}")

    # Hold the window open briefly so the user can see the render.
    print(f"\n  Holding window for 5 s so you can confirm the render...")
    deadline = time.time() + 5
    while time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.05)

    app.close()
    qapp.processEvents()
    return 0


if __name__ == "__main__":
    sys.exit(main())
