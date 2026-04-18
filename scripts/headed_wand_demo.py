"""Headed demo of the 3D magic-wand (Fill) tool.

Launches the full MontarisApp against a real display, imports a GQR183
z-stack, opens the 3D viewer, switches to Fill mode, and seeds the flood
fill at the brightest voxel in the volume so the viewer visibly paints a
coloured overlay over that structure. Saves before/after screenshots and
keeps the window open ~40 seconds so you can watch.

Run with:

    QT_QPA_PLATFORM=wayland .venv/bin/python scripts/headed_wand_demo.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ZSTACK_TIF = ROOT / (
    "GQR183_s02_561_JF552_PSD93cHalo_PSD93SiRHalo_PSD93JF552Halo_"
    "eGFPfill_bottom_100x_100nmstep_10mthick_CrotexVIS_27072025-1.tif"
)
SHOT_DIR = ROOT / "tests" / "_screenshots"
SHOT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[wand] {msg}", flush=True)


def pump(app, seconds):
    """Spin the event loop for the given wall-clock duration."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)


def take_shot(widget, name):
    """Save a QPixmap grab of ``widget`` under tests/_screenshots/<name>."""
    out = SHOT_DIR / name
    widget.grab().save(str(out))
    log(f"screenshot → {out.relative_to(ROOT)}")
    return out


def main() -> int:
    if not ZSTACK_TIF.exists():
        log(f"SKIP: z-stack not at {ZSTACK_TIF}")
        return 0

    log("importing Qt + Montaris...")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from montaris.app import MontarisApp, apply_dark_theme
    from montaris.io.image_io import load_volume, max_projection

    # Load the volume directly from disk and downsample before touching Qt.
    # Going through window.open_image() segfaults on this particular dataset
    # because the full 892 MB tifffile memmap collides with the z-stack
    # dialog's preview render — bypassing that path keeps the demo stable.
    log(f"loading volume: {ZSTACK_TIF.name}")
    t0 = time.time()
    vol_full, _ = load_volume(str(ZSTACK_TIF))
    log(f"  full shape={vol_full.shape} dtype={vol_full.dtype} "
        f"({vol_full.nbytes/1024/1024:.0f} MB) in {time.time()-t0:.2f}s")
    vol = vol_full[::2, ::2, ::2]  # 2x each axis → ~1/8 the voxels
    mp = max_projection(vol, axis=0)
    log(f"  demo volume {vol.shape}, max-proj {mp.shape}")

    app = QApplication.instance() or QApplication(sys.argv)
    apply_dark_theme(app)

    window = MontarisApp()
    window.resize(1400, 900)
    window.show()
    pump(app, 0.3)
    log("main window shown")

    # Wire the volume in through _load_single_channel — same entry point the
    # real z-stack loader uses after the user accepts the projection dialog.
    log("attaching volume to document via _load_single_channel...")
    window._load_single_channel(
        name=ZSTACK_TIF.stem, data=mp, ds_factor=1, skipped=[],
        image_path=str(ZSTACK_TIF), volume_data=vol,
    )
    pump(app, 0.5)
    doc = window._documents[window._active_doc_index]
    assert doc.volume_data is not None, "volume_data did not attach"
    log(f"document loaded: name={doc.name!r} vol.shape={doc.volume_data.shape}")

    take_shot(window, "wand_01_loaded.png")

    # Open the 3D viewer. _open_view_3d handles building the panel, wiring
    # the label_added signal, and swapping the central stack to the 3D page.
    log("opening 3D view...")
    window._open_view_3d()
    pump(app, 2.5)  # GPU upload for GQR183 stacks is noticeable
    panel = window._view3d_panel
    assert panel is not None, "3D panel not attached to window"
    log(f"  3D panel active: volumes={len(panel._volumes)}")
    pump(app, 1.0)
    take_shot(window, "wand_02_3d_open.png")

    # Switch to Fill (the "magic wand"). The tool combo lives inside the
    # panel; changing the current text fires _on_tool_changed.
    log("activating Fill tool...")
    panel._tool_combo.setCurrentText("Fill")
    panel._tol_slider.setValue(15)  # ±15% of the channel's dynamic range
    pump(app, 0.5)

    # Pick a seed inside an actual structure. Plain argmax often lands on
    # a saturated cosmic-ray voxel whose tolerance window contains only
    # itself (tiny fill). The 99.5th percentile is almost always inside a
    # neurite for GQR183-style stacks, and its connected component covers
    # a meaningful region without flooding the whole background.
    vol = panel._active_channel_volume()
    target = float(np.percentile(vol, 99.5))
    hits = np.argwhere(vol >= target)
    seed = tuple(int(v) for v in hits[len(hits) // 2])
    log(f"  seed voxel (z,y,x) = {seed}, intensity={int(vol[seed])} "
        f"(target p99.5={target:.0f})")

    # Run the fill. _run_fill writes voxels into doc.labels_3d, refreshes
    # the overlay, and emits label_added — MontarisApp listens and adds a
    # VolumeROILayer to the LayerPanel on the right.
    log("running flood fill → writing labels_3d + adding ROI...")
    panel._run_fill(seed)
    pump(app, 1.0)
    assert doc.labels_3d is not None, "labels_3d should have been allocated"
    filled_voxels = int((doc.labels_3d > 0).sum())
    log(f"  filled voxels: {filled_voxels:,}")
    log(f"  labels_meta: {list(doc.labels_meta.items())}")
    take_shot(window, "wand_03_after_fill.png")

    # Second seed: another bright voxel at a different spatial location so
    # the two resulting ROIs are visually separable. We look for the top-
    # intensity voxel in the half of the volume the first seed wasn't in.
    try:
        Z = vol.shape[0]
        half_lo, half_hi = (Z // 2, Z) if seed[0] < Z // 2 else (0, Z // 2)
        sub = vol[half_lo:half_hi]
        target2 = float(np.percentile(sub, 99.5))
        hits2 = np.argwhere(sub >= target2)
        if len(hits2):
            sub_seed = hits2[len(hits2) // 2]
            seed2 = (int(sub_seed[0] + half_lo),
                     int(sub_seed[1]), int(sub_seed[2]))
            log(f"  seed voxel 2 (z,y,x) = {seed2}, "
                f"intensity={int(vol[seed2])}")
            panel._run_fill(seed2)
            pump(app, 1.0)
            filled_voxels2 = int((doc.labels_3d > 0).sum())
            log(f"  total filled voxels after 2nd wand: {filled_voxels2:,}")
            log(f"  labels now: {[(k, v['name'], v['color']) for k, v in doc.labels_meta.items()]}")
            log(f"  LayerPanel rows: {len(window.layer_stack.roi_layers)}")
            take_shot(window, "wand_04_after_second_fill.png")
    except Exception as e:  # noqa: BLE001 — demo, not critical
        log(f"  second fill skipped: {e}")

    # Hold the window open so the viewer is visible and interactive.
    hold_seconds = int(os.environ.get("WAND_HOLD", "40"))
    log(f"holding window open for {hold_seconds}s — interact freely, "
        "then close it or wait for auto-exit")
    pump(app, hold_seconds)

    window.close()
    QTimer.singleShot(50, app.quit)
    app.exec()
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
