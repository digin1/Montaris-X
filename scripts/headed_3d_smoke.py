"""Headful smoke test for the 3D viewer.

Loads one of the GQR183 z-stacks via the new loader, constructs View3DPanel
directly, forces a render, iterates render modes, saves screenshots, exits.

    QT_QPA_PLATFORM=wayland .venv/bin/python scripts/headed_3d_smoke.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ZSTACK_TIF = ROOT / (
    "GQR183_s02_561_JF552_PSD93cHalo_PSD93SiRHalo_PSD93JF552Halo_"
    "eGFPfill_bottom_100x_100nmstep_10mthick_CrotexVIS_27072025-1.tif"
)
SHOT_DIR = ROOT / "tests" / "_screenshots"
SHOT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[smoke] {msg}", flush=True)


def main() -> int:
    if not ZSTACK_TIF.exists():
        log(f"SKIP: z-stack not at {ZSTACK_TIF}")
        return 0

    log("importing Qt + vispy...")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from montaris.io.image_io import load_volume, probe_tiff
    from montaris.widgets.view_3d import View3DPanel, VISPY_AVAILABLE

    if not VISPY_AVAILABLE:
        log("FAIL: vispy not available")
        return 1

    log(f"probing {ZSTACK_TIF.name}...")
    p = probe_tiff(str(ZSTACK_TIF))
    log(f"  probe: {p}")

    log("loading volume...")
    t0 = time.time()
    vol, _ = load_volume(str(ZSTACK_TIF))
    log(f"  shape={vol.shape} dtype={vol.dtype} in {time.time()-t0:.2f}s")

    # Downsample 4x in each axis to keep GPU upload small for the headed test.
    vol_small = vol[::4, ::4, ::4]
    log(f"  downsampled to {vol_small.shape} ({vol_small.nbytes/1024/1024:.1f} MB)")

    log("creating QApplication...")
    app = QApplication.instance() or QApplication(sys.argv)

    log("constructing View3DPanel...")
    dlg = View3DPanel(None, channels=[("561", vol_small, (0.0, 1.0, 1.0))])
    dlg.resize(960, 720)
    dlg.show()
    app.processEvents()
    log(f"  dialog shown; volumes={len(dlg._volumes)}")
    if dlg._volumes:
        v = dlg._volumes[0]
        log(f"  volume clim={v.clim}, threshold={v.threshold}, method={v.method}")
        log(f"  data stats: min={vol_small.min()}, max={vol_small.max()}, mean={float(vol_small.mean()):.1f}")

    # Let the event loop run a bit so the GL context is fully initialized.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)

    results = {}
    try:
        import imageio.v3 as iio
        # Iterate every label in the dropdown so the smoke test follows the UI.
        for i in range(dlg._mode_combo.count()):
            mode = dlg._mode_combo.itemText(i)
            dlg._mode_combo.setCurrentIndex(i)
            app.processEvents()
            time.sleep(0.3)
            img = dlg._canvas.render()
            slug = mode.lower().replace(' ', '_')
            out = SHOT_DIR / f"view_3d_{slug}.png"
            iio.imwrite(str(out), img)
            nonzero = int((img[..., :3].sum(axis=-1) > 0).sum())
            results[mode] = (img.shape, nonzero)
            log(f"  {mode}: {img.shape}, nonzero_pixels={nonzero} → {out.name}")
    except Exception as e:
        log(f"WARN: render loop failed: {e}")
        return 2

    # Final grab of the whole dialog for UI verification.
    dlg_shot = SHOT_DIR / "view_3d_dialog_ui.png"
    dlg.grab().save(str(dlg_shot))
    log(f"saved dialog grab → {dlg_shot.name}")

    dlg.close()
    QTimer.singleShot(50, app.quit)
    app.exec()

    # Basic assertion: at least one mode produced non-black pixels.
    if not any(n > 1000 for (_, n) in results.values()):
        log("FAIL: all renders were black / near-black")
        return 3
    log("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
