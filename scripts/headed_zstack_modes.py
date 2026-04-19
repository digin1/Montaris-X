"""Headed end-to-end exercise of the z-stack import modes.

Runs a single mode (passed on argv) through a live MontarisApp, loads the
three GQR183 TIFFs as a batch, captures a screenshot of the main window,
and reports timing/memory. Intended to be invoked once per mode by
``scripts/run_headed_zstack_modes.py`` so each run starts with a clean
process (892 MB TIFFs × 3 modes would otherwise peak RAM).

Usage:
    .venv/bin/python scripts/headed_zstack_modes.py {max|slice|synced}
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QDialog

from montaris.app import MontarisApp
from montaris.widgets.z_stack_dialog import ZStackImportDialog
from montaris.io.image_io import probe_tiff

SCREENSHOT_DIR = ROOT / "tests" / "_screenshots" / "zstack_modes"
TIFFS = sorted(
    str(p) for p in ROOT.iterdir()
    if p.name.startswith("GQR183") and p.suffix == ".tif"
)
MODES = {"max", "slice", "synced"}


def _pump(ms=50):
    deadline = time.monotonic() + ms / 1000.0
    while time.monotonic() < deadline:
        QApplication.processEvents()


def _sample_cpu(proc, duration_s=1.0):
    """Return peak CPU% sampled over ``duration_s`` while pumping events."""
    peak = 0.0
    proc.cpu_percent(None)  # prime
    start = time.monotonic()
    while time.monotonic() - start < duration_s:
        _pump(50)
        v = proc.cpu_percent(None)
        if v > peak:
            peak = v
    return peak


def main(mode: str) -> dict:
    assert mode in MODES, f"unknown mode: {mode}"
    assert len(TIFFS) == 3, f"expected 3 TIFFs, found {len(TIFFS)}"
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    dialog_mode = mode
    tiffs = TIFFS

    # Force xcb so we get a real window (QT_QPA_PLATFORM=offscreen would work
    # but wouldn't exercise the paint path).
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    proc = psutil.Process(os.getpid())
    mem_start = proc.memory_info().rss / 1024**2
    t_total = time.monotonic()

    qapp = QApplication.instance() or QApplication(sys.argv)
    window = MontarisApp()
    window.resize(1400, 900)
    window.show()
    _pump(300)

    slice_idx = 66  # middle slice for 'slice' mode

    # Patch the dialog so the headed run is non-interactive.
    def fake_exec(self):
        return QDialog.Accepted

    def fake_result(self):
        return (dialog_mode, slice_idx, True)  # apply_to_batch=True

    t_load = time.monotonic()
    with patch.object(ZStackImportDialog, "exec", fake_exec), \
         patch.object(ZStackImportDialog, "result_tuple", property(fake_result)):
        window.open_image(tiffs)
    _pump(300)
    load_s = time.monotonic() - t_load

    # Probe live state.
    n_docs = len(window._documents)
    docs_with_volume = sum(
        1 for d in window._documents
        if getattr(d, "volume_data", None) is not None
    )
    sync_groups = sorted({
        getattr(d, "z_sync_group", None) for d in window._documents
    })
    slider_visible = window._z_bar.isVisible()
    slider_max = window._z_slider.maximum()
    composite_on = window.display_panel.composite_cb.isChecked()
    display_panel_visible = window._display_settings_panel.isVisible()
    display_auto_expanded = window._display_settings_auto_expanded

    # Drive the Z slider to exercise the scrub path (synced/slice/all).
    z_scrub_ok = None
    if slider_visible and slider_max > 0:
        z_scrub_ok = True
        for z in (0, slider_max // 2, slider_max):
            window._z_slider.setValue(z)
            _pump(80)
            if window._active_document().active_z != z:
                z_scrub_ok = False
                break

    # Composite render — only meaningful when >1 doc and a shared Z.
    if n_docs >= 2 and mode in ("max", "slice", "synced"):
        window.display_panel.composite_cb.setChecked(True)
        _pump(200)
        composite_on = window.display_panel.composite_cb.isChecked()

    # Measure peak CPU over a 1s window while the UI is interactive.
    cpu_peak = _sample_cpu(proc, 1.0)
    mem_peak = proc.memory_info().rss / 1024**2

    # Trigger a toast so the screenshot captures the styling.
    window.toast.show(f"Loaded {n_docs} channel(s) in {mode} mode", "success")
    _pump(250)

    # Screenshot the whole main window.
    shot_path = SCREENSHOT_DIR / f"{mode}.png"
    pix: QPixmap = window.grab()
    pix.save(str(shot_path))

    # Also screenshot the desktop area so the floating toast is captured.
    try:
        from PySide6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            g = window.geometry()
            full = screen.grabWindow(0, g.x(), g.y(), g.width(), g.height())
            full.save(str(SCREENSHOT_DIR / f"{mode}_with_toast.png"))
    except Exception:
        pass

    total_s = time.monotonic() - t_total

    result = {
        "mode": mode,
        "n_tiffs": len(tiffs),
        "n_docs": n_docs,
        "docs_with_volume": docs_with_volume,
        "sync_groups": [g for g in sync_groups if g is not None],
        "slider_visible": slider_visible,
        "slider_max": slider_max,
        "z_scrub_ok": z_scrub_ok,
        "composite_on": composite_on,
        "display_panel_visible": display_panel_visible,
        "display_auto_expanded": display_auto_expanded,
        "load_seconds": round(load_s, 2),
        "total_seconds": round(total_s, 2),
        "mem_start_mb": round(mem_start, 1),
        "mem_peak_mb": round(mem_peak, 1),
        "cpu_peak_pct": round(cpu_peak, 1),
        "screenshot": str(shot_path.relative_to(ROOT)),
    }

    window.close()
    _pump(100)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: headed_zstack_modes.py {max|slice|synced}", file=sys.stderr)
        sys.exit(2)
    out = main(sys.argv[1])
    print("RESULT::" + json.dumps(out))
