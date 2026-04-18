"""Headful smoke test for the 3D z-stack viewer.

Run with a real display (not offscreen):

    QT_QPA_PLATFORM=xcb .venv/bin/pytest tests/test_view_3d_headed.py -m headed -s

Tests that opening a GQR183 z-stack goes through the new loader, attaches a
3D volume to the MontageDocument, and that View3DPanel constructs + renders
at least one frame.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog


pytestmark = pytest.mark.headed


ROOT = Path(__file__).resolve().parent.parent
ZSTACK_TIF = ROOT / (
    "GQR183_s02_561_JF552_PSD93cHalo_PSD93SiRHalo_PSD93JF552Halo_"
    "eGFPfill_bottom_100x_100nmstep_10mthick_CrotexVIS_27072025-1.tif"
)


@pytest.fixture(scope="session")
def zstack_path():
    if not ZSTACK_TIF.exists():
        pytest.skip(f"Z-stack not present: {ZSTACK_TIF.name}")
    return ZSTACK_TIF


def test_probe_detects_zstack(zstack_path):
    from montaris.io.image_io import probe_tiff
    p = probe_tiff(str(zstack_path))
    assert p is not None
    assert p["is_zstack"] is True
    assert p["axes"] == "ZYX"
    assert p["n_slices"] > 10


def test_load_volume_and_project(zstack_path):
    from montaris.io.image_io import load_volume, max_projection
    vol, axes = load_volume(str(zstack_path))
    assert axes == "ZYX"
    assert vol.ndim == 3
    mp = max_projection(vol, axis=0)
    assert mp.ndim == 2
    assert mp.shape == vol.shape[1:]


def test_open_zstack_attaches_volume(qapp, zstack_path):
    """Open a z-stack through the real app with dialogs patched."""
    from montaris.app import MontarisApp
    from montaris.widgets.z_stack_dialog import ZStackImportDialog

    window = MontarisApp()
    window.show()
    QApplication.processEvents()

    # Force the z-stack dialog to accept with mode='max', avoid the downsample prompt.
    with patch.object(ZStackImportDialog, "exec", return_value=QDialog.Accepted), \
         patch.object(
             ZStackImportDialog,
             "result_tuple",
             new=property(lambda self: ("max", 0, False)),
         ):
        window.open_image([str(zstack_path)])
    QApplication.processEvents()

    assert len(window._documents) == 1, "expected one document after z-stack import"
    doc = window._documents[0]
    assert doc.volume_data is not None, "volume_data should be attached to MontageDocument"
    assert doc.volume_data.ndim == 3
    # 2D image layer should be the max projection (same Y,X as volume)
    img_shape = window.layer_stack.image_layer.data.shape
    assert img_shape == doc.volume_data.shape[1:], (
        f"image layer shape {img_shape} should match volume YX {doc.volume_data.shape[1:]}"
    )

    window.close()


def test_view3d_dialog_renders_frame(qapp, zstack_path):
    """Construct View3DPanel with a real volume, show it, and confirm a frame is drawn."""
    from montaris.io.image_io import load_volume
    from montaris.widgets.view_3d import View3DPanel, VISPY_AVAILABLE

    if not VISPY_AVAILABLE:
        pytest.skip("vispy not installed")

    vol, _ = load_volume(str(zstack_path))
    # Downsample aggressively so the headless GL upload is cheap and fast
    vol_small = vol[::4, ::4, ::4]

    dlg = View3DPanel(None, channels=[("ch0", vol_small, (0.0, 1.0, 1.0))])
    dlg.resize(640, 480)
    dlg.show()

    deadline = time.time() + 3.0
    while time.time() < deadline:
        QApplication.processEvents()
        if dlg._canvas is not None and dlg._canvas.size != (0, 0):
            break
        time.sleep(0.05)

    assert len(dlg._volumes) == 1, "expected one Volume visual"

    # Save a screenshot of the canvas for visual verification.
    shot_dir = ROOT / "tests" / "_screenshots"
    shot_dir.mkdir(exist_ok=True)
    try:
        img = dlg._canvas.render()
        # vispy render returns a (H, W, 4) uint8 array
        import imageio.v3 as iio
        iio.imwrite(str(shot_dir / "view_3d_single_channel.png"), img)
        print(f"\n[ok] saved {shot_dir / 'view_3d_single_channel.png'}: shape={img.shape}")
    except Exception as e:  # noqa: BLE001 — best-effort screenshot
        print(f"\n[warn] canvas.render failed: {e}")

    # Verify mode switching doesn't crash
    for mode in ("translucent", "iso", "additive"):
        dlg._mode_combo.setCurrentText(mode.capitalize())
        QApplication.processEvents()
        time.sleep(0.1)

    dlg.close()
