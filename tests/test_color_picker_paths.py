"""Simulation tests for every ``QColorDialog.getColor`` entry point.

Motivation: a user hit a hang where the native GTK color dialog spawned
at (0,0) with size 100x30 and blocked the Qt event loop. The fix is to
pass ``options=QColorDialog.DontUseNativeDialog`` everywhere. These tests

  1. Verify the flag is actually on every call site so the hang can't
     regress silently.
  2. Exercise happy path (user picks a valid colour) and unhappy path
     (user cancels → invalid QColor) for each picker.
  3. Exercise edge cases that would crash the app: picker invoked with
     no active document, active_doc_index out of range, empty document
     list.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog

from montaris.layers import ImageLayer, MontageDocument
from montaris.widgets.layer_panel import ColorPaletteDialog
from montaris.widgets.properties_panel import PropertiesPanel


def _valid(r, g, b):
    c = QColor(r, g, b)
    assert c.isValid()
    return c


def _invalid():
    """A QColor with isValid() == False (user clicked Cancel)."""
    return QColor()


# ── PropertiesPanel._pick_boundary_color ──────────────────────────────


def test_boundary_color_happy_path_updates_layer_stack(app, qapp):
    panel = PropertiesPanel(app)
    original = app.layer_stack.boundary_color
    with patch.object(QColorDialog, "getColor", return_value=_valid(10, 20, 30)) as gc, \
         patch.object(app.canvas, "refresh_overlays") as refresh:
        panel._pick_boundary_color()
    assert app.layer_stack.boundary_color == (10, 20, 30)
    assert app.layer_stack.boundary_color != original
    refresh.assert_called_once()
    # Flag is what prevents the hang — don't let it regress.
    _, kwargs = gc.call_args
    assert kwargs.get("options") == QColorDialog.DontUseNativeDialog


def test_boundary_color_cancel_leaves_state_untouched(app, qapp):
    panel = PropertiesPanel(app)
    before = app.layer_stack.boundary_color
    with patch.object(QColorDialog, "getColor", return_value=_invalid()), \
         patch.object(app.canvas, "refresh_overlays") as refresh:
        panel._pick_boundary_color()
    assert app.layer_stack.boundary_color == before
    refresh.assert_not_called()


# ── PropertiesPanel._pick_active_color ────────────────────────────────


def test_active_color_happy_path_updates_layer_stack(app, qapp):
    panel = PropertiesPanel(app)
    with patch.object(QColorDialog, "getColor", return_value=_valid(100, 150, 200)) as gc, \
         patch.object(app.canvas, "refresh_overlays"):
        panel._pick_active_color()
    assert app.layer_stack.active_boundary_color == (100, 150, 200)
    _, kwargs = gc.call_args
    assert kwargs.get("options") == QColorDialog.DontUseNativeDialog


def test_active_color_cancel_leaves_state_untouched(app, qapp):
    panel = PropertiesPanel(app)
    before = app.layer_stack.active_boundary_color
    with patch.object(QColorDialog, "getColor", return_value=_invalid()):
        panel._pick_active_color()
    assert app.layer_stack.active_boundary_color == before


# ── MontarisApp._pick_tint_color ──────────────────────────────────────


def _doc(name="ch", shape=(32, 32)):
    arr = np.zeros(shape, dtype=np.uint8)
    return MontageDocument(name=name, image_layer=ImageLayer(name, arr))


def test_pick_tint_color_happy_path(app, qapp):
    doc = _doc()
    app._documents = [doc]
    app._active_doc_index = 0
    with patch.object(QColorDialog, "getColor", return_value=_valid(255, 0, 0)) as gc, \
         patch.object(app.canvas, "set_tint_color"), \
         patch.object(app.canvas, "refresh_image"):
        app._pick_tint_color()
    assert doc.tint_color == (255, 0, 0)
    _, kwargs = gc.call_args
    assert kwargs.get("options") == QColorDialog.DontUseNativeDialog


def test_pick_tint_color_cancel_leaves_doc_untouched(app, qapp):
    doc = _doc()
    doc.tint_color = (5, 5, 5)
    app._documents = [doc]
    app._active_doc_index = 0
    with patch.object(QColorDialog, "getColor", return_value=_invalid()):
        app._pick_tint_color()
    assert doc.tint_color == (5, 5, 5)


def test_pick_tint_color_no_active_doc_returns_early(app, qapp):
    app._documents = []
    app._active_doc_index = -1
    # Must not open a dialog; must not crash.
    with patch.object(QColorDialog, "getColor") as gc:
        app._pick_tint_color()
    gc.assert_not_called()


def test_pick_tint_color_index_out_of_range_returns_early(app, qapp):
    app._documents = [_doc()]
    app._active_doc_index = 99
    with patch.object(QColorDialog, "getColor") as gc:
        app._pick_tint_color()
    gc.assert_not_called()


def test_pick_tint_color_initial_uses_white_when_doc_has_none(app, qapp):
    """A doc with tint_color=None must not make QColor(*None) crash."""
    doc = _doc()
    doc.tint_color = None
    app._documents = [doc]
    app._active_doc_index = 0
    with patch.object(QColorDialog, "getColor", return_value=_invalid()) as gc:
        app._pick_tint_color()
    # First positional arg is the initial QColor; must be the white fallback.
    initial = gc.call_args.args[0]
    assert (initial.red(), initial.green(), initial.blue()) == (255, 255, 255)


# ── MontarisApp._clear_tint_color ─────────────────────────────────────


def test_clear_tint_color_happy_path(app, qapp):
    doc = _doc()
    doc.tint_color = (200, 100, 50)
    app._documents = [doc]
    app._active_doc_index = 0
    with patch.object(app.canvas, "set_tint_color") as set_tint, \
         patch.object(app.canvas, "refresh_image"):
        app._clear_tint_color()
    assert doc.tint_color is None
    set_tint.assert_called_once_with(None)


def test_clear_tint_color_no_active_doc_returns_early(app, qapp):
    app._documents = []
    app._active_doc_index = -1
    # Must not crash on empty documents list.
    app._clear_tint_color()


def test_clear_tint_color_index_out_of_range_returns_early(app, qapp):
    doc = _doc()
    doc.tint_color = (1, 2, 3)
    app._documents = [doc]
    app._active_doc_index = 5
    app._clear_tint_color()
    # Out-of-range index must not touch any document.
    assert doc.tint_color == (1, 2, 3)


# ── ColorPaletteDialog._custom (layer_panel) ──────────────────────────


def test_pick_tint_color_composite_mode_refreshes_composite(app, qapp):
    """In composite mode the tint change must re-run the compositor,
    not the single-channel canvas path. Otherwise the 2D display
    silently keeps the old LUT until the user toggles composite."""
    doc = _doc()
    app._documents = [doc]
    app._active_doc_index = 0
    app._composite_mode = True
    with patch.object(QColorDialog, "getColor", return_value=_valid(12, 34, 56)), \
         patch.object(app, "_refresh_composite") as refresh_comp, \
         patch.object(app.canvas, "refresh_image") as refresh_canvas, \
         patch.object(app.canvas, "set_tint_color") as set_tint:
        app._pick_tint_color()
    assert doc.tint_color == (12, 34, 56)
    refresh_comp.assert_called_once()
    refresh_canvas.assert_not_called()
    set_tint.assert_not_called()


def test_clear_tint_color_composite_mode_refreshes_composite(app, qapp):
    doc = _doc()
    doc.tint_color = (99, 99, 99)
    app._documents = [doc]
    app._active_doc_index = 0
    app._composite_mode = True
    with patch.object(app, "_refresh_composite") as refresh_comp, \
         patch.object(app.canvas, "refresh_image") as refresh_canvas:
        app._clear_tint_color()
    assert doc.tint_color is None
    refresh_comp.assert_called_once()
    refresh_canvas.assert_not_called()


def test_update_tint_btn_survives_none_tint(app, qapp):
    """Clearing tint must leave the button in a valid state."""
    doc = _doc()
    doc.tint_color = None
    app._documents = [doc]
    app._active_doc_index = 0
    # Should not crash — falls back to the toolbar style.
    app._update_tint_btn()


def test_palette_custom_happy_path(qapp):
    dlg = ColorPaletteDialog(current_color=(10, 20, 30))
    with patch.object(QColorDialog, "getColor", return_value=_valid(40, 50, 60)) as gc, \
         patch.object(dlg, "accept") as accept:
        dlg._custom()
    assert dlg.selected_color == (40, 50, 60)
    accept.assert_called_once()
    _, kwargs = gc.call_args
    assert kwargs.get("options") == QColorDialog.DontUseNativeDialog


def test_palette_custom_cancel_leaves_selected_none(qapp):
    dlg = ColorPaletteDialog(current_color=(10, 20, 30))
    with patch.object(QColorDialog, "getColor", return_value=_invalid()), \
         patch.object(dlg, "accept") as accept:
        dlg._custom()
    assert dlg.selected_color is None
    accept.assert_not_called()


# ── View3DPanel per-channel colour swatch ─────────────────────────────


def _view3d_with_channels(n=2):
    """Build a View3DPanel with ``n`` fake channels (no vispy volumes).

    ``_rebuild_volumes`` is deferred via ``QTimer.singleShot`` in init, so we
    only need to ensure we never drive the event loop here. That keeps
    ``self._volumes`` empty; each test injects MagicMock volumes as needed.
    """
    from montaris.widgets import view_3d as v3d
    if not v3d.VISPY_AVAILABLE:
        pytest.skip("vispy not available")
    shape = (4, 6, 8)
    channels = []
    for i in range(n):
        arr = np.zeros(shape, dtype=np.uint8)
        channels.append((f"ch{i}", arr, None))
    with patch.object(v3d.View3DPanel, "_rebuild_volumes", lambda self: None):
        panel = v3d.View3DPanel(channels=channels)
    return panel


def test_channel_swatch_created_per_channel(qapp):
    panel = _view3d_with_channels(3)
    assert len(panel._channel_swatches) == 3
    assert len(panel._channel_toggles) == 3
    # Each swatch has a non-empty background (default palette).
    for sw in panel._channel_swatches:
        assert "background-color:" in sw.styleSheet()
    panel.close()


def test_pick_channel_tint_updates_cmap_and_swatch(qapp):
    panel = _view3d_with_channels(2)
    # Fake a volume object so _pick_channel_tint has something to update.
    fake_vol = MagicMock()
    panel._volumes = [fake_vol, MagicMock()]
    with patch.object(QColorDialog, "getColor",
                      return_value=_valid(255, 128, 0)) as gc:
        panel._pick_channel_tint(0)
    # Stored tint is normalised to 0..1.
    assert panel._channels_raw[0][2] == pytest.approx((1.0, 128 / 255.0, 0.0))
    # Volume cmap was reassigned.
    assert fake_vol.cmap is not None
    # Swatch + checkbox reflect the new colour.
    assert "rgb(255,128,0)" in panel._channel_swatches[0].styleSheet()
    assert "rgb(255,128,0)" in panel._channel_toggles[0].styleSheet()
    # Flag guards against Wayland hang.
    _, kwargs = gc.call_args
    assert kwargs.get("options") == QColorDialog.DontUseNativeDialog
    panel.close()


def test_pick_channel_tint_cancel_keeps_old_colour(qapp):
    panel = _view3d_with_channels(2)
    panel._volumes = [MagicMock(), MagicMock()]
    before = panel._channels_raw[0][2]
    before_swatch = panel._channel_swatches[0].styleSheet()
    with patch.object(QColorDialog, "getColor", return_value=_invalid()):
        panel._pick_channel_tint(0)
    assert panel._channels_raw[0][2] == before
    assert panel._channel_swatches[0].styleSheet() == before_swatch
    panel.close()


def test_pick_channel_tint_out_of_range_is_safe(qapp):
    panel = _view3d_with_channels(1)
    panel._volumes = [MagicMock()]
    with patch.object(QColorDialog, "getColor") as gc:
        panel._pick_channel_tint(99)
    gc.assert_not_called()
    panel.close()


def test_pick_channel_tint_rebuilds_on_cmap_error(qapp):
    """If the vispy visual rejects a live cmap swap, fall back to rebuild."""
    panel = _view3d_with_channels(1)
    bad_vol = MagicMock()
    type(bad_vol).cmap = PropertyMockRaising()
    panel._volumes = [bad_vol]
    with patch.object(QColorDialog, "getColor", return_value=_valid(10, 20, 30)), \
         patch.object(panel, "_rebuild_volumes") as rebuild:
        panel._pick_channel_tint(0)
    rebuild.assert_called_once()
    panel.close()


class PropertyMockRaising:
    def __set__(self, obj, value):
        raise RuntimeError("driver rejected live cmap swap")

    def __get__(self, obj, objtype=None):
        return None


# ── Cross-cutting: DontUseNativeDialog is set at every call site ──────


def _extract_getcolor_calls(text):
    """Yield the argument list of every ``QColorDialog.getColor(...)`` call.

    Hand-rolls paren matching so nested calls like ``QColor(*c)`` don't
    prematurely close the outer call the way a lazy regex does.
    """
    needle = "QColorDialog.getColor("
    i = 0
    while True:
        start = text.find(needle, i)
        if start < 0:
            return
        j = start + len(needle)
        depth = 1
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            j += 1
        yield text[start + len(needle):j - 1]
        i = j


@pytest.mark.parametrize("path", [
    "montaris/widgets/properties_panel.py",
    "montaris/app.py",
    "montaris/widgets/layer_panel.py",
    "montaris/widgets/view_3d.py",
])
def test_every_getColor_call_uses_non_native_flag(path):
    """Static guard: every QColorDialog.getColor call must pass
    DontUseNativeDialog. If someone adds a new picker without the flag,
    they bring back the XWayland hang."""
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent / path
    text = src.read_text()
    calls = list(_extract_getcolor_calls(text))
    assert calls, f"No QColorDialog.getColor call found in {path}"
    for m in calls:
        assert "DontUseNativeDialog" in m, (
            f"QColorDialog.getColor call in {path} is missing "
            f"DontUseNativeDialog — this brings back the Wayland hang.\n"
            f"Call args:\n{m}"
        )
