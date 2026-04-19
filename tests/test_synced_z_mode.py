"""Synced z-stack mode — one channel per file, Z slider scrubs in lockstep.

Covers:
  - Docs in the same ``z_sync_group`` all advance when the slider moves.
  - Slider range is the minimum Z-depth across the group.
  - Non-composite checkbox: 2+ checked auto-enables composite;
    exactly 1 checked switches the active doc.
  - Display Settings auto-expands once per session on multi-channel load.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from montaris.layers import ImageLayer, MontageDocument


def _add_synced_doc(app, name, shape, sync_group):
    """Append a z-stack doc to the app with the given sync group id."""
    z, h, w = shape
    vol = np.zeros(shape, dtype=np.uint8)
    for i in range(z):
        vol[i] = (i + 1) * 10
    layer = ImageLayer(name, vol[0].copy())
    if app.layer_stack.image_layer is None:
        app.layer_stack.set_image(layer)
    else:
        app.layer_stack.image_layer = layer
    doc = MontageDocument(
        name=name,
        image_layer=layer,
        volume_data=vol,
        volume_axes="ZYX",
        z_sync_group=sync_group,
    )
    app._documents.append(doc)
    app._active_doc_index = len(app._documents) - 1
    app._doc_combo.addItem(name)
    app._doc_combo.setCurrentIndex(app._active_doc_index)
    app._update_display_channels()
    app._update_z_slider_visibility()
    QApplication.processEvents()
    return doc


def test_synced_group_scrubs_in_lockstep(app):
    """Moving the slider updates ``active_z`` on every doc in the group."""
    a = _add_synced_doc(app, "chA", (8, 20, 20), sync_group=1)
    b = _add_synced_doc(app, "chB", (8, 20, 20), sync_group=1)
    c = _add_synced_doc(app, "chC", (8, 20, 20), sync_group=1)

    app._z_slider.setValue(5)
    QApplication.processEvents()
    assert a.active_z == 5
    assert b.active_z == 5
    assert c.active_z == 5


def test_synced_group_range_uses_min_depth(app):
    """Unequal stack depths — slider range clamps to the shortest."""
    _add_synced_doc(app, "chA", (10, 16, 16), sync_group=2)
    _add_synced_doc(app, "chB", (6, 16, 16), sync_group=2)

    assert app._z_slider.maximum() == 5  # min(10, 6) - 1


def test_synced_group_image_layers_advance(app):
    """Each sibling doc's image_layer.data reflects the new slice."""
    a = _add_synced_doc(app, "chA", (5, 8, 8), sync_group=3)
    b = _add_synced_doc(app, "chB", (5, 8, 8), sync_group=3)

    app._z_slider.setValue(3)
    QApplication.processEvents()
    np.testing.assert_array_equal(a.image_layer.data, a.volume_data[3])
    np.testing.assert_array_equal(b.image_layer.data, b.volume_data[3])


def test_independent_docs_do_not_co_scrub(app):
    """``z_sync_group=None`` docs scrub alone even when others are loaded."""
    solo = _add_synced_doc(app, "solo", (4, 8, 8), sync_group=None)
    other = _add_synced_doc(app, "other", (4, 8, 8), sync_group=None)

    # Activate 'solo' then scrub — 'other' must not move.
    app._active_doc_index = 0
    app._update_z_slider_visibility()
    app._z_slider.setValue(2)
    QApplication.processEvents()
    assert solo.active_z == 2
    assert other.active_z == 0


def test_two_channels_checked_auto_enables_composite(app):
    _add_synced_doc(app, "chA", (3, 8, 8), sync_group=4)
    _add_synced_doc(app, "chB", (3, 8, 8), sync_group=4)

    # Start from composite off with only chA checked, then turn chB back on.
    app.display_panel.composite_cb.setChecked(False)
    cbs = app.display_panel._channel_checkboxes
    cbs[1].setChecked(False)
    QApplication.processEvents()
    assert not app.display_panel.composite_cb.isChecked()

    cbs[1].setChecked(True)  # second channel — should trip auto-composite
    QApplication.processEvents()
    assert app.display_panel.composite_cb.isChecked()


def test_one_channel_checked_switches_active_doc(app):
    _add_synced_doc(app, "chA", (3, 8, 8), sync_group=5)
    _add_synced_doc(app, "chB", (3, 8, 8), sync_group=5)

    # Start with composite off and both unchecked except chA.
    app.display_panel.composite_cb.setChecked(False)
    cbs = app.display_panel._channel_checkboxes
    cbs[0].setChecked(False)
    cbs[1].setChecked(False)
    QApplication.processEvents()

    cbs[0].setChecked(True)  # only chA checked
    QApplication.processEvents()
    assert app._active_doc_index == 0
    assert not app.display_panel.composite_cb.isChecked()


def test_display_settings_auto_expands_once(app):
    """First multi-channel load expands the panel; a later load does not
    re-expand after the user has collapsed it."""
    assert app._display_settings_auto_expanded is False
    _add_synced_doc(app, "chA", (3, 8, 8), sync_group=6)
    _add_synced_doc(app, "chB", (3, 8, 8), sync_group=6)
    assert app._display_settings_auto_expanded is True
    assert app._display_settings_panel.isVisible() is True

    # User collapses it and loads another channel — should stay collapsed.
    app._display_settings_panel.setVisible(False)
    _add_synced_doc(app, "chC", (3, 8, 8), sync_group=6)
    assert app._display_settings_panel.isVisible() is False


def test_mint_z_sync_group_id_is_unique(app):
    """Freshly-minted ids never collide with existing docs."""
    _add_synced_doc(app, "chA", (3, 8, 8), sync_group=1)
    _add_synced_doc(app, "chB", (3, 8, 8), sync_group=7)

    new_id = app._mint_z_sync_group_id()
    assert new_id == 8  # max(1, 7) + 1
