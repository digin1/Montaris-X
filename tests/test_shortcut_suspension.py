"""Verify 2D ToolPanel shortcuts are cleared while the 3D viewer owns focus.

The 2D tool buttons set shortcuts with the default ``Qt.WindowShortcut`` scope
(fires window-wide), while ``View3DPanel`` installs ``QShortcut`` objects at
``Qt.WidgetWithChildrenShortcut`` scope. V/E/P/R overlap between the two, so
when the 3D panel is mounted in the central stack, Qt sees both scopes match
and reports ambiguity. ``ToolPanel.suspend_tool_shortcuts`` clears the 2D
entries on entry; ``restore_tool_shortcuts`` reinstates them on exit.
"""
from __future__ import annotations

import pytest
from PySide6.QtGui import QKeySequence

from montaris.tools import TOOL_REGISTRY


CONFLICTING = ("V", "E", "P", "R")


def _shortcut_map(tool_panel):
    return {
        name: btn.shortcut().toString()
        for name, btn in tool_panel._tool_buttons.items()
    }


def test_initial_state_has_registry_shortcuts(qapp, app):
    mapping = _shortcut_map(app.tool_panel)
    for name, (_mod, _cls, shortcut, _cat) in TOOL_REGISTRY.items():
        assert mapping.get(name) == QKeySequence(shortcut).toString(), (
            f"{name!r} should start with shortcut {shortcut!r}"
        )


def test_suspend_clears_all_tool_shortcuts(qapp, app):
    app.tool_panel.suspend_tool_shortcuts()
    for name, btn in app.tool_panel._tool_buttons.items():
        assert btn.shortcut().isEmpty(), (
            f"{name!r} still has shortcut after suspend"
        )


def test_restore_reinstates_shortcuts(qapp, app):
    before = _shortcut_map(app.tool_panel)
    app.tool_panel.suspend_tool_shortcuts()
    app.tool_panel.restore_tool_shortcuts()
    after = _shortcut_map(app.tool_panel)
    assert before == after


def test_suspend_is_idempotent(qapp, app):
    app.tool_panel.suspend_tool_shortcuts()
    # A second call must not overwrite the saved copy with empty shortcuts.
    app.tool_panel.suspend_tool_shortcuts()
    app.tool_panel.restore_tool_shortcuts()
    mapping = _shortcut_map(app.tool_panel)
    for name, (_mod, _cls, shortcut, _cat) in TOOL_REGISTRY.items():
        assert mapping.get(name) == QKeySequence(shortcut).toString()


def test_restore_without_suspend_is_noop(qapp, app):
    before = _shortcut_map(app.tool_panel)
    app.tool_panel.restore_tool_shortcuts()
    assert _shortcut_map(app.tool_panel) == before


def test_conflicting_keys_freed_when_suspended(qapp, app):
    """After suspend, the keys the 3D viewer wants are no longer owned by 2D."""
    app.tool_panel.suspend_tool_shortcuts()
    cleared = {
        btn.shortcut().toString().upper()
        for btn in app.tool_panel._tool_buttons.values()
    }
    for key in CONFLICTING:
        assert key not in cleared, (
            f"Key {key!r} still owned by a 2D tool button after suspend"
        )


def _entering_3d_suspends_and_leaving_restores(app, monkeypatch):
    """Helper: toggle view_3d without needing a real vispy canvas."""
    import montaris.widgets.view_3d as v3d
    from PySide6.QtWidgets import QWidget

    class _DummySignal:
        def connect(self, _fn):
            pass

    class _FakePanel(QWidget):
        label_added = _DummySignal()
        undo_pushed = _DummySignal()

        def __init__(self, parent=None, **_k):
            super().__init__(parent)

        def capture_state(self):
            return None

        def apply_state(self, _s):
            pass

        def release_gl(self):
            pass

        def set_active_volume_roi_id(self, _i):
            pass

    monkeypatch.setattr(v3d, "VISPY_AVAILABLE", True)
    monkeypatch.setattr(v3d, "channels_from_documents", lambda _docs: [("ch0", object())])
    monkeypatch.setattr(v3d, "View3DPanel", _FakePanel)


def test_roundtrip_through_3d_toggle(qapp, app, monkeypatch):
    """Happy path: enter 3D clears shortcuts, leave 3D brings them back."""
    _entering_3d_suspends_and_leaving_restores(app, monkeypatch)
    before = _shortcut_map(app.tool_panel)

    app._open_view_3d()
    assert app._view3d_panel is not None
    for btn in app.tool_panel._tool_buttons.values():
        assert btn.shortcut().isEmpty()

    app._open_view_3d()
    assert app._view3d_panel is None
    assert _shortcut_map(app.tool_panel) == before
