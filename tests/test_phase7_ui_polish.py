"""Tests for Phase 7: UI Polish."""
import pytest


class TestHelpModal:
    def test_help_modal_creation(self, qapp):
        from montaris.widgets.help_modal import HelpModal
        dlg = HelpModal(None)
        assert dlg.windowTitle() == "Montaris-X User Guide"

    def test_help_modal_has_tabs(self, qapp):
        from montaris.widgets.help_modal import HelpModal
        from PySide6.QtWidgets import QTabWidget
        dlg = HelpModal(None)
        tabs = dlg.findChild(QTabWidget)
        assert tabs is not None
        assert tabs.count() == 5


class TestToolStatusWidget:
    def test_tool_status_exists(self, qapp, app):
        assert hasattr(app, '_tool_status_label')
        assert app._tool_status_label is not None

    def test_tool_status_updates_on_tool_change(self, qapp, app):
        app.tool_panel._select_tool('Brush')
        assert 'Brush' in app._tool_status_label.text()

    def test_tool_status_updates_on_layer_select(self, qapp, app_with_image):
        app = app_with_image
        roi = app.layer_stack.roi_layers[0]
        app._on_layer_selected(roi)
        assert roi.name in app._tool_status_label.text()


class TestFlipRotateOnLoad:
    def test_flip_on_load_action_exists(self, qapp, app):
        assert hasattr(app, '_flip_on_load_act')
        assert app._flip_on_load_act.isCheckable()
        assert not app._flip_on_load_act.isChecked()

    def test_rotate_on_load_action_exists(self, qapp, app):
        assert hasattr(app, '_rotate_on_load_act')
        assert app._rotate_on_load_act.isCheckable()
        assert not app._rotate_on_load_act.isChecked()


class TestColorPaletteDialog:
    def test_creation(self, qapp):
        from montaris.widgets.layer_panel import ColorPaletteDialog
        dlg = ColorPaletteDialog((255, 0, 0))
        assert dlg.selected_color is None


class TestROINavBar:
    def test_creation(self, qapp):
        from montaris.widgets.layer_panel import ROINavBar
        bar = ROINavBar()
        bar.set_segments([(0.5, (255, 0, 0), 0), (0.5, (0, 255, 0), 1)])
        bar.resize(200, 20)
