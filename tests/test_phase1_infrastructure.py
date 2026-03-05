"""Tests for Phase 1: Toast, Alert, Undo Enhancements."""
import pytest
import numpy as np
from PySide6.QtCore import Qt

from montaris.core.undo import UndoCommand, UndoStack
from montaris.core.multi_undo import CompoundUndoCommand, SnapshotUndoCommand
from montaris.layers import ROILayer


class TestUndoReturnsCommand:
    def test_undo_returns_command(self):
        stack = UndoStack()
        layer = ROILayer("test", 10, 10)
        layer.mask[2:4, 2:4] = 255
        cmd = UndoCommand(layer, (2, 4, 2, 4),
                          np.zeros((2, 2), dtype=np.uint8),
                          layer.mask[2:4, 2:4])
        stack.push(cmd)
        result = stack.undo()
        assert result is cmd

    def test_undo_returns_none_when_empty(self):
        stack = UndoStack()
        result = stack.undo()
        assert result is None

    def test_redo_returns_command(self):
        stack = UndoStack()
        layer = ROILayer("test", 10, 10)
        layer.mask[2:4, 2:4] = 255
        cmd = UndoCommand(layer, (2, 4, 2, 4),
                          np.zeros((2, 2), dtype=np.uint8),
                          layer.mask[2:4, 2:4])
        stack.push(cmd)
        stack.undo()
        result = stack.redo()
        assert result is cmd

    def test_redo_returns_none_when_empty(self):
        stack = UndoStack()
        result = stack.redo()
        assert result is None


class TestCompoundUndoCommand:
    def test_roi_layer_property(self):
        layer = ROILayer("test", 10, 10)
        cmd1 = UndoCommand(layer, (0, 1, 0, 1), np.zeros((1, 1), np.uint8), np.ones((1, 1), np.uint8) * 255)
        compound = CompoundUndoCommand([cmd1])
        assert compound.roi_layer is layer

    def test_roi_layer_empty(self):
        compound = CompoundUndoCommand([])
        assert compound.roi_layer is None


class TestSnapshotUndoCommand:
    def test_snapshot_undo_redo(self):
        layer = ROILayer("test", 10, 10)
        old_mask = layer.mask.copy()
        layer.mask[3:5, 3:5] = 255
        cmd = SnapshotUndoCommand([(layer, old_mask)])
        layer.mask[3:5, 3:5] = 100  # further change
        cmd.undo()
        assert layer.mask[3, 3] == 0
        cmd.redo()
        assert layer.mask[3, 3] == 255

    def test_roi_layer_property(self):
        layer = ROILayer("test", 10, 10)
        cmd = SnapshotUndoCommand([(layer, layer.mask.copy())])
        assert cmd.roi_layer is layer


class TestToast:
    def test_toast_manager_creation(self, qapp, app):
        assert hasattr(app, 'toast')
        assert app.toast is not None

    def test_toast_show(self, qapp, app):
        app.toast.show("Test message", "info", 100)
        assert len(app.toast._toasts) == 1


class TestAlertModal:
    def test_alert_modal_creation(self, qapp):
        from montaris.widgets.alert_modal import AlertModal
        dlg = AlertModal(None, "Test", "Message", ["OK"])
        assert dlg._clicked is None


class TestAutoSelectOnUndo:
    def test_undo_auto_selects_roi(self, qapp, app_with_image):
        app = app_with_image
        roi = app.layer_stack.roi_layers[0]
        roi.mask[5:10, 5:10] = 255
        from montaris.core.undo import UndoCommand
        cmd = UndoCommand(roi, (5, 10, 5, 10),
                          np.zeros((5, 5), np.uint8),
                          roi.mask[5:10, 5:10])
        app.undo_stack.push(cmd)
        app.undo()
        assert app.canvas._active_layer is roi
