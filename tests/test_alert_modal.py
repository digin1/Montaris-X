"""Tests for montaris.widgets.alert_modal.AlertModal."""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys
import pytest
from unittest.mock import patch, MagicMock

from PySide6.QtWidgets import QApplication, QPushButton

# Ensure QApplication exists
_qapp = QApplication.instance() or QApplication(sys.argv)

from montaris.widgets.alert_modal import AlertModal


class TestAlertModalCreation:
    """Basic construction and property tests."""

    def test_create_with_title_and_message(self):
        dlg = AlertModal(title="Test Title", message="Hello world")
        assert dlg.windowTitle() == "Test Title"

    def test_default_ok_button(self):
        """When no buttons are specified, a single OK button should be created."""
        dlg = AlertModal(title="Info", message="msg")
        buttons = dlg.findChildren(QPushButton)
        assert len(buttons) == 1
        assert buttons[0].text() == "OK"

    def test_custom_buttons(self):
        dlg = AlertModal(title="Pick", message="Choose", buttons=["Yes", "No", "Cancel"])
        buttons = dlg.findChildren(QPushButton)
        texts = sorted(b.text() for b in buttons)
        assert texts == ["Cancel", "No", "Yes"]

    def test_clicked_button_initially_none(self):
        dlg = AlertModal(title="X", message="Y")
        assert dlg.clicked_button is None

    def test_on_clicked_sets_value_and_accepts(self):
        dlg = AlertModal(title="X", message="Y", buttons=["A", "B"])
        dlg._on_clicked("B")
        assert dlg.clicked_button == "B"

    def test_minimum_width(self):
        dlg = AlertModal(title="T", message="M")
        assert dlg.minimumWidth() >= 300


class TestStaticMethods:
    """Static helper methods: error, warning, confirm."""

    @patch.object(AlertModal, "exec", return_value=0)
    def test_error_creates_and_execs(self, mock_exec):
        result = AlertModal.error(None, "Error Title", "Something broke")
        mock_exec.assert_called_once()
        # Return value is clicked_button which is None since we didn't click
        assert result is None

    @patch.object(AlertModal, "exec", return_value=0)
    def test_warning_creates_and_execs(self, mock_exec):
        result = AlertModal.warning(None, "Warning", "Be careful")
        mock_exec.assert_called_once()

    @patch.object(AlertModal, "exec", return_value=0)
    def test_confirm_default_buttons(self, mock_exec):
        result = AlertModal.confirm(None, "Confirm", "Are you sure?")
        mock_exec.assert_called_once()

    @patch.object(AlertModal, "exec", return_value=0)
    def test_confirm_custom_buttons(self, mock_exec):
        result = AlertModal.confirm(None, "Save?", "Save changes?",
                                    buttons=["Save", "Discard", "Cancel"])
        mock_exec.assert_called_once()
