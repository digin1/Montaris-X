"""Extended tests for montaris.widgets.debug_console.DebugConsole."""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys
import logging
import pytest
from unittest.mock import MagicMock, patch

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QEvent
from PySide6.QtGui import QCloseEvent

# Ensure QApplication exists
_qapp = QApplication.instance() or QApplication(sys.argv)

from montaris.widgets.debug_console import DebugConsole, QtLogHandler


class TestOnEvalValid:
    """_on_eval() with a valid expression should show the result."""

    def test_eval_simple_expression(self):
        console = DebugConsole(app=None)
        console.eval_input.setText("2 + 3")
        console._on_eval()

        output = console.log_output.toPlainText()
        assert ">>> 2 + 3" in output
        assert "5" in output

    def test_eval_string_expression(self):
        console = DebugConsole(app=None)
        console.eval_input.setText("'hello'.upper()")
        console._on_eval()

        output = console.log_output.toPlainText()
        assert "HELLO" in output

    def test_eval_none_result_no_extra_output(self):
        """When eval returns None, no extra result line should appear."""
        console = DebugConsole(app=None)
        console.eval_input.setText("None")
        console._on_eval()

        output = console.log_output.toPlainText()
        lines = [l for l in output.strip().split("\n") if l.strip()]
        # Should only have the ">>> None" line, no result line
        assert len(lines) == 1
        assert ">>> None" in lines[0]

    def test_eval_clears_input(self):
        console = DebugConsole(app=None)
        console.eval_input.setText("1+1")
        console._on_eval()
        assert console.eval_input.text() == ""

    def test_eval_empty_input_ignored(self):
        console = DebugConsole(app=None)
        console.eval_input.setText("   ")
        console._on_eval()
        assert console.log_output.toPlainText() == ""


class TestOnEvalException:
    """_on_eval() with an invalid expression should show the error."""

    def test_eval_syntax_error(self):
        console = DebugConsole(app=None)
        console.eval_input.setText("1 / 0")
        console._on_eval()

        output = console.log_output.toPlainText()
        assert "Error:" in output
        assert "division by zero" in output

    def test_eval_name_error(self):
        console = DebugConsole(app=None)
        console.eval_input.setText("undefined_var")
        console._on_eval()

        output = console.log_output.toPlainText()
        assert "Error:" in output


class TestOnEvalWithApp:
    """_on_eval() with an app context provides canvas/layers in scope."""

    def test_eval_app_context(self):
        mock_app = MagicMock()
        mock_app.canvas = "canvas_mock"
        mock_app.layer_stack = "layers_mock"
        console = DebugConsole(app=mock_app)

        console.eval_input.setText("type(canvas).__name__")
        console._on_eval()

        output = console.log_output.toPlainText()
        assert "str" in output


class TestExportDiagnostics:
    """_export_diagnostics() delegates to app._export_diagnostics."""

    def test_export_delegates_to_app(self):
        mock_app = MagicMock()
        console = DebugConsole(app=mock_app)
        console._export_diagnostics()
        mock_app._export_diagnostics.assert_called_once()

    def test_export_no_app(self):
        """No crash when app is None."""
        console = DebugConsole(app=None)
        console._export_diagnostics()  # Should not raise

    def test_export_app_without_method(self):
        """No crash when app lacks _export_diagnostics."""
        mock_app = MagicMock(spec=[])
        console = DebugConsole(app=mock_app)
        console._export_diagnostics()  # Should not raise


class TestCloseEvent:
    """closeEvent() should remove the logging handler."""

    def test_close_removes_handler(self):
        console = DebugConsole(app=None)
        handler = console._handler
        root_logger = logging.getLogger()

        assert handler in root_logger.handlers

        event = QCloseEvent()
        console.closeEvent(event)

        assert handler not in root_logger.handlers

        # Re-add a no-op check: the console's widget may be deleted after
        # this test, so we must ensure the handler is gone from root logger.
        # (Already verified above.)


class TestRefreshTheme:
    """refresh_theme() should apply updated styles without error."""

    def test_refresh_theme_applies_styles(self):
        console = DebugConsole(app=None)
        # Should not raise
        console.refresh_theme()

        # Verify widgets have non-empty stylesheets
        assert console._header.styleSheet() != ""
        assert console.log_output.styleSheet() != ""
        assert console._export_btn.styleSheet() != ""
        assert console.eval_input.styleSheet() != ""


class TestLogMethod:
    """The log() convenience method should append text."""

    def test_log_appends_message(self):
        console = DebugConsole(app=None)
        console.log("test message")
        assert "test message" in console.log_output.toPlainText()


class TestQtLogHandler:
    """QtLogHandler should forward log records to the text widget."""

    def test_handler_emits_to_widget(self):
        from PySide6.QtWidgets import QTextEdit
        widget = QTextEdit()
        handler = QtLogHandler(widget)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger = logging.getLogger("test_debug_console_handler_isolated")
        logger.propagate = False  # Prevent propagation to root (avoids dead widget handlers)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("handler test msg")

        output = widget.toPlainText()
        assert "handler test msg" in output

        logger.removeHandler(handler)
