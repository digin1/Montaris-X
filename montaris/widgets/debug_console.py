import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QLineEdit, QLabel, QPushButton,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from montaris import theme as _theme


class QtLogHandler(logging.Handler):
    """Logging handler that writes to a QTextEdit."""

    def __init__(self, text_widget):
        super().__init__()
        self._widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        self._widget.append(msg)


class DebugConsole(QWidget):
    """Debug console with log output and eval input."""

    def __init__(self, app=None, parent=None):
        super().__init__(parent)
        self._app = app

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        _styles = _theme.debug_console_style()

        self._header = QLabel("Debug Console")
        self._header.setStyleSheet(_styles["header"])
        layout.addWidget(self._header)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(_styles["log"])
        layout.addWidget(self.log_output)

        self._export_btn = QPushButton("Export Diagnostics")
        self._export_btn.setStyleSheet(_styles["button"])
        self._export_btn.clicked.connect(self._export_diagnostics)
        layout.addWidget(self._export_btn)

        self.eval_input = QLineEdit()
        self.eval_input.setPlaceholderText(">>> Enter Python expression...")
        self.eval_input.setStyleSheet(_styles["input"])
        self.eval_input.returnPressed.connect(self._on_eval)
        layout.addWidget(self.eval_input)

        # Setup logging handler
        self._handler = QtLogHandler(self.log_output)
        self._handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S'
        ))
        logging.getLogger().addHandler(self._handler)

    def _on_eval(self):
        expr = self.eval_input.text().strip()
        if not expr:
            return
        self.log_output.append(f">>> {expr}")
        self.eval_input.clear()
        try:
            context = {'app': self._app}
            if self._app:
                context.update({
                    'canvas': self._app.canvas,
                    'layers': self._app.layer_stack,
                })
            result = eval(expr, {"__builtins__": __builtins__}, context)
            if result is not None:
                self.log_output.append(str(result))
        except Exception as e:
            self.log_output.append(f"Error: {e}")

    def _export_diagnostics(self):
        if self._app and hasattr(self._app, '_export_diagnostics'):
            self._app._export_diagnostics()

    def log(self, message):
        self.log_output.append(message)

    def refresh_theme(self):
        _styles = _theme.debug_console_style()
        self._header.setStyleSheet(_styles["header"])
        self.log_output.setStyleSheet(_styles["log"])
        self._export_btn.setStyleSheet(_styles["button"])
        self.eval_input.setStyleSheet(_styles["input"])

    def closeEvent(self, event):
        logging.getLogger().removeHandler(self._handler)
        super().closeEvent(event)
