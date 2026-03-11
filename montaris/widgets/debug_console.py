import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QLineEdit, QLabel, QPushButton,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


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

        header = QLabel("Debug Console")
        header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(header)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Monospace", 9))
        self.log_output.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        layout.addWidget(self.log_output)

        export_btn = QPushButton("Export Diagnostics")
        export_btn.clicked.connect(self._export_diagnostics)
        layout.addWidget(export_btn)

        self.eval_input = QLineEdit()
        self.eval_input.setPlaceholderText(">>> Enter Python expression...")
        self.eval_input.setFont(QFont("Monospace", 9))
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

    def closeEvent(self, event):
        logging.getLogger().removeHandler(self._handler)
        super().closeEvent(event)
