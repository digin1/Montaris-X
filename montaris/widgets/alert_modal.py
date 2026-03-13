from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt
from montaris import theme as _theme


class AlertModal(QDialog):
    def __init__(self, parent=None, title="Alert", message="", buttons=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(300)
        self.setStyleSheet(_theme.alert_modal_style())
        self._clicked = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        for btn_text in (buttons or ["OK"]):
            btn = QPushButton(btn_text)
            btn.clicked.connect(lambda checked, t=btn_text: self._on_clicked(t))
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

    def _on_clicked(self, text):
        self._clicked = text
        self.accept()

    @property
    def clicked_button(self):
        return self._clicked

    @staticmethod
    def error(parent, title, message):
        dlg = AlertModal(parent, title, message, ["OK"])
        dlg.exec()
        return dlg.clicked_button

    @staticmethod
    def warning(parent, title, message):
        dlg = AlertModal(parent, title, message, ["OK"])
        dlg.exec()
        return dlg.clicked_button

    @staticmethod
    def confirm(parent, title, message, buttons=None):
        dlg = AlertModal(parent, title, message, buttons or ["Yes", "No"])
        dlg.exec()
        return dlg.clicked_button
