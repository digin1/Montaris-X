from PySide6.QtWidgets import QFrame, QLabel, QHBoxLayout
from PySide6.QtCore import Qt, QTimer, QObject, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QColor


_LEVEL_COLORS = {
    "success": "#2d7d46",
    "error": "#b5332a",
    "warning": "#c77d1a",
    "info": "#2a6fb5",
}


class ToastNotification(QFrame):
    def __init__(self, message, level="info", parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setFixedHeight(36)

        color = _LEVEL_COLORS.get(level, _LEVEL_COLORS["info"])
        self.setStyleSheet(
            f"QFrame {{ background: #2a2a2a; border: 2px solid {color};"
            f" border-radius: 6px; }}"
            f" QLabel {{ color: #dcdcdc; font-size: 12px; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        self._label = QLabel(message)
        layout.addWidget(self._label)
        self.adjustSize()

    def mousePressEvent(self, event):
        self.close()


class ToastManager(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._toasts = []
        self._parent_widget = parent

    def show(self, message, level="info", timeout_ms=3000):
        toast = ToastNotification(message, level, self._parent_widget)
        self._toasts.append(toast)
        self._layout_toasts()
        toast.show()

        timer = QTimer(toast)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self._dismiss(toast))
        timer.start(timeout_ms)

    def _dismiss(self, toast):
        if toast in self._toasts:
            self._toasts.remove(toast)
            toast.close()
            toast.deleteLater()
            self._layout_toasts()

    def _layout_toasts(self):
        if self._parent_widget is None:
            return
        pw = self._parent_widget
        margin = 10
        y_offset = margin
        for toast in reversed(self._toasts):
            toast.setFixedWidth(min(350, pw.width() - 2 * margin))
            x = pw.width() - toast.width() - margin
            y = pw.height() - y_offset - toast.height()
            toast.move(pw.mapToGlobal(pw.rect().topLeft()).x() + x,
                       pw.mapToGlobal(pw.rect().topLeft()).y() + y)
            y_offset += toast.height() + 4
