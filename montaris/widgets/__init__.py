from PySide6.QtWidgets import QPushButton
from PySide6.QtCore import QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QPainter, QColor


class AnimatedButton(QPushButton):
    """QPushButton with a fast smooth press/release brightness animation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._press_opacity = 0.0
        self._anim = QPropertyAnimation(self, b"press_opacity")
        self._anim.setDuration(120)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def _get_press_opacity(self):
        return self._press_opacity

    def _set_press_opacity(self, v):
        self._press_opacity = v
        self.update()

    press_opacity = Property(float, _get_press_opacity, _set_press_opacity)

    def mousePressEvent(self, e):
        self._anim.stop()
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(0.22)
        self._anim.setDuration(60)
        self._anim.start()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        self._anim.stop()
        self._anim.setStartValue(self._press_opacity)
        self._anim.setEndValue(0.0)
        self._anim.setDuration(150)
        self._anim.start()
        super().mouseReleaseEvent(e)

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._press_opacity > 0.001:
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            c = QColor(255, 255, 255, int(self._press_opacity * 255))
            p.fillRect(self.rect(), c)
            p.end()
