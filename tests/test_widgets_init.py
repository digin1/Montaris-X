"""Tests for AnimatedButton in montaris.widgets.__init__."""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys
import pytest

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPoint, QEvent
from PySide6.QtGui import QMouseEvent

# Ensure QApplication exists
_qapp = QApplication.instance() or QApplication(sys.argv)

from montaris.widgets import AnimatedButton


class TestAnimatedButtonCreation:
    """Basic construction tests."""

    def test_create_with_text(self):
        btn = AnimatedButton("Click Me")
        assert btn.text() == "Click Me"

    def test_create_empty(self):
        btn = AnimatedButton()
        assert btn.text() == ""

    def test_initial_press_opacity_zero(self):
        btn = AnimatedButton("Test")
        assert btn.press_opacity == 0.0


class TestPressOpacityProperty:
    """Verify the press_opacity property getter/setter."""

    def test_set_and_get_opacity(self):
        btn = AnimatedButton("Test")
        btn._set_press_opacity(0.5)
        assert btn._get_press_opacity() == 0.5

    def test_property_access(self):
        btn = AnimatedButton("Test")
        btn.press_opacity = 0.3
        assert btn.press_opacity == pytest.approx(0.3)


class TestMousePressAnimation:
    """Mouse press should start the animation toward 0.22."""

    def test_mouse_press_starts_animation(self):
        btn = AnimatedButton("Press")
        btn.show()

        event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPoint(5, 5),
            btn.mapToGlobal(QPoint(5, 5)),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
        btn.mousePressEvent(event)

        # After press, animation should be running with end value 0.22
        assert btn._anim.endValue() == pytest.approx(0.22)
        assert btn._anim.duration() == 60
        btn.hide()

    def test_mouse_release_reverses_animation(self):
        btn = AnimatedButton("Release")
        btn.show()

        # Simulate press
        press_event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPoint(5, 5),
            btn.mapToGlobal(QPoint(5, 5)),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
        btn.mousePressEvent(press_event)

        # Manually set opacity as if animation ran
        btn._press_opacity = 0.22

        # Simulate release
        release_event = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPoint(5, 5),
            btn.mapToGlobal(QPoint(5, 5)),
            Qt.LeftButton,
            Qt.NoButton,
            Qt.NoModifier,
        )
        btn.mouseReleaseEvent(release_event)

        # After release, animation should target 0.0 with 150ms duration
        assert btn._anim.endValue() == pytest.approx(0.0)
        assert btn._anim.duration() == 150
        assert btn._anim.startValue() == pytest.approx(0.22)
        btn.hide()


class TestPaintEvent:
    """Paint should overlay white when press_opacity > 0."""

    def test_paint_with_zero_opacity_no_overlay(self):
        """With opacity 0, paintEvent should still succeed without error."""
        btn = AnimatedButton("Paint")
        btn.show()
        btn._press_opacity = 0.0
        # Trigger a repaint; no crash expected
        btn.repaint()
        btn.hide()

    def test_paint_with_nonzero_opacity(self):
        """With nonzero opacity, the overlay branch should execute."""
        btn = AnimatedButton("Paint")
        btn.show()
        btn._press_opacity = 0.15
        # Trigger a repaint; no crash expected
        btn.repaint()
        btn.hide()
