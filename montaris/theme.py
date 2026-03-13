"""Centralized theme-aware style helpers.

All hardcoded dark/light CSS lives here so widgets stay theme-neutral.
Call ``is_dark()`` to detect the active palette, then use the style
functions to get appropriate CSS strings.
"""

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette


def is_dark():
    """Return True if the current application palette is dark."""
    app = QApplication.instance()
    if app is None:
        return True
    bg = app.palette().color(QPalette.Window)
    # Luminance < 128 → dark
    return (bg.red() * 0.299 + bg.green() * 0.587 + bg.blue() * 0.114) < 128


def section_header_style():
    """CSS for section header labels (category bars, dock title bars)."""
    if is_dark():
        return (
            "QLabel { font-size: 12px; font-weight: bold; color: #ccc;"
            " background: #2a2a2a; border-bottom: 1px solid #444;"
            " border-left: 3px solid #5a9ad5; padding-left: 6px; }"
        )
    return (
        "QLabel { font-size: 12px; font-weight: bold; color: #333;"
        " background: #d8d8d8; border-bottom: 1px solid #bbb;"
        " border-left: 3px solid #4a8abf; padding-left: 6px; }"
    )


def collapse_btn_style():
    """CSS for sidebar collapse/expand header buttons (Toolbox, Layers & Properties)."""
    if is_dark():
        return (
            "QPushButton { border: 1px solid #444; border-radius: 4px;"
            " font-size: 12px; font-weight: bold; background: #383838;"
            " color: #ccc; padding: 0 10px; }"
            "QPushButton:hover { background: #404040; }"
        )
    return (
        "QPushButton { border: 1px solid #bbb; border-radius: 4px;"
        " font-size: 12px; font-weight: bold; background: #d8d8d8;"
        " color: #333; padding: 0 10px; }"
        "QPushButton:hover { background: #c8c8c8; }"
    )


def toolbar_btn_style():
    """CSS for top toolbar buttons (Undo, Redo, Save Progress, etc.)."""
    if is_dark():
        return (
            "QPushButton { background: #3a3a3a; color: #dcdcdc;"
            " border: 1px solid #4a4a4a; border-radius: 3px; padding: 3px 10px; }"
            "QPushButton:hover { background: #4a4a4a; border-color: #5a5a5a; }"
            "QPushButton:pressed { background: #2a2a2a; border-color: #555; }"
        )
    return (
        "QPushButton { background: #e0e0e0; color: #222;"
        " border: 1px solid #ccc; border-radius: 3px; padding: 3px 10px; }"
        "QPushButton:hover { background: #d0d0d0; border-color: #bbb; }"
        "QPushButton:pressed { background: #c0c0c0; border-color: #aaa; }"
    )


def separator_style():
    """CSS for horizontal separator lines."""
    if is_dark():
        return "background: #555; margin: 6px 0;"
    return "background: #bbb; margin: 6px 0;"


def toast_style(border_color):
    """CSS for toast notification frame + label."""
    if is_dark():
        return (
            f"QFrame {{ background: #2a2a2a; border: 2px solid {border_color};"
            f" border-radius: 6px; }}"
            f" QLabel {{ color: #dcdcdc; font-size: 12px; border: none; }}"
        )
    return (
        f"QFrame {{ background: #f5f5f5; border: 2px solid {border_color};"
        f" border-radius: 6px; }}"
        f" QLabel {{ color: #222; font-size: 12px; border: none; }}"
    )


def alert_modal_style():
    """CSS for alert/confirm dialogs."""
    if is_dark():
        return (
            "QDialog { background: #2a2a2a; }"
            " QLabel { color: #dcdcdc; font-size: 13px; }"
            " QPushButton { min-width: 80px; padding: 6px 16px; }"
        )
    return (
        "QDialog { background: #f0f0f0; }"
        " QLabel { color: #222; font-size: 13px; }"
        " QPushButton { min-width: 80px; padding: 6px 16px; }"
    )


def hint_style():
    """CSS for subtle hint labels (move hint, etc.)."""
    if is_dark():
        return "color: #888; font-size: 11px; padding: 0 8px;"
    return "color: #666; font-size: 11px; padding: 0 8px;"


def tool_button_style():
    """CSS for tool panel buttons (Brush, Eraser, Hand, etc.)."""
    if is_dark():
        return (
            "QPushButton { background: #333; color: #ccc; border: 1px solid #444;"
            " border-radius: 4px; padding: 4px 6px; font-size: 12px; }"
            "QPushButton:hover { background: #484848; border-color: #5a8abf; }"
            "QPushButton:checked { background: #2a5a8a; border-color: #5a9ad5;"
            " color: #fff; }"
            "QPushButton:pressed { background: #244d73; }"
        )
    return (
        "QPushButton { background: #e4e4e4; color: #333; border: 1px solid #bbb;"
        " border-radius: 4px; padding: 4px 6px; font-size: 12px; }"
        "QPushButton:hover { background: #c8c8c8; border-color: #4a8abf; }"
        "QPushButton:checked { background: #b8d4ef; border-color: #4a8abf;"
        " color: #1a3a5a; }"
        "QPushButton:pressed { background: #a0c0df; }"
    )


def action_button_style():
    """CSS for Quick Actions buttons (Load Image, Import, Export, etc.)."""
    if is_dark():
        return (
            "QPushButton { background: #2e2e2e; color: #bbb; border: 1px solid #3a3a3a;"
            " border-radius: 4px; padding: 5px 8px; font-size: 12px;"
            " text-align: left; }"
            "QPushButton:hover { background: #3e3e3e; border-color: #5a8abf;"
            " color: #ddd; }"
            "QPushButton:pressed { background: #252525; }"
        )
    return (
        "QPushButton { background: #ececec; color: #444; border: 1px solid #ccc;"
        " border-radius: 4px; padding: 5px 8px; font-size: 12px;"
        " text-align: left; }"
        "QPushButton:hover { background: #d8d8d8; border-color: #4a8abf;"
        " color: #222; }"
        "QPushButton:pressed { background: #d0d0d0; }"
    )


def layer_btn_style():
    """CSS for layer panel buttons (+, -, Dup, Merge)."""
    if is_dark():
        return (
            "QPushButton { background: #333; color: #ccc; border: 1px solid #444;"
            " border-radius: 3px; padding: 3px 8px; }"
            "QPushButton:hover { background: #404040; border-color: #5a8abf; }"
            "QPushButton:pressed { background: #2a2a2a; }"
        )
    return (
        "QPushButton { background: #e4e4e4; color: #333; border: 1px solid #bbb;"
        " border-radius: 3px; padding: 3px 8px; }"
        "QPushButton:hover { background: #d0d0d0; border-color: #4a8abf; }"
        "QPushButton:pressed { background: #c0c0c0; }"
    )


def slider_style():
    """CSS for QSlider (brush size, opacity, adjustments)."""
    if is_dark():
        return (
            "QSlider::groove:horizontal { background: #2a2a2a; height: 4px;"
            " border-radius: 2px; border: 1px solid #3a3a3a; }"
            "QSlider::handle:horizontal { background: #5a9ad5; width: 10px;"
            " height: 10px; margin: -4px 0; border-radius: 5px;"
            " border: 1px solid #4a8abf; }"
            "QSlider::handle:horizontal:hover { background: #6aaaee;"
            " border-color: #5a9ad5; }"
            "QSlider::sub-page:horizontal { background: #3a6a9a;"
            " border-radius: 2px; }"
        )
    return (
        "QSlider::groove:horizontal { background: #d0d0d0; height: 4px;"
        " border-radius: 2px; border: 1px solid #bbb; }"
        "QSlider::handle:horizontal { background: #4a8abf; width: 10px;"
        " height: 10px; margin: -4px 0; border-radius: 5px;"
        " border: 1px solid #3a7aaf; }"
        "QSlider::handle:horizontal:hover { background: #5a9acf;"
        " border-color: #4a8abf; }"
        "QSlider::sub-page:horizontal { background: #7ab0d8;"
        " border-radius: 2px; }"
    )


def spinbox_style():
    """CSS for QSpinBox controls."""
    if is_dark():
        return (
            "QSpinBox { background: #2a2a2a; color: #dcdcdc; border: 1px solid #444;"
            " border-radius: 3px; padding: 1px 4px; }"
            "QSpinBox::up-button, QSpinBox::down-button { width: 14px;"
            " border-radius: 2px; }"
            "QSpinBox::up-button:hover, QSpinBox::down-button:hover"
            " { background: #404040; }"
        )
    return (
        "QSpinBox { background: #fff; color: #222; border: 1px solid #bbb;"
        " border-radius: 3px; padding: 1px 4px; }"
        "QSpinBox::up-button, QSpinBox::down-button { width: 14px;"
        " border-radius: 2px; }"
        "QSpinBox::up-button:hover, QSpinBox::down-button:hover"
        " { background: #ddd; }"
    )


def combobox_style():
    """CSS for QComboBox controls."""
    if is_dark():
        return (
            "QComboBox { background: #2a2a2a; color: #dcdcdc; border: 1px solid #444;"
            " border-radius: 3px; padding: 2px 6px; }"
            "QComboBox:hover { border-color: #5a8abf; }"
            "QComboBox::drop-down { border: none; width: 18px; }"
            "QComboBox QAbstractItemView { background: #2e2e2e; color: #dcdcdc;"
            " selection-background-color: #2a5a8a; border: 1px solid #444; }"
        )
    return (
        "QComboBox { background: #fff; color: #222; border: 1px solid #bbb;"
        " border-radius: 3px; padding: 2px 6px; }"
        "QComboBox:hover { border-color: #4a8abf; }"
        "QComboBox::drop-down { border: none; width: 18px; }"
        "QComboBox QAbstractItemView { background: #fff; color: #222;"
        " selection-background-color: #b8d4ef; border: 1px solid #bbb; }"
    )


def _checkmark_path():
    """Return path to a white checkmark PNG, creating it once on first call."""
    import tempfile, os
    path = os.path.join(tempfile.gettempdir(), "montaris_checkmark.png")
    if not os.path.exists(path):
        from PySide6.QtGui import QImage, QPainter, QPen, QColor
        from PySide6.QtCore import Qt, QPoint
        img = QImage(16, 16, QImage.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 0))
        p = QPainter(img)
        pen = QPen(QColor(255, 255, 255), 2.2)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setRenderHint(QPainter.Antialiasing)
        p.drawLine(QPoint(3, 8), QPoint(6, 11))
        p.drawLine(QPoint(6, 11), QPoint(12, 4))
        p.end()
        img.save(path)
    return path.replace("\\", "/")


def checkbox_style():
    """CSS for QCheckBox controls."""
    tick = _checkmark_path()
    if is_dark():
        return (
            "QCheckBox { spacing: 6px; color: #ccc; }"
            "QCheckBox::indicator { width: 16px; height: 16px;"
            " border-radius: 3px; border: 1px solid #555; background: #2a2a2a; }"
            "QCheckBox::indicator:checked { background: #2a5a8a;"
            f" border-color: #5a9ad5; image: url({tick}); }}"
            "QCheckBox::indicator:hover { border-color: #5a8abf; }"
        )
    return (
        "QCheckBox { spacing: 6px; color: #333; }"
        "QCheckBox::indicator { width: 16px; height: 16px;"
        " border-radius: 3px; border: 1px solid #aaa; background: #fff; }"
        "QCheckBox::indicator:checked { background: #4a8abf;"
        f" border-color: #3a7aaf; image: url({tick}); }}"
        "QCheckBox::indicator:hover { border-color: #4a8abf; }"
    )


def list_widget_style():
    """CSS for QListWidget (ROI list)."""
    if is_dark():
        return (
            "QListWidget { background: #252525; border: 1px solid #3a3a3a;"
            " border-radius: 3px; outline: none; }"
            "QListWidget::item { padding: 5px 6px; border-radius: 2px; }"
            "QListWidget::item:selected { background: #2a5a8a; color: #fff; }"
            "QListWidget::item:hover { background: #333; }"
        )
    return (
        "QListWidget { background: #fff; border: 1px solid #ccc;"
        " border-radius: 3px; outline: none; }"
        "QListWidget::item { padding: 5px 6px; border-radius: 2px; }"
        "QListWidget::item:selected { background: #b8d4ef; color: #1a3a5a; }"
        "QListWidget::item:hover { background: #eee; }"
    )


def groupbox_style():
    """CSS for QGroupBox containers."""
    if is_dark():
        return (
            "QGroupBox { border: 1px solid #3a3a3a; border-radius: 4px;"
            " margin-top: 8px; padding-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin;"
            " subcontrol-position: top left; padding: 0 4px; color: #aaa; }"
        )
    return (
        "QGroupBox { border: 1px solid #ccc; border-radius: 4px;"
        " margin-top: 8px; padding-top: 10px; }"
        "QGroupBox::title { subcontrol-origin: margin;"
        " subcontrol-position: top left; padding: 0 4px; color: #555; }"
    )


def empty_state_style():
    """CSS for the canvas empty-state hint overlay."""
    if is_dark():
        return (
            "QLabel { color: #666; font-size: 15px; background: transparent;"
            " border: 2px dashed #444; border-radius: 12px;"
            " padding: 30px; }"
        )
    return (
        "QLabel { color: #999; font-size: 15px; background: transparent;"
        " border: 2px dashed #bbb; border-radius: 12px;"
        " padding: 30px; }"
    )


def toolbar_group_style():
    """CSS for toolbar control group frames."""
    return "QFrame { background: transparent; border: none; }"


def collapsible_header_style():
    """CSS for collapsible section header buttons."""
    if is_dark():
        return (
            "QPushButton { font-size: 12px; font-weight: bold; color: #ccc;"
            " background: #2a2a2a; border: none; border-bottom: 1px solid #444;"
            " border-left: 3px solid #5a9ad5; padding-left: 6px;"
            " text-align: left; }"
            "QPushButton:hover { background: #333; }"
        )
    return (
        "QPushButton { font-size: 12px; font-weight: bold; color: #333;"
        " background: #d8d8d8; border: none; border-bottom: 1px solid #bbb;"
        " border-left: 3px solid #4a8abf; padding-left: 6px;"
        " text-align: left; }"
        "QPushButton:hover { background: #ccc; }"
    )


def canvas_background():
    """QColor for the canvas/viewport background."""
    from PySide6.QtGui import QColor as _QColor
    if is_dark():
        return _QColor(40, 40, 40)
    return _QColor(200, 200, 200)


def hud_label_style():
    """CSS for the canvas HUD overlay (coordinates, zoom)."""
    if is_dark():
        return (
            "QLabel { color: #ddd; background: rgba(0,0,0,150);"
            " padding: 2px 6px; font-size: 11px; }"
        )
    return (
        "QLabel { color: #222; background: rgba(255,255,255,180);"
        " padding: 2px 6px; font-size: 11px; }"
    )


def perf_label_style():
    """CSS for performance monitor labels."""
    if is_dark():
        return "QLabel { color: #aaa; font-size: 11px; padding: 1px 0; }"
    return "QLabel { color: #555; font-size: 11px; padding: 1px 0; }"


def debug_console_style():
    """CSS for debug console log output and input."""
    if is_dark():
        return {
            "header": "QLabel { font-weight: bold; font-size: 13px; color: #ccc; }",
            "log": "QTextEdit { background: #1a1a1a; color: #d4d4d4;"
                   " border: 1px solid #3a3a3a; border-radius: 3px;"
                   " font-family: Consolas, monospace; font-size: 9pt; }",
            "input": "QLineEdit { background: #1a1a1a; color: #d4d4d4;"
                     " border: 1px solid #3a3a3a; border-radius: 3px;"
                     " font-family: Consolas, monospace; font-size: 9pt;"
                     " padding: 3px 6px; }",
            "button": "QPushButton { background: #333; color: #ccc;"
                      " border: 1px solid #444; border-radius: 3px; padding: 4px 10px; }"
                      "QPushButton:hover { background: #404040; border-color: #5a8abf; }",
        }
    return {
        "header": "QLabel { font-weight: bold; font-size: 13px; color: #333; }",
        "log": "QTextEdit { background: #fff; color: #222;"
               " border: 1px solid #ccc; border-radius: 3px;"
               " font-family: Consolas, monospace; font-size: 9pt; }",
        "input": "QLineEdit { background: #fff; color: #222;"
                 " border: 1px solid #ccc; border-radius: 3px;"
                 " font-family: Consolas, monospace; font-size: 9pt;"
                 " padding: 3px 6px; }",
        "button": "QPushButton { background: #e4e4e4; color: #333;"
                  " border: 1px solid #bbb; border-radius: 3px; padding: 4px 10px; }"
                  "QPushButton:hover { background: #d0d0d0; border-color: #4a8abf; }",
    }


def minimap_colors():
    """Return (background_QColor, viewport_pen_QColor) for the minimap."""
    from PySide6.QtGui import QColor as _QColor
    if is_dark():
        return _QColor(30, 30, 30), _QColor(255, 255, 0)
    return _QColor(210, 210, 210), _QColor(0, 100, 220)


def status_label_style():
    """CSS for status bar labels."""
    if is_dark():
        return "QLabel { color: #aaa; font-size: 11px; padding: 0 8px; }"
    return "QLabel { color: #555; font-size: 11px; padding: 0 8px; }"


def student_label_style():
    """CSS for the Student Session indicator."""
    if is_dark():
        return ("QLabel { font-size: 11px; font-weight: bold;"
                " color: #4ec9b0; padding: 0 8px; }")
    return ("QLabel { font-size: 11px; font-weight: bold;"
            " color: #1a8a6a; padding: 0 8px; }")


def zoom_bar_style():
    """CSS for the floating zoom bar container."""
    if is_dark():
        return ("QWidget { background: rgba(30,30,30,200);"
                " border-radius: 6px; }")
    return ("QWidget { background: rgba(245,245,245,210);"
            " border-radius: 6px; }")


def zoom_bar_button_style():
    """CSS for zoom bar icon buttons."""
    if is_dark():
        return (
            "QPushButton { background: transparent; border: none;"
            " border-radius: 4px; padding: 2px; }"
            "QPushButton:hover { background: rgba(255,255,255,30); }"
            "QPushButton:pressed { background: rgba(255,255,255,50); }"
        )
    return (
        "QPushButton { background: transparent; border: none;"
        " border-radius: 4px; padding: 2px; }"
        "QPushButton:hover { background: rgba(0,0,0,15); }"
        "QPushButton:pressed { background: rgba(0,0,0,30); }"
    )


def zoom_bar_pct_style():
    """CSS for the zoom percentage label button."""
    if is_dark():
        return (
            "QPushButton { background: transparent; border: none;"
            " color: #bbb; font-size: 11px; font-weight: bold;"
            " padding: 2px 4px; }"
            "QPushButton:hover { color: #fff; }"
        )
    return (
        "QPushButton { background: transparent; border: none;"
        " color: #555; font-size: 11px; font-weight: bold;"
        " padding: 2px 4px; }"
        "QPushButton:hover { color: #111; }"
    )
