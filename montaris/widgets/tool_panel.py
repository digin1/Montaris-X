from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QSlider, QSpinBox,
    QButtonGroup, QHBoxLayout,
)
from PySide6.QtCore import Signal, Qt
from montaris.tools import TOOL_REGISTRY, get_tool_class
from montaris.tools.polygon import PolygonTool


class ToolPanel(QWidget):
    tool_changed = Signal(object)

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self._tool_buttons = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Tools")
        header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(header)

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)

        # Build tool buttons from registry, grouped by category
        categories = {}
        for name, (module, cls_name, shortcut, category) in TOOL_REGISTRY.items():
            categories.setdefault(category, []).append((name, shortcut))

        for category, tools in categories.items():
            cat_label = QLabel(category)
            cat_label.setStyleSheet(
                "font-weight: bold; font-size: 11px; margin-top: 6px;"
            )
            layout.addWidget(cat_label)
            for name, shortcut in tools:
                btn = self._add_tool_button(name, shortcut, layout)
                self._tool_buttons[name] = btn

        layout.addSpacing(10)

        size_label = QLabel("Brush Size")
        layout.addWidget(size_label)

        size_layout = QHBoxLayout()
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 100)
        self.size_slider.setValue(10)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        size_layout.addWidget(self.size_slider)

        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 100)
        self.size_spin.setValue(10)
        self.size_spin.valueChanged.connect(self.size_slider.setValue)
        self.size_slider.valueChanged.connect(self.size_spin.setValue)
        size_layout.addWidget(self.size_spin)

        layout.addLayout(size_layout)

        self.finish_polygon_btn = QPushButton("Close Polygon (Enter)")
        self.finish_polygon_btn.clicked.connect(self._finish_polygon)
        self.finish_polygon_btn.setVisible(False)
        layout.addWidget(self.finish_polygon_btn)

        layout.addStretch()

        self._current_tool = None

        # Default to Hand tool on startup
        if 'Hand' in self._tool_buttons:
            self._tool_buttons['Hand'].setChecked(True)
            self._select_tool('Hand')

    def _add_tool_button(self, text, shortcut, layout):
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setShortcut(shortcut)
        btn.setToolTip(f"{text} ({shortcut})")
        self.tool_group.addButton(btn)
        btn.clicked.connect(lambda checked, t=text: self._select_tool(t))
        layout.addWidget(btn)
        return btn

    def _select_tool(self, tool_name):
        if tool_name not in TOOL_REGISTRY:
            return

        tool_cls = get_tool_class(tool_name)
        tool = tool_cls(self.app)

        # Apply current size to tools that support it
        if hasattr(tool, 'size'):
            tool.size = self.size_slider.value()

        # Show/hide polygon finish button
        self.finish_polygon_btn.setVisible(tool_name == "Polygon")

        self._current_tool = tool
        self.tool_changed.emit(tool)

    def _on_size_changed(self, value):
        if self._current_tool and hasattr(self._current_tool, 'size'):
            self._current_tool.size = value

    def _finish_polygon(self):
        if self._current_tool and isinstance(self._current_tool, PolygonTool):
            self._current_tool.finish()
