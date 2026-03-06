from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QSlider, QSpinBox,
    QButtonGroup, QHBoxLayout, QStackedWidget, QSizePolicy,
)
from PySide6.QtCore import Signal, Qt
from montaris.tools import TOOL_REGISTRY, get_tool_class
from montaris.tools.polygon import PolygonTool

# Tool icons: Unicode symbols for collapsed icon strip
TOOL_ICONS = {
    'Hand': '✋',
    'Select': '⬚',
    'Brush': '✎',
    'Eraser': '⌫',
    'Polygon': '⬠',
    'Bucket Fill': '▼',
    'Rectangle': '▭',
    'Circle': '○',
    'Stamp': '■',
    'Transform': '⟲',
    'Move': '✥',
}


class ToolPanel(QWidget):
    tool_changed = Signal(object)

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self._tool_buttons = {}
        self._icon_buttons = {}
        self._collapsed = False

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Toggle button at the top
        self._toggle_btn = QPushButton("◀")
        self._toggle_btn.setFixedHeight(22)
        self._toggle_btn.setToolTip("Collapse/Expand tools panel")
        self._toggle_btn.clicked.connect(self.toggle_collapsed)
        self._toggle_btn.setStyleSheet(
            "QPushButton { border: none; font-size: 12px; padding: 2px; }"
        )
        root_layout.addWidget(self._toggle_btn)

        # Stacked widget: page 0 = expanded, page 1 = collapsed icon strip
        self._stack = QStackedWidget()
        root_layout.addWidget(self._stack)

        # --- Expanded view ---
        expanded = QWidget()
        layout = QVBoxLayout(expanded)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(False)

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

        # Brush size (C.5: range 1-500)
        size_label = QLabel("Brush Size")
        layout.addWidget(size_label)

        size_layout = QHBoxLayout()
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 500)
        self.size_slider.setValue(10)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        size_layout.addWidget(self.size_slider)

        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 500)
        self.size_spin.setValue(10)
        self.size_spin.valueChanged.connect(self.size_slider.setValue)
        self.size_slider.valueChanged.connect(self.size_spin.setValue)
        size_layout.addWidget(self.size_spin)

        layout.addLayout(size_layout)

        # Tolerance slider (C.10) — shown when BucketFill active
        self.tolerance_label = QLabel("Fill Tolerance")
        self.tolerance_label.setVisible(False)
        layout.addWidget(self.tolerance_label)

        tol_layout = QHBoxLayout()
        self.tolerance_slider = QSlider(Qt.Horizontal)
        self.tolerance_slider.setRange(0, 255)
        self.tolerance_slider.setValue(0)
        self.tolerance_slider.valueChanged.connect(self._on_tolerance_changed)
        self.tolerance_slider.setVisible(False)
        tol_layout.addWidget(self.tolerance_slider)

        self.tolerance_spin = QSpinBox()
        self.tolerance_spin.setRange(0, 255)
        self.tolerance_spin.setValue(0)
        self.tolerance_spin.valueChanged.connect(self.tolerance_slider.setValue)
        self.tolerance_slider.valueChanged.connect(self.tolerance_spin.setValue)
        self.tolerance_spin.setVisible(False)
        tol_layout.addWidget(self.tolerance_spin)
        layout.addLayout(tol_layout)

        # Stamp W/H inputs — shown when Stamp active
        self.stamp_label = QLabel("Stamp Size")
        self.stamp_label.setVisible(False)
        layout.addWidget(self.stamp_label)

        stamp_layout = QHBoxLayout()
        stamp_layout.setContentsMargins(0, 0, 0, 0)
        stamp_w_label = QLabel("W:")
        stamp_w_label.setVisible(False)
        self._stamp_w_label = stamp_w_label
        stamp_layout.addWidget(stamp_w_label)

        self.stamp_w_spin = QSpinBox()
        self.stamp_w_spin.setRange(1, 1000)
        self.stamp_w_spin.setValue(20)
        self.stamp_w_spin.setSuffix(" px")
        self.stamp_w_spin.setVisible(False)
        self.stamp_w_spin.valueChanged.connect(self._on_stamp_size_changed)
        stamp_layout.addWidget(self.stamp_w_spin)

        stamp_h_label = QLabel("H:")
        stamp_h_label.setVisible(False)
        self._stamp_h_label = stamp_h_label
        stamp_layout.addWidget(stamp_h_label)

        self.stamp_h_spin = QSpinBox()
        self.stamp_h_spin.setRange(1, 1000)
        self.stamp_h_spin.setValue(20)
        self.stamp_h_spin.setSuffix(" px")
        self.stamp_h_spin.setVisible(False)
        self.stamp_h_spin.valueChanged.connect(self._on_stamp_size_changed)
        stamp_layout.addWidget(self.stamp_h_spin)

        layout.addLayout(stamp_layout)

        self.finish_polygon_btn = QPushButton("Close Polygon (Enter)")
        self.finish_polygon_btn.clicked.connect(self._finish_polygon)
        self.finish_polygon_btn.setVisible(False)
        layout.addWidget(self.finish_polygon_btn)

        # Deselect button (C.23)
        self.deselect_btn = QPushButton("Deselect Tool")
        self.deselect_btn.setToolTip("Deselect current tool")
        self.deselect_btn.clicked.connect(self._deselect_tool)
        layout.addWidget(self.deselect_btn)

        layout.addStretch()
        self._stack.addWidget(expanded)

        # --- Collapsed icon strip ---
        collapsed = QWidget()
        icon_layout = QVBoxLayout(collapsed)
        icon_layout.setContentsMargins(2, 2, 2, 2)
        icon_layout.setSpacing(2)

        for name in TOOL_REGISTRY:
            shortcut = TOOL_REGISTRY[name][2]
            icon = TOOL_ICONS.get(name, shortcut)
            ibtn = QPushButton(icon)
            ibtn.setCheckable(True)
            ibtn.setFixedSize(32, 32)
            ibtn.setToolTip(f"{name} [{shortcut}]")
            ibtn.setStyleSheet(
                "QPushButton { font-size: 16px; padding: 0; }"
                "QPushButton:checked { background-color: #507cc8; }"
            )
            ibtn.clicked.connect(lambda checked, t=name: self._select_tool(t))
            icon_layout.addWidget(ibtn)
            self._icon_buttons[name] = ibtn

        icon_layout.addStretch()
        self._stack.addWidget(collapsed)

        self._current_tool = None
        self._current_tool_name = None

        # Default to Hand tool on startup
        if 'Hand' in self._tool_buttons:
            self._tool_buttons['Hand'].setChecked(True)
            self._select_tool('Hand')

    def _add_tool_button(self, text, shortcut, layout):
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setShortcut(shortcut)
        btn.setToolTip(f"{text} [{shortcut}]")
        self.tool_group.addButton(btn)
        btn.clicked.connect(lambda checked, t=text: self._select_tool(t))
        layout.addWidget(btn)
        return btn

    def toggle_collapsed(self):
        self._collapsed = not self._collapsed
        self._stack.setCurrentIndex(1 if self._collapsed else 0)
        self._toggle_btn.setText("▶" if self._collapsed else "◀")
        # Sync icon button states
        for name, ibtn in self._icon_buttons.items():
            ibtn.setChecked(name == self._current_tool_name)
        # Adjust parent dock width
        dock = self.parent()
        if dock and hasattr(dock, 'setFixedWidth'):
            if self._collapsed:
                dock.setFixedWidth(44)
            else:
                dock.setMinimumWidth(0)
                dock.setMaximumWidth(16777215)

    def _select_tool(self, tool_name):
        if tool_name not in TOOL_REGISTRY:
            return

        # Uncheck other buttons (manual exclusive)
        for name, btn in self._tool_buttons.items():
            btn.setChecked(name == tool_name)
        for name, ibtn in self._icon_buttons.items():
            ibtn.setChecked(name == tool_name)

        tool_cls = get_tool_class(tool_name)
        tool = tool_cls(self.app)

        # Apply current size to tools that support it
        if hasattr(tool, 'size'):
            tool.size = self.size_slider.value()
        # Apply stamp dimensions
        if hasattr(tool, 'width') and hasattr(tool, 'height'):
            tool.width = self.stamp_w_spin.value()
            tool.height = self.stamp_h_spin.value()
        # Apply tolerance
        if hasattr(tool, 'tolerance'):
            tool.tolerance = self.tolerance_slider.value()

        # Show/hide context-specific controls
        self.finish_polygon_btn.setVisible(tool_name == "Polygon")
        is_bucket = tool_name == "Bucket Fill"
        self.tolerance_label.setVisible(is_bucket)
        self.tolerance_slider.setVisible(is_bucket)
        self.tolerance_spin.setVisible(is_bucket)
        is_stamp = tool_name == "Stamp"
        self.stamp_label.setVisible(is_stamp)
        self._stamp_w_label.setVisible(is_stamp)
        self.stamp_w_spin.setVisible(is_stamp)
        self._stamp_h_label.setVisible(is_stamp)
        self.stamp_h_spin.setVisible(is_stamp)

        self._current_tool = tool
        self._current_tool_name = tool_name
        self.tool_changed.emit(tool)

    def _deselect_tool(self):
        """Deselect all tools (C.23)."""
        for btn in self._tool_buttons.values():
            btn.setChecked(False)
        for ibtn in self._icon_buttons.values():
            ibtn.setChecked(False)
        self._current_tool = None
        self._current_tool_name = None
        self.tool_changed.emit(None)

    def _on_size_changed(self, value):
        if self._current_tool and hasattr(self._current_tool, 'size'):
            self._current_tool.size = value

    def _on_tolerance_changed(self, value):
        if self._current_tool and hasattr(self._current_tool, 'tolerance'):
            self._current_tool.tolerance = value

    def _on_stamp_size_changed(self):
        if self._current_tool and hasattr(self._current_tool, 'width'):
            self._current_tool.width = self.stamp_w_spin.value()
            self._current_tool.height = self.stamp_h_spin.value()

    def _finish_polygon(self):
        if self._current_tool and isinstance(self._current_tool, PolygonTool):
            self._current_tool.finish()
