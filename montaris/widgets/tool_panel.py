from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QSlider, QSpinBox,
    QButtonGroup, QHBoxLayout, QDialog, QFormLayout, QDialogButtonBox,
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
        self.tool_group.setExclusive(False)

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

        # Stamp settings (C.19) — shown when Stamp active
        self.stamp_settings_btn = QPushButton("Stamp Settings...")
        self.stamp_settings_btn.setVisible(False)
        self.stamp_settings_btn.clicked.connect(self._open_stamp_settings)
        layout.addWidget(self.stamp_settings_btn)

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

    def _select_tool(self, tool_name):
        if tool_name not in TOOL_REGISTRY:
            return

        # Uncheck other buttons (manual exclusive)
        for name, btn in self._tool_buttons.items():
            btn.setChecked(name == tool_name)

        tool_cls = get_tool_class(tool_name)
        tool = tool_cls(self.app)

        # Apply current size to tools that support it
        if hasattr(tool, 'size'):
            tool.size = self.size_slider.value()
        # Apply stamp dimensions
        if hasattr(tool, 'width') and hasattr(tool, 'height'):
            tool.width = getattr(self, '_stamp_width', 20)
            tool.height = getattr(self, '_stamp_height', 20)
        # Apply tolerance
        if hasattr(tool, 'tolerance'):
            tool.tolerance = self.tolerance_slider.value()

        # Show/hide context-specific controls
        self.finish_polygon_btn.setVisible(tool_name == "Polygon")
        is_bucket = tool_name == "Bucket Fill"
        self.tolerance_label.setVisible(is_bucket)
        self.tolerance_slider.setVisible(is_bucket)
        self.tolerance_spin.setVisible(is_bucket)
        self.stamp_settings_btn.setVisible(tool_name == "Stamp")

        self._current_tool = tool
        self._current_tool_name = tool_name
        self.tool_changed.emit(tool)

    def _deselect_tool(self):
        """Deselect all tools (C.23)."""
        for btn in self._tool_buttons.values():
            btn.setChecked(False)
        self._current_tool = None
        self._current_tool_name = None
        self.tool_changed.emit(None)

    def _on_size_changed(self, value):
        if self._current_tool and hasattr(self._current_tool, 'size'):
            self._current_tool.size = value

    def _on_tolerance_changed(self, value):
        if self._current_tool and hasattr(self._current_tool, 'tolerance'):
            self._current_tool.tolerance = value

    def _open_stamp_settings(self):
        """Open stamp W/H settings dialog (C.19)."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Stamp Settings")
        form = QFormLayout(dlg)
        w_spin = QSpinBox()
        w_spin.setRange(1, 1000)
        w_spin.setValue(getattr(self, '_stamp_width', 20))
        h_spin = QSpinBox()
        h_spin.setRange(1, 1000)
        h_spin.setValue(getattr(self, '_stamp_height', 20))
        form.addRow("Width:", w_spin)
        form.addRow("Height:", h_spin)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec():
            self._stamp_width = w_spin.value()
            self._stamp_height = h_spin.value()
            if self._current_tool and hasattr(self._current_tool, 'width'):
                self._current_tool.width = self._stamp_width
                self._current_tool.height = self._stamp_height

    def _finish_polygon(self):
        if self._current_tool and isinstance(self._current_tool, PolygonTool):
            self._current_tool.finish()
