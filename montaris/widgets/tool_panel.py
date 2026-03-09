from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QSlider, QSpinBox,
    QButtonGroup, QHBoxLayout,
)
from PySide6.QtCore import Signal, Qt, QTimer
from montaris.tools import TOOL_REGISTRY, get_tool_class
from montaris.tools.polygon import PolygonTool

# Tool icons: emojis matching the web app (threejsroieditor)
TOOL_ICONS = {
    'Hand': '🖐️',
    'Select': '🖱️',
    'Brush': '🖌️',
    'Eraser': '🧹',
    'Polygon': '⬟',
    'Bucket Fill': '🪣',
    'Rectangle': '🔲',
    'Circle': '⬤',
    'Stamp': '■',
    'Transform': '🔧',
    'Move': '✥',
}


class ToolPanel(QWidget):
    tool_changed = Signal(object)
    collapse_requested = Signal()
    open_montage_requested = Signal()
    import_roi_zip_requested = Signal()
    export_roi_zip_requested = Signal()
    load_instructions_requested = Signal()
    view_instructions_requested = Signal()

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self._tool_buttons = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Collapse button at top
        collapse_btn = QPushButton("\u25c0")  # ◀
        collapse_btn.setFixedHeight(22)
        collapse_btn.setToolTip("Collapse sidebar (Ctrl+[)")
        collapse_btn.setStyleSheet(
            "QPushButton { border: none; font-size: 12px; padding: 2px; }"
        )
        collapse_btn.clicked.connect(self.collapse_requested.emit)
        layout.addWidget(collapse_btn, alignment=Qt.AlignLeft)

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
            # Lay out buttons in rows of 2
            for i in range(0, len(tools), 2):
                pair = tools[i:i + 2]
                if len(pair) == 2:
                    row = QHBoxLayout()
                    row.setContentsMargins(0, 0, 0, 0)
                    row.setSpacing(4)
                    for name, shortcut in pair:
                        btn = self._add_tool_button(name, shortcut)
                        self._tool_buttons[name] = btn
                        row.addWidget(btn)
                    layout.addLayout(row)
                else:
                    name, shortcut = pair[0]
                    btn = self._add_tool_button(name, shortcut)
                    self._tool_buttons[name] = btn
                    layout.addWidget(btn)

        layout.addSpacing(10)

        # Brush size — hidden slider used as internal state (UI in toolbar)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 2000)
        self.size_slider.setValue(100)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        self.size_slider.setVisible(False)
        layout.addWidget(self.size_slider)

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

        # --- File upload buttons ---
        layout.addSpacing(10)
        file_label = QLabel("File")
        file_label.setStyleSheet(
            "font-weight: bold; font-size: 11px; margin-top: 6px;"
        )
        layout.addWidget(file_label)

        montage_btn = QPushButton("🖼️  Load Image(s)")
        montage_btn.setToolTip("Open montage or image file(s)")
        montage_btn.clicked.connect(lambda: QTimer.singleShot(0, self.open_montage_requested.emit))
        layout.addWidget(montage_btn)

        roi_zip_btn = QPushButton("📦  Import ROI ZIP")
        roi_zip_btn.setToolTip("Import ROIs from a ZIP file")
        roi_zip_btn.clicked.connect(lambda: QTimer.singleShot(0, self.import_roi_zip_requested.emit))
        layout.addWidget(roi_zip_btn)

        instr_btn = QPushButton("📝  Load Instructions")
        instr_btn.setToolTip("Load instructions file (.json/.txt)")
        instr_btn.clicked.connect(lambda: QTimer.singleShot(0, self.load_instructions_requested.emit))
        layout.addWidget(instr_btn)

        view_instr_btn = QPushButton("👁️  View Instructions")
        view_instr_btn.setToolTip("View loaded instructions")
        view_instr_btn.clicked.connect(lambda: QTimer.singleShot(0, self.view_instructions_requested.emit))
        layout.addWidget(view_instr_btn)

        # --- Separator + Export ---
        separator = QLabel()
        separator.setFixedHeight(2)
        separator.setStyleSheet("background: #555; margin: 6px 0;")
        layout.addWidget(separator)

        export_zip_btn = QPushButton("📦  Export ROIs ZIP")
        export_zip_btn.setToolTip("Export all ROIs as a ZIP file")
        export_zip_btn.clicked.connect(lambda: QTimer.singleShot(0, self.export_roi_zip_requested.emit))
        layout.addWidget(export_zip_btn)

        layout.addStretch()

        self._current_tool = None
        self._current_tool_name = None

        # Mark Hand as checked; actual tool activation deferred until
        # signal is connected (app calls activate_default_tool).
        if 'Hand' in self._tool_buttons:
            self._tool_buttons['Hand'].setChecked(True)

    def activate_default_tool(self):
        """Activate the default tool. Call after signal connections are set up."""
        self._select_tool('Hand')

    def _add_tool_button(self, text, shortcut):
        icon = TOOL_ICONS.get(text, '')
        btn = QPushButton(f"{icon}  {text}" if icon else text)
        btn.setCheckable(True)
        btn.setShortcut(shortcut)
        btn.setToolTip(f"{text} [{shortcut}]")
        self.tool_group.addButton(btn)
        btn.clicked.connect(lambda checked, t=text: self._select_tool(t))
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
