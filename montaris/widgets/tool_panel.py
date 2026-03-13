from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QSlider, QSpinBox,
    QButtonGroup, QHBoxLayout,
)
from PySide6.QtCore import Signal, Qt, QTimer
from montaris.tools import TOOL_REGISTRY, get_tool_class
from montaris.tools.polygon import PolygonTool
from montaris import theme as _theme
from montaris.widgets import AnimatedButton

try:
    import qtawesome as qta
    _HAS_QTA = True
except ImportError:
    _HAS_QTA = False

# QtAwesome icon names for each tool
_QTA_ICONS = {
    'Hand': 'fa6s.hand',
    'Select ROI': 'fa6s.arrow-pointer',
    'Brush': 'fa6s.paintbrush',
    'Eraser': 'fa6s.eraser',
    'Polygon': 'fa6s.draw-polygon',
    'Bucket Fill': 'fa6s.fill-drip',
    'Rectangle': 'fa6s.vector-square',
    'Circle': 'fa6.circle',
    'Stamp': 'fa6s.stamp',
    'Transform (selected)': 'fa6s.crop-simple',
    'Move (selected)': 'fa6s.arrows-up-down-left-right',
    'Transform All': 'fa6s.maximize',
    'Move All': 'fa6s.expand',
}

# Emoji fallback when QtAwesome is not installed
TOOL_ICONS = {
    'Hand': '🖐️',
    'Select ROI': '🖱️',
    'Brush': '🖌️',
    'Eraser': '🧹',
    'Polygon': '⬟',
    'Bucket Fill': '🪣',
    'Rectangle': '🔲',
    'Circle': '⬤',
    'Stamp': '■',
    'Transform (selected)': '🔧',
    'Move (selected)': '✥',
    'Transform All': '⛭',
    'Move All': '⊞',
}


def _tool_icon(name):
    """Return a QIcon for the tool, or None if unavailable."""
    if _HAS_QTA and name in _QTA_ICONS:
        color = '#dcdcdc' if _theme.is_dark() else '#333'
        return qta.icon(_QTA_ICONS[name], color=color)
    return None


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

        # Full-width clickable header bar — matches section style
        if _HAS_QTA:
            color = '#dcdcdc' if _theme.is_dark() else '#333'
            collapse_btn = QPushButton(qta.icon('fa6s.angles-left', color=color), " Toolbox")
        else:
            collapse_btn = QPushButton("\u25c0  Toolbox")
        collapse_btn.setFixedHeight(26)
        collapse_btn.setToolTip("Collapse sidebar (Ctrl+[)")
        collapse_btn.setStyleSheet(_theme.collapse_btn_style())
        self._collapse_btn = collapse_btn
        collapse_btn.clicked.connect(self.collapse_requested.emit)
        layout.addWidget(collapse_btn)

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(False)

        # Build tool buttons from registry, grouped by category
        categories = {}
        for name, (module, cls_name, shortcut, category) in TOOL_REGISTRY.items():
            categories.setdefault(category, []).append((name, shortcut))

        _section_style = _theme.section_header_style()
        self._section_labels = []
        self._action_btns = []

        for category, tools in categories.items():
            cat_label = QLabel(f"  {category}")
            cat_label.setFixedHeight(26)
            cat_label.setStyleSheet(_section_style)
            self._section_labels.append(cat_label)
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
        self.tolerance_slider.setStyleSheet(_theme.slider_style())
        self.tolerance_slider.valueChanged.connect(self._on_tolerance_changed)
        self.tolerance_slider.setVisible(False)
        tol_layout.addWidget(self.tolerance_slider)

        self.tolerance_spin = QSpinBox()
        self.tolerance_spin.setRange(0, 255)
        self.tolerance_spin.setValue(0)
        self.tolerance_spin.setStyleSheet(_theme.spinbox_style())
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
        file_label = QLabel("  Quick Actions")
        file_label.setFixedHeight(26)
        file_label.setStyleSheet(_section_style)
        self._section_labels.append(file_label)
        layout.addWidget(file_label)

        montage_btn = self._action_btn('fa6s.image', "Load Image(s)")
        montage_btn.setToolTip("Open montage or image file(s)")
        montage_btn.clicked.connect(lambda: QTimer.singleShot(0, self.open_montage_requested.emit))
        layout.addWidget(montage_btn)

        roi_zip_btn = self._action_btn('fa6s.file-import', "Import ROI ZIP")
        roi_zip_btn.setToolTip("Import ROIs from a ZIP file")
        roi_zip_btn.clicked.connect(lambda: QTimer.singleShot(0, self.import_roi_zip_requested.emit))
        layout.addWidget(roi_zip_btn)

        instr_btn = self._action_btn('fa6s.file-lines', "Load Instructions")
        instr_btn.setToolTip("Load instructions file (.json/.txt)")
        instr_btn.clicked.connect(lambda: QTimer.singleShot(0, self.load_instructions_requested.emit))
        layout.addWidget(instr_btn)

        view_instr_btn = self._action_btn('fa6s.eye', "View Instructions")
        view_instr_btn.setToolTip("View loaded instructions")
        view_instr_btn.clicked.connect(lambda: QTimer.singleShot(0, self.view_instructions_requested.emit))
        layout.addWidget(view_instr_btn)

        # --- Separator + Export ---
        separator = QLabel()
        separator.setFixedHeight(2)
        separator.setStyleSheet(_theme.separator_style())
        self._separator = separator
        layout.addWidget(separator)

        export_zip_btn = self._action_btn('fa6s.file-export', "Export ROIs ZIP")
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

    def _action_btn(self, qta_name, text):
        """Create an AnimatedButton with a QtAwesome icon (or plain text fallback)."""
        if _HAS_QTA:
            color = '#dcdcdc' if _theme.is_dark() else '#333'
            icon = qta.icon(qta_name, color=color)
            btn = AnimatedButton(icon, f" {text}")
        else:
            btn = AnimatedButton(text)
        btn.setStyleSheet(_theme.action_button_style())
        self._action_btns.append(btn)
        return btn

    def refresh_theme(self):
        """Re-apply theme-dependent styles after a theme switch."""
        self._collapse_btn.setStyleSheet(_theme.collapse_btn_style())
        ss = _theme.section_header_style()
        for lbl in self._section_labels:
            lbl.setStyleSheet(ss)
        self._separator.setStyleSheet(_theme.separator_style())
        ts = _theme.tool_button_style()
        for btn in self._tool_buttons.values():
            btn.setStyleSheet(ts)
        for btn in self._action_btns:
            btn.setStyleSheet(_theme.action_button_style())
        self.tolerance_slider.setStyleSheet(_theme.slider_style())
        self.tolerance_spin.setStyleSheet(_theme.spinbox_style())
        # Refresh icon colors
        if _HAS_QTA:
            color = '#dcdcdc' if _theme.is_dark() else '#333'
            self._collapse_btn.setIcon(qta.icon('fa6s.angles-left', color=color))
            for name, btn in self._tool_buttons.items():
                if name in _QTA_ICONS:
                    btn.setIcon(qta.icon(_QTA_ICONS[name], color=color))

    def _add_tool_button(self, text, shortcut):
        qicon = _tool_icon(text)
        if qicon:
            btn = AnimatedButton(qicon, f" {text}")
        else:
            emoji = TOOL_ICONS.get(text, '')
            btn = AnimatedButton(f"{emoji}  {text}" if emoji else text)
        btn.setCheckable(True)
        btn.setShortcut(shortcut)
        btn.setToolTip(f"{text} [{shortcut}]")
        btn.setStyleSheet(_theme.tool_button_style())
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
        """Deselect current tool and activate Hand (C.23)."""
        self._select_tool('Hand')

    def _on_size_changed(self, value):
        if self._current_tool and hasattr(self._current_tool, 'size'):
            self._current_tool.size = value
        if hasattr(self.app, 'canvas'):
            self.app.canvas.refresh_brush_cursor()

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
