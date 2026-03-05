from PySide6.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QTextEdit, QDialogButtonBox
from PySide6.QtCore import Qt


class HelpModal(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Montaris-X User Guide")
        self.resize(650, 500)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Getting Started
        tabs.addTab(self._text_tab(
            "Welcome to Montaris-X\n\n"
            "Montaris-X is a native ROI editor for scientific images.\n\n"
            "Quick Start:\n"
            "1. File > Open Image to load a TIFF/PNG/JPG\n"
            "2. Click '+' in the Layers panel to add ROI layers\n"
            "3. Select a drawing tool (Brush, Polygon, etc.)\n"
            "4. Draw on the canvas to create ROI masks\n"
            "5. File > Save ROI Set to save your work\n\n"
            "Navigation:\n"
            "- Scroll wheel to zoom\n"
            "- Middle-click or Space+drag to pan\n"
            "- Ctrl+0 to fit to window\n"
            "- Ctrl+1 for 1:1 zoom"
        ), "Getting Started")

        # Tools
        from montaris.tools import TOOL_REGISTRY
        tools_text = ""
        for name, (_, _, shortcut, category) in TOOL_REGISTRY.items():
            tools_text += f"  [{shortcut}] {name} ({category})\n"
        tools_text += "\nBrush Size: [ / ] keys or slider\n"
        tools_text += "Polygon: Click to add vertices, Enter/double-click to close\n"
        tools_text += "Transform: Click ROI to show handles, drag to scale/rotate\n"
        tools_text += "Move: Click+drag to move ROI (component-aware)\n"
        tabs.addTab(self._text_tab(tools_text), "Tools")

        # Keyboard Shortcuts
        shortcuts_text = (
            "File:\n"
            "  Ctrl+O         Open Image\n"
            "  Ctrl+Shift+O   Load ROI Set\n"
            "  Ctrl+S         Save ROI Set\n"
            "  Ctrl+E         Export ROI as PNG\n"
            "  Ctrl+Q         Quit\n\n"
            "Edit:\n"
            "  Ctrl+Z         Undo\n"
            "  Ctrl+Y / Ctrl+Shift+Z   Redo\n"
            "  Ctrl+Alt+Z     Layer Undo\n"
            "  Ctrl+Alt+Y     Layer Redo\n"
            "  Ctrl+A         Select All ROIs\n"
            "  Delete          Clear Active ROI\n\n"
            "View:\n"
            "  Ctrl+0         Fit to Window\n"
            "  Ctrl+1         Reset Zoom (1:1)\n"
            "  Ctrl+=         Zoom In\n"
            "  Ctrl+-         Zoom Out\n"
            "  Ctrl+R         Rotate 90 CW\n"
            "  H              Flip Horizontal\n\n"
            "Tools:\n"
            "  Ctrl+B         Brush\n"
            "  Ctrl+U         Bucket Fill\n"
            "  Escape          Clear selection / cancel transform\n"
            "  [ / ]          Adjust brush size\n\n"
            "Selection:\n"
            "  Ctrl+Click     Toggle ROI selection\n"
        )
        tabs.addTab(self._text_tab(shortcuts_text), "Keyboard Shortcuts")

        # File Formats
        formats_text = (
            "Images:\n"
            "  TIFF (.tif, .tiff) — including 16/32-bit scientific\n"
            "  PNG (.png)\n"
            "  JPEG (.jpg, .jpeg)\n"
            "  BMP (.bmp)\n\n"
            "ROI Formats:\n"
            "  NumPy Archive (.npz) — native Montaris format\n"
            "  ImageJ ROI (.roi) — binary format\n"
            "  PNG Masks (.png) — binary masks as images\n"
            "  ZIP Archive (.zip) — containing .roi or .png files\n\n"
            "Instructions:\n"
            "  JSON (.json) — batch operation scripts\n"
            "  Text (.txt) — human-readable instructions\n"
        )
        tabs.addTab(self._text_tab(formats_text), "File Formats")

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _text_tab(self, text):
        te = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText(text)
        return te
