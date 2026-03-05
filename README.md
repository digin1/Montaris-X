# Montaris-X

Cross-platform ROI editor for scientific images.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Features

- Load TIFF, PNG, JPG images (16-bit TIFF supported)
- Multiple ROI layers with unique colors
- Brush, Eraser, and Polygon tools
- Undo/Redo (Ctrl+Z / Ctrl+Shift+Z)
- Save/Load ROI sets (.npz format)
- Export composited ROI overlay as PNG
- GPU-accelerated rendering (OpenGL)
- Pan (middle mouse or Space+drag) and zoom (scroll wheel)
- Dark theme UI
- Cross-platform: Windows, macOS, Linux

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Ctrl+O | Open image |
| Ctrl+S | Save ROI set |
| Ctrl+Shift+O | Load ROI set |
| Ctrl+E | Export ROI as PNG |
| Ctrl+Z | Undo |
| Ctrl+Shift+Z | Redo |
| Ctrl+0 | Fit to window |
| Ctrl+1 | Reset zoom (1:1) |
| B | Brush tool |
| E | Eraser tool |
| P | Polygon tool |
| [ / ] | Decrease/Increase brush size |
| Enter | Close polygon |
| Delete | Clear active ROI |
| Space+drag | Pan |
| Middle mouse | Pan |
| Scroll wheel | Zoom |

## Building Standalone App

```bash
pip install pyinstaller
pyinstaller --name Montaris-X --onedir --windowed --noconfirm main.py
```

The built application will be in the `dist/Montaris-X/` folder.

## Requirements

- Python 3.10+
- PySide6
- NumPy
- tifffile
- Pillow
