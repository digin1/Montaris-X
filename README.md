<p align="center">
  <img src="montaris/assets/logo.png" alt="Montaris-X" width="128">
</p>

<h1 align="center">Montaris-X</h1>

<p align="center">
  Desktop ROI editor for scientific microscopy images
</p>

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Features

**Image Support**
- TIFF (8/16-bit, multi-channel), PNG, JPG
- Multi-channel composite and false-color display modes
- Brightness, contrast, and gamma adjustments
- Tiled rendering with LOD for large images

**ROI Editing**
- Multiple ROI layers with 20 distinct colors
- Brush, Eraser, Rectangle, Circle, Polygon, Stamp, Bucket Fill tools
- Transform and Move tools with connected-component awareness
- Multi-selection editing (Ctrl+click)
- Auto-overlap brush mode
- Undo/Redo with full history

**Import/Export**
- Save/Load ROI sets (.npz)
- Import/Export ImageJ .roi files (single and ZIP bundles)
- Import PNG mask files
- Export composited ROI overlay as PNG
- Batch instruction files (JSON/TXT)
- Auto-fit out-of-bounds ROIs on import

**UI**
- Global opacity control and per-ROI opacity
- Solid, Outline, and Both fill modes
- Inline layer rename, color palette dialog
- Minimap, performance monitor, debug console
- HUD overlay with zoom/tool info
- Help modal with tool reference and keyboard shortcuts
- Dark theme, cross-platform (Windows, macOS, Linux)

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
| R | Rectangle tool |
| C | Circle tool |
| T | Transform tool |
| M | Move tool |
| S | Stamp tool |
| F | Bucket Fill tool |
| [ / ] | Decrease/Increase brush size |
| Enter | Close polygon |
| Delete | Clear active ROI |
| Space+drag | Pan |
| Scroll wheel | Zoom |

## Building Standalone App

```bash
pip install pyinstaller
pyinstaller --name Montaris-X --onedir --windowed --noconfirm main.py
```

The built application will be in the `dist/Montaris-X/` folder.

## Code Signing Policy

Free code signing provided by [SignPath.io](https://signpath.io), certificate by [SignPath Foundation](https://signpath.org).

- **Committers and reviewers**: [Repository contributors](https://github.com/digin1/Montaris-X/graphs/contributors)
- **Approvers**: [Repository owner](https://github.com/digin1)
- **Privacy policy**: This program will not transfer any information to other networked systems unless specifically requested by the user or the person installing or operating it.

## Requirements

- Python 3.10+
- PySide6
- NumPy
- SciPy
- tifffile
- Pillow
