<p align="center">
  <img src="https://raw.githubusercontent.com/digin1/Montaris-X/main/montaris/assets/logo.png" alt="Montaris-X" width="128">
</p>

<h1 align="center">Montaris-X</h1>

<p align="center">
  <strong>Cross-platform ROI editor for scientific microscopy images</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/montaris-x/"><img src="https://img.shields.io/pypi/v/montaris-x?color=blue&v=2.1.4" alt="PyPI"></a>
  <a href="https://pypi.org/project/montaris-x/"><img src="https://img.shields.io/pypi/pyversions/montaris-x?v=2.1.4" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/digin1/Montaris-X" alt="License"></a>
  <a href="https://github.com/digin1/Montaris-X/actions"><img src="https://img.shields.io/github/actions/workflow/status/digin1/Montaris-X/tests.yml?label=tests" alt="Tests"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/digin1/Montaris-X/main/docs/demo.gif" alt="Montaris-X Demo" width="800">
</p>

---

## About

Montaris-X is a native desktop application for drawing, editing, and managing **Regions of Interest (ROIs)** on scientific microscopy images. It is built for researchers who need precise, pixel-level annotation of histological sections, brain atlases, fluorescence micrographs, and other large-format scientific imagery.

**Why Montaris-X?**

- **Purpose-built for science** — handles 16/32-bit TIFFs, multi-channel composites, and images with thousands of ROIs without breaking a sweat.
- **ImageJ-compatible** — import and export `.roi` / `.zip` files directly from ImageJ/FIJI workflows.
- **No cloud, no account** — runs entirely offline on your machine. Your data never leaves your computer.
- **Cross-platform** — native support for Windows, macOS, and Linux.

---

## Installation

### Option 1: Download Pre-built Executable (Recommended)

Download the latest release for your platform from the [Releases page](https://github.com/digin1/Montaris-X/releases):

| Platform | File | Notes |
|----------|------|-------|
| **Windows** | `montaris-x-windows.exe` | Double-click to run. Signed binary. |
| **macOS** | `montaris-x-macos` | Run `chmod +x montaris-x-macos && ./montaris-x-macos` |
| **Linux** | `montaris-x-linux` | Run `chmod +x montaris-x-linux && ./montaris-x-linux` |

> **macOS note:** If macOS shows "unidentified developer", right-click the file and select **Open**, then click **Open** in the dialog.

> **Linux note:** You may need to install EGL and D-Bus libraries: `sudo apt install libegl1 libxkbcommon0 libdbus-1-3`

### Option 2: Install from PyPI

```bash
pip install montaris-x
```

Then launch:

```bash
montaris
```

### Option 3: Install from Source

```bash
git clone https://github.com/digin1/Montaris-X.git
cd Montaris-X
pip install -e .
```

Then launch:

```bash
python main.py
# or
python -m montaris
# or
montaris
```

### Optional: GPU Acceleration

Install [Numba](https://numba.pydata.org/) for JIT-compiled rendering on large images:

```bash
pip install montaris-x[accel]
# or, if installing from source:
pip install numba
```

---

## Features

### Image Support
- **Scientific formats** — TIFF (8/16/32-bit, multi-channel), PNG, JPEG, BMP
- **Multi-channel composites** — split and view individual channels with false-color display
- **Image adjustments** — brightness, contrast, exposure, gamma with Smart Auto and Quick Boost
- **Tiled rendering** — LOD-based rendering for smooth navigation on large images
- **Flip & Rotate** — non-destructive horizontal flip and 90-degree rotation

### ROI Editing
- **Drawing tools** — Brush, Eraser, Rectangle, Circle, Polygon, Stamp, Bucket Fill
- **Transform & Move** — scale, rotate, and reposition ROIs with connected-component awareness
- **Multiple ROI layers** — 20+ distinct colors with per-ROI opacity and fill modes (Solid, Outline, Both)
- **Multi-selection** — Ctrl+click to select and edit multiple ROIs simultaneously
- **Undo/Redo** — full global history plus per-layer undo
- **Overlap handling** — auto-overlap brush mode and fix-overlaps utility

### Import / Export
- **Native format** — save/load ROI sets as `.npz` archives
- **ImageJ compatible** — import/export `.roi` files and ZIP bundles
- **PNG masks** — import/export binary mask images
- **Composited overlay** — export the ROI overlay as a single PNG
- **Batch instructions** — JSON/TXT instruction files for automated workflows
- **Drag-and-drop** — drop images and ROI ZIPs directly onto the window

### Interface
- **Dark & Light themes** — switch between themes on the fly
- **Collapsible sidebars** — minimize panels for a distraction-free canvas
- **Fullscreen mode** — F11 for immersive editing
- **Minimap** — always-visible overview of the full image
- **Performance monitor** — real-time FPS and memory usage
- **Global opacity slider** — adjust overlay transparency from the toolbar
- **Session save** — auto-save and restore progress across sessions

---

## Keyboard Shortcuts

<details>
<summary>Click to expand full shortcut reference</summary>

### File

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open Image(s) |
| `Ctrl+Shift+O` | Load ROI Set |
| `Ctrl+S` | Save ROI Set |
| `Ctrl+Shift+S` | Save Progress (Session) |
| `Ctrl+E` | Export ROI(s) as PNG |
| `Ctrl+W` | Close Image(s) |
| `Ctrl+Q` | Quit |

### Edit

| Shortcut | Action |
|----------|--------|
| `Ctrl+Z` | Undo |
| `Ctrl+Y` / `Ctrl+Shift+Z` | Redo |
| `Ctrl+Alt+Z` | Layer Undo |
| `Ctrl+Alt+Y` | Layer Redo |
| `Ctrl+A` | Select All ROIs |
| `Delete` | Delete Active ROI(s) |

### View

| Shortcut | Action |
|----------|--------|
| `Ctrl+0` | Fit to Window |
| `Ctrl+1` | Reset Zoom (1:1) |
| `Ctrl+=` | Zoom In |
| `Ctrl+-` | Zoom Out |
| `Ctrl+H` | Flip Horizontal |
| `Ctrl+R` | Rotate 90 CW |
| `Ctrl+[` | Collapse Left Sidebar |
| `Ctrl+]` | Collapse Right Sidebar |
| `F11` | Fullscreen |
| `Ctrl+Shift+P` | Screenshot |

### Tools

| Shortcut | Action |
|----------|--------|
| `H` | Hand (pan) |
| `Q` | Select ROI |
| `B` | Brush |
| `E` | Eraser |
| `P` | Polygon |
| `G` | Bucket Fill |
| `R` | Rectangle |
| `C` | Circle |
| `S` | Stamp |
| `T` | Transform (selected) |
| `Shift+T` | Transform All |
| `V` | Move (selected) |
| `Shift+V` | Move All |
| `[` / `]` | Adjust brush size |
| `Space` | Pan (hold) |
| `Enter` | Finish Polygon |
| `Escape` | Cancel / Clear selection |

### Navigation

| Shortcut | Action |
|----------|--------|
| `Scroll wheel` | Zoom in/out |
| `Middle-click drag` | Pan |
| `Ctrl+Click` | Toggle ROI selection |

</details>

---

## Supported Formats

| Type | Formats |
|------|---------|
| **Images** | TIFF (`.tif`, `.tiff`) including 16/32-bit, PNG, JPEG, BMP |
| **ROI Sets** | NumPy Archive (`.npz`), ImageJ ROI (`.roi`), ZIP bundles, PNG masks |
| **Instructions** | JSON (`.json`), Text (`.txt`) |

---

## Building from Source

### Running Tests

```bash
pip install -e ".[test]"
pytest
```

### Building a Standalone Executable

```bash
pip install pyinstaller
pyinstaller --name montaris-x --onefile --windowed --hidden-import montaris --collect-all montaris montaris/__main__.py
```

The executable will be in the `dist/` folder.

### Building a Python Package

```bash
pip install build
python -m build
```

This produces `.whl` and `.tar.gz` files in `dist/` for distribution.

---

## System Requirements

- **Python** 3.10 or higher (for source/PyPI installation)
- **OS:** Windows 10+, macOS 11+, or Linux (X11/Wayland with EGL)
- **RAM:** 4 GB minimum, 8 GB+ recommended for large images
- **Dependencies:** PySide6, NumPy, SciPy, scikit-image, tifffile, Pillow, psutil

### Linux System Dependencies

On Debian/Ubuntu, install the required Qt runtime libraries:

```bash
sudo apt install libegl1 libxkbcommon0 libdbus-1-3
```

---

## Code Signing

Code signing via [SignPath.io](https://signpath.io) (certificate by [SignPath Foundation](https://signpath.org)) has been applied for and is pending approval.

- **Privacy policy**: This program does not transfer any information to other networked systems unless specifically requested by the user or the person installing or operating it.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/digin1/Montaris-X).

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Push to the branch and open a Pull Request

---

## License

Montaris-X is released under the [MIT License](LICENSE).

Copyright (c) 2026 Digin Dominic and Montaris-X Contributors
