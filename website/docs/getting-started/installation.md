---
sidebar_position: 1
title: Installation
description: Install Montaris-X on Windows, macOS, or Linux — pre-built executables, PyPI, or from source.
keywords: [install montaris-x, roi editor install, scientific image annotation setup]
---

# Installation

Montaris-X runs on **Windows 10+**, **macOS 11+**, and **Linux** (X11/Wayland). Choose the installation method that works best for you.

## Option 1: Pre-built Executable (Recommended)

Download the latest release for your platform from the [GitHub Releases page](https://github.com/digin1/Montaris-X/releases):

| Platform    | File                    | Notes                                                        |
|-------------|-------------------------|--------------------------------------------------------------|
| **Windows** | `montaris-x-windows.exe`| Double-click to run                                          |
| **macOS**   | `montaris-x-macos`      | Run `chmod +x montaris-x-macos && ./montaris-x-macos`        |
| **Linux**   | `montaris-x-linux`      | Run `chmod +x montaris-x-linux && ./montaris-x-linux`        |

:::tip macOS Gatekeeper
If macOS shows "unidentified developer", right-click the file and select **Open**, then click **Open** in the dialog.
:::

## Option 2: Install from PyPI

```bash
pip install montaris-x
```

Then launch:

```bash
montaris
```

## Option 3: Install from Source

```bash
git clone https://github.com/digin1/Montaris-X.git
cd Montaris-X
pip install -e .
```

Then launch with any of:

```bash
python main.py
python -m montaris
montaris
```

## Linux System Dependencies

On Debian/Ubuntu, install the required Qt runtime libraries:

```bash
sudo apt install libegl1 libxkbcommon0 libdbus-1-3
```

## System Requirements

- **Python** 3.10 or higher (for PyPI / source install)
- **RAM:** 4 GB minimum, 8 GB+ recommended for large images
- **Dependencies** (installed automatically): PySide6, NumPy, SciPy, scikit-image, tifffile, Pillow, psutil, Numba
