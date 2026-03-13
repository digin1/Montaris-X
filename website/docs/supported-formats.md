---
sidebar_position: 12
title: Supported Formats
description: All file formats supported by Montaris-X — images, ROIs, and instruction files.
keywords: [tiff support, imagej roi format, supported file formats, scientific image formats]
---

# Supported Formats

Montaris-X supports a range of scientific and standard image formats.

## Image Formats

| Format         | Extensions         | Bit Depth           | Notes                          |
|----------------|--------------------|---------------------|--------------------------------|
| **TIFF**       | `.tif`, `.tiff`    | 8, 16, 32-bit       | Multi-channel, multi-page supported. Memory-mapped for large files. |
| **PNG**        | `.png`             | 8-bit               | Standard lossless format       |
| **JPEG**       | `.jpg`, `.jpeg`    | 8-bit               | Lossy — not recommended for quantitative work |
| **BMP**        | `.bmp`             | 8-bit               | Windows bitmap                 |

### TIFF Details

Montaris-X uses [tifffile](https://github.com/cgohlke/tifffile) for TIFF reading, providing full support for:

- 8, 16, and 32-bit integer and floating-point data
- Multi-channel images (e.g., RGB, multi-fluorescence)
- Multi-page TIFFs
- BigTIFF format for files larger than 4 GB
- Memory-mapped reading for large files (reduces RAM usage)

## ROI Formats

| Format              | Extension(s)    | Description                                    |
|---------------------|-----------------|------------------------------------------------|
| **NPZ Archive**     | `.npz`          | Native format. Stores all layers, names, colors, and metadata. |
| **ImageJ ROI**      | `.roi`          | Binary ImageJ ROI format. Single region per file. |
| **ImageJ ZIP**      | `.zip`          | ZIP bundle of multiple `.roi` files. Compatible with ImageJ ROI Manager. |
| **PNG Mask**        | `.png`          | Binary mask image (black = background, white = ROI). |

### ImageJ Compatibility

Montaris-X writes `.roi` files using the ImageJ binary format specification. These files can be:

- Opened directly in ImageJ/FIJI via Analyze → Tools → ROI Manager
- Loaded from ZIP bundles containing multiple ROIs
- Round-tripped between Montaris-X and ImageJ without data loss

## Instruction Formats

| Format    | Extension | Description                              |
|-----------|-----------|------------------------------------------|
| **JSON**  | `.json`   | Structured instruction file for batch operations |
| **Text**  | `.txt`    | Simple line-based instruction format     |

Instruction files define batch workflows for automated ROI creation and processing.

## Export Formats

| Export Type          | Format    | Description                              |
|----------------------|-----------|------------------------------------------|
| ROI Set              | `.npz`    | All layers in native format              |
| Individual ROIs      | `.roi`    | ImageJ binary format                     |
| ROI Bundle           | `.zip`    | ZIP of `.roi` files                      |
| Individual Masks     | `.png`    | One PNG per ROI layer                    |
| Composited Overlay   | `.png`    | All ROIs combined as a single colored image |
