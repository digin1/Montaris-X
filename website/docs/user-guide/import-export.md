---
sidebar_position: 4
title: Import & Export
description: Import and export ROIs in Montaris-X — NPZ, ImageJ .roi, ZIP bundles, PNG masks, and batch instructions.
keywords: [roi import export, imagej roi format, png mask export, scientific data export]
---

# Import & Export

Montaris-X supports multiple formats for interoperability with ImageJ/FIJI and other tools.

## Native Format (NPZ)

The default format for saving and loading ROI sets.

- **Save:** File → Save ROI Set (`Ctrl+S`)
- **Load:** File → Load ROI Set (`Ctrl+Shift+O`)
- Stores all layers, names, colors, and metadata in a single `.npz` archive
- Fastest save/load — recommended for work-in-progress

## ImageJ ROI Format

Full compatibility with ImageJ/FIJI's `.roi` binary format.

### Import
- File → Import → ImageJ ROI (`.roi` file or `.zip` bundle)
- Drag & drop `.roi` or `.zip` files onto the window
- ROIs are converted to mask layers automatically

### Export
- File → Export → ImageJ ROI
- Exports each layer as an individual `.roi` file
- Export as ZIP bundle for multi-ROI sets
- Compatible with ImageJ's ROI Manager

## PNG Masks

Import and export ROIs as binary mask images.

- **Import:** File → Import → PNG Mask — loads a black/white image as an ROI layer
- **Export individual:** File → Export → PNG Mask — saves each ROI as a separate PNG
- **Export composite:** File → Export → Composited Overlay — saves all ROIs as a single colored PNG

## Batch Instructions

For automated workflows, Montaris-X can load instruction files:

- **JSON format:** Structured list of operations
- **TXT format:** Simple line-based format
- Useful for scripted pipelines and reproducible analysis

## Auto-fit on Import

When importing ROIs that extend outside the image bounds (e.g., from a different-sized image), Montaris-X automatically fits them to the canvas.

## Drag & Drop

Drop any supported file directly onto the Montaris-X window:

- Images: `.tif`, `.tiff`, `.png`, `.jpg`, `.bmp`
- ROIs: `.npz`, `.roi`, `.zip`
- The file type is detected automatically

## Mask Upscaling on Export

When working with downsampled montages, exported masks are automatically upscaled to match the original image resolution.
