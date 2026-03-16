---
sidebar_position: 5
title: Multi-Channel Images
description: Work with multi-channel microscopy images in Montaris-X — montage, composite, and false-color display.
keywords: [multi-channel microscopy, fluorescence channels, composite image, montage view]
---

# Multi-Channel Images

Montaris-X has full support for multi-channel scientific images — common in fluorescence microscopy, immunohistochemistry, and multi-spectral imaging.

## Opening Multi-Channel Images

When you open a multi-channel TIFF, Montaris-X detects the channels automatically and presents a **Montage Document** dialog:

- **Montage view:** All channels arranged in a grid
- **Single channel:** View one channel at a time
- **Downsample dialog:** For very large images, choose a working resolution

## Composite Display

View multiple channels overlaid with distinct color tints:

- Each channel gets its own tint color (e.g., DAPI = blue, GFP = green, RFP = red)
- Adjust individual channel visibility
- The composite updates in real time as you toggle channels

## False Color

Apply false-color lookup tables to single-channel images:

- Switch between grayscale and false-color in the Display Panel
- Useful for visualizing intensity variations in 16/32-bit data

## Channel Tints

Customize the color assigned to each channel:

- Click the color swatch next to a channel in the Display Panel
- Choose from preset fluorophore colors or pick a custom color
- Tints are applied in real time to the composite view

## Document Switching

When working with multiple montage documents:

- Switch between documents using the document tabs
- Each document maintains its own ROI layers, zoom level, and display settings
- ROIs are independent per document

## Downsampling

For very large multi-channel images:

- A dialog prompts you to choose a downsample factor on load
- Work at reduced resolution for faster navigation
- Exported masks are automatically **upscaled** to the original resolution
