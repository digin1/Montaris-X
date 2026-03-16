---
sidebar_position: 8
title: Display Modes
description: Grayscale, false color, and composite display modes in Montaris-X.
keywords: [display modes, false color, composite view, grayscale microscopy]
---

# Display Modes

Montaris-X offers multiple display modes to help you visualize different types of scientific images. Switch modes from the **Display Panel** in the right sidebar.

## Grayscale

The default display mode for single-channel images. Maps pixel intensity to a linear gray ramp.

- Works with 8, 16, and 32-bit images
- Intensity range is automatically scaled to the image's data range
- Adjust with brightness/contrast/gamma for optimal visualization

## False Color

Apply a color lookup table (LUT) to map intensity values to colors. Useful for:

- Highlighting subtle intensity differences invisible in grayscale
- Matching the display style of other imaging software
- Visualizing specific intensity ranges

## Composite

For multi-channel images, composite mode overlays all channels with individual color tints:

- Each channel is assigned a distinct color (customizable)
- Toggle individual channels on/off
- The final image is a maximum-intensity projection of all tinted channels
- Standard fluorescence colors (blue/green/red) are assigned by default

## Flip & Rotate

Non-destructive orientation controls:

- **Flip Horizontal** (`Ctrl+H`): Mirror the image left-to-right
- **Rotate 90 CW** (`Ctrl+R`): Rotate the view clockwise
- These affect display only — exported data matches the displayed orientation
- Orientation is applied on load if specified in image metadata
