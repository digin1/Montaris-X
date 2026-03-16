---
sidebar_position: 7
title: Image Adjustments
description: Brightness, contrast, exposure, gamma, Smart Auto, and Quick Boost in Montaris-X.
keywords: [image adjustments, brightness contrast, scientific image enhancement, gamma correction]
---

# Image Adjustments

Montaris-X provides non-destructive image adjustments to help you visualize structures in your scientific images. Adjustments affect only the display — your original image data is never modified.

## Adjustment Controls

Access adjustments from the **Adjustments Panel** in the right sidebar.

### Brightness
Shift the overall luminance of the image up or down. Useful for images that are too dark or washed out.

### Contrast
Increase or decrease the difference between light and dark regions. Higher contrast makes edges and boundaries easier to see.

### Exposure
Simulates camera exposure adjustment. Particularly useful for 16/32-bit images where the full dynamic range may not be visible at default settings.

### Gamma
Non-linear brightness correction. Values below 1.0 brighten shadows while preserving highlights. Values above 1.0 darken midtones.

## Smart Auto

Click **Smart Auto** to automatically optimize brightness, contrast, and gamma based on the image histogram. Works well for:

- Under-exposed fluorescence images
- Low-contrast histology sections
- 16-bit images with narrow intensity ranges

## Quick Boost

One-click enhancement that applies a moderate contrast and brightness boost. Less aggressive than Smart Auto — good for images that just need a slight lift.

## Reset

Click **Reset** to return all adjustments to their default values.

## Notes

- All adjustments are **non-destructive** — the original pixel data is preserved
- Adjustments apply to the **display only** and do not affect exported ROI masks
- Adjustments are saved with your session and restored on reload
