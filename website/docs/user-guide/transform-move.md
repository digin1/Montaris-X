---
sidebar_position: 2
title: Transform & Move
description: Scale, rotate, and reposition ROIs with component-aware editing in Montaris-X.
keywords: [roi transform, roi move, component-aware editing, affine transform roi]
---

# Transform & Move

Montaris-X provides two spatial editing tools that understand the structure of your ROIs.

## Transform Tool (`T`)

Scale and rotate ROIs using interactive handles.

- **Selected ROI:** Press `T` to transform the active ROI
- **All ROIs:** Press `Shift+T` to transform all layers simultaneously
- Drag corner handles to **scale**
- Drag the rotation handle to **rotate**
- Press `Enter` to apply, `Escape` to cancel

### Component-Aware Mode

When transforming a single ROI, Montaris-X detects **connected components** (separate blobs within the same layer). Click on a specific component to transform just that piece — no need to split the layer first.

This uses `scipy.ndimage.binary_propagation` for fast component extraction, so even complex ROIs with many disconnected regions respond instantly.

## Move Tool (`V`)

Reposition ROIs by dragging.

- **Selected ROI:** Press `V` to move the active ROI
- **All ROIs:** Press `Shift+V` to move all layers together
- Click and drag to reposition

### Component-Aware Move

Like the Transform tool, Move is component-aware. Click on an individual connected component to move just that piece within the layer.

## Affine Transforms

Under the hood, transforms use **Pillow's `Image.transform(AFFINE, NEAREST)`** for fast nearest-neighbor resampling. This preserves the binary nature of mask data (no interpolation artifacts) and handles large masks efficiently.

## Undo Support

Both Transform and Move generate undo commands with **crop-based diffs** — only the affected bounding box is stored, keeping undo memory minimal even for large transforms.

| Action           | Shortcut     |
|------------------|--------------|
| Transform active | `T`          |
| Transform all    | `Shift+T`    |
| Move active      | `V`          |
| Move all         | `Shift+V`    |
| Apply            | `Enter`      |
| Cancel           | `Escape`     |
| Undo             | `Ctrl+Z`     |
