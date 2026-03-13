---
sidebar_position: 3
title: Layer Management
description: Manage ROI layers in Montaris-X — colors, opacity, fill modes, naming, and multi-selection.
keywords: [roi layers, roi colors, layer management, roi opacity, annotation layers]
---

# Layer Management

Each ROI in Montaris-X lives on its own layer. The Layer Panel (left sidebar) gives you full control over your ROIs.

## Adding & Removing ROIs

- **Add ROI:** Click the **+** button. A new layer is created with a unique auto-generated name and a color from the 20-color palette.
- **Delete ROI:** Select the layer and press `Delete`, or right-click → Delete. A confirmation dialog prevents accidental deletion.
- **Clear All:** Remove all ROI layers at once from the menu.

## Colors

Montaris-X provides **20 distinct colors** designed for maximum contrast on scientific images. Colors are auto-assigned but can be changed:

- Click the color swatch next to a layer name to open the **Color Palette Dialog**
- Choose from the 20 preset colors or pick a custom color
- Colors are consistent across export formats

## Naming

- Layers get unique auto-generated names (e.g., `ROI-1`, `ROI-2`)
- **Inline rename:** Double-click a layer name to edit it directly in the panel
- Names are preserved in `.npz` saves and ImageJ `.roi` exports

## Fill Modes

Each ROI can be displayed in one of three fill modes (set in the Properties Panel):

| Mode        | Display                                    |
|-------------|--------------------------------------------|
| **Solid**   | Filled region with the layer's color       |
| **Outline** | Only the boundary edge is drawn            |
| **Both**    | Filled region with a highlighted border    |

## Opacity

- **Per-layer opacity:** Adjust individual layer transparency
- **Global opacity slider:** In the toolbar, adjust the overlay transparency for all ROIs at once

## Multi-Selection

- **Ctrl+Click** layers to add/remove them from the selection
- **Ctrl+A** selects all ROIs
- Selected ROIs can be moved, transformed, or deleted together
- The selection syncs bidirectionally with the canvas — clicking an ROI on the canvas selects its layer, and vice versa

## ROI Navigation Bar

At the top of the Layer Panel, the **Nav Bar** provides quick navigation:

- Arrow buttons to jump between ROIs
- Shows the current ROI index and total count
- Useful when working with many layers

## Pixel Counts

Each layer shows its pixel count — the number of non-zero pixels in the mask. This updates automatically as you draw and is useful for measuring ROI area.
