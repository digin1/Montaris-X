---
sidebar_position: 1
title: Drawing Tools
description: Guide to all 7 drawing tools in Montaris-X — Brush, Eraser, Polygon, Rectangle, Circle, Stamp, and Bucket Fill.
keywords: [roi drawing tools, brush tool, polygon tool, scientific annotation tools]
---

# Drawing Tools

Montaris-X provides 7 drawing tools optimized for ROI delineation on scientific images. Select tools from the toolbar or use keyboard shortcuts.

## Brush (`B`)

Freehand drawing tool for tracing ROI boundaries.

- **Size:** Adjust with `[` / `]` keys or the slider in the Tool Panel (1–500 px)
- **Auto-overlap mode:** When enabled, the brush can paint over existing ROIs on other layers. Toggle in the Tool Panel.
- **Color preview:** The brush cursor shows the active ROI's color

**Tips:**
- Zoom in (`Ctrl+=`) for precise edges
- Hold `Space` to pan without switching tools
- The brush only draws on the currently selected ROI layer

## Eraser (`E`)

Removes pixels from the active ROI layer.

- **Size:** Same `[` / `]` controls as the brush
- Works identically to the brush but removes instead of adding

## Polygon (`P`)

Click to place vertices, creating straight-edged boundaries.

- Click to add each vertex
- Press `Enter` to close the polygon and fill it
- Press `Escape` to cancel
- Works well for regions with clean geometric boundaries

## Rectangle (`R`)

Click and drag to draw a filled rectangle.

- Hold `Shift` to constrain to a square
- Release to commit the shape

## Circle (`C`)

Click and drag to draw a filled ellipse.

- Hold `Shift` to constrain to a perfect circle
- The shape is defined by the bounding box of your drag

## Stamp (`S`)

Places a pre-sized rectangular or circular region.

- **Width / Height:** Set in the Tool Panel
- Click to stamp at the cursor position
- Useful for placing uniform-sized ROIs (e.g., counting frames, sampling regions)

## Bucket Fill (`G`)

Flood-fills an enclosed area on the active ROI layer.

- **Tolerance:** Adjust in the Tool Panel (0–255). Higher tolerance fills through more variation.
- Click inside an enclosed boundary to fill
- Works best after outlining a region with the brush or polygon tool

## Common Controls

| Action             | Control               |
|--------------------|-----------------------|
| Switch tool        | Keyboard shortcut     |
| Adjust brush size  | `[` decrease / `]` increase |
| Pan while drawing  | Hold `Space`          |
| Zoom               | Scroll wheel          |
| Undo last stroke   | `Ctrl+Z`              |
| Redo               | `Ctrl+Y` or `Ctrl+Shift+Z` |
