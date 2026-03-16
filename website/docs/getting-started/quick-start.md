---
sidebar_position: 2
title: Quick Start
description: Open your first image and draw an ROI in under 5 minutes with Montaris-X.
keywords: [montaris-x tutorial, roi editor quickstart, draw roi scientific image]
---

# Quick Start

Get from zero to your first annotated ROI in under 5 minutes.

## 1. Open an Image

Launch Montaris-X and open an image:

- **Menu:** File → Open Image (`Ctrl+O`)
- **Drag & drop:** Drop a TIFF, PNG, or JPEG file onto the window

Montaris-X handles 8/16/32-bit TIFFs natively — no conversion needed.

## 2. Add an ROI Layer

Click **Add ROI** in the Layer Panel (left sidebar). A new ROI layer appears with a unique name and color.

You can add as many ROI layers as you need — each gets its own color from a palette of 20 distinct colors.

## 3. Draw Your First ROI

Select a drawing tool from the toolbar:

| Tool        | Shortcut | Best for                          |
|-------------|----------|-----------------------------------|
| **Brush**   | `B`      | Freehand tracing along edges      |
| **Polygon** | `P`      | Clean boundaries with straight edges |
| **Rectangle** | `R`    | Rectangular regions               |
| **Circle**  | `C`      | Circular/elliptical regions       |

- Adjust brush size with `[` and `]` keys
- Hold `Space` to pan while drawing
- Scroll to zoom in/out

## 4. Refine

- **Eraser** (`E`): Remove parts of the ROI
- **Bucket Fill** (`G`): Fill enclosed areas (adjust tolerance in the tool panel)
- **Transform** (`T`): Scale or rotate the ROI
- **Undo** (`Ctrl+Z`): Undo any mistake

## 5. Save Your Work

- **Save ROI Set** (`Ctrl+S`): Save all ROIs as a `.npz` archive
- **Save Session** (`Ctrl+Shift+S`): Save the entire workspace for later
- **Export:** File → Export to save as ImageJ `.roi` files, PNG masks, or ZIP bundles

## Next Steps

- [Drawing Tools](/docs/user-guide/drawing-tools) — detailed guide for all 7 tools
- [Layer Management](/docs/user-guide/layer-management) — colors, opacity, fill modes
- [Import & Export](/docs/user-guide/import-export) — all supported formats
- [Keyboard Shortcuts](/docs/keyboard-shortcuts) — full reference
