---
sidebar_position: 7
title: Grid Canvas
description: Work with multiple images side-by-side using the Grid Canvas in Montaris-X.
keywords: [grid canvas, multi-image, side-by-side, batch annotation, compare images]
---

# Grid Canvas

The Grid Canvas lets you define an NxM grid (up to 4x4) where each cell holds an independent image with its own ROIs, undo history, and adjustments. This is useful for comparing samples, batch annotation, or working with multiple sections simultaneously.

## Setting Up a Grid

1. Go to **View > Grid Layout...** to open the grid setup dialog
2. Choose the number of rows and columns (1-4 each)
3. Click OK to create the grid

The grid starts at 1x1 (single image, the default). You can change the grid size at any time — existing cells with content are preserved when growing.

## Loading Images into Cells

1. Click a cell to make it active (highlighted with a cyan border)
2. Open an image normally with **File > Open Image** (`Ctrl+O`)
3. Import ROIs with **File > Import > Load ROI ZIP**
4. Click another cell and repeat

Each cell is a fully independent workspace — it has its own:
- Image and channel stack
- ROI layers with colors and names
- Undo/redo history
- Image adjustments (brightness, contrast, etc.)
- Downsample factor
- Session directory

## Switching Between Cells

- **Click** any cell to switch to it — all panels and tools update to show that cell's state
- The active cell is highlighted with a cyan border
- A toast notification shows which cell is active (e.g., "Active: cell (2, 3)")

## Maximizing a Cell

To focus on a single cell without changing the grid:

- **Double-click** a cell to maximize it to fill the entire grid area
- **Double-click** again to restore all cells
- Or use **View > Maximize Cell** (`Ctrl+Shift+M`)

This is useful for detailed work on one image before switching back to the overview.

## Saving and Exporting

### Save All Sessions

Save progress for every cell at once:

- **File > Save All Grid Sessions** (`Ctrl+Alt+S`)
- Each cell's ROIs are saved to a session folder alongside its source image
- Empty cells are automatically skipped

### Export All as ZIP

Export each cell's ROIs as a separate ImageJ-compatible ZIP:

- **File > Export > All Grid Cells as ZIP** (`Ctrl+Alt+E`)
- Choose a directory — one ZIP per cell is created
- Files are named by source image: `brain_A_R1C1.zip`, `brain_B_R2C1.zip`
- Per-cell downsample factors are respected for upscaling

### Per-Cell Operations

All standard save/export operations (Save ROI Set, Export PNG, etc.) work on the currently active cell.

## Resizing the Grid

When shrinking the grid (e.g., from 3x3 to 2x2):

- Cells that still fit are preserved
- If any out-of-bounds cells have content, a confirmation dialog appears before discarding them

## Tips

- Use a **2x1** grid to compare a sample before and after annotation
- Use a **2x2** grid to work on four related sections from the same specimen
- **Double-click** to maximize when you need the full workspace for detailed drawing
- **Save All** (`Ctrl+Alt+S`) regularly to protect all your work across cells
