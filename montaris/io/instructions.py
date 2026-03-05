"""Load and apply JSON instruction files for batch operations."""

import json
from pathlib import Path


def load_instructions(path):
    """Load a JSON instructions file.

    Expected format:
    {
        "version": 1,
        "image_path": "path/to/image.tif",
        "roi_path": "path/to/rois.npz",
        "adjustments": {
            "brightness": 0.1,
            "contrast": 1.2,
            "exposure": 0.0,
            "gamma": 1.0
        },
        "display_mode": "composite_rgb",
        "operations": [
            {"type": "fix_overlaps", "priority": "later_wins"},
            {"type": "export", "format": "npz", "path": "output/rois.npz"},
            {"type": "export", "format": "imagej", "path": "output/"},
            {"type": "export", "format": "png", "path": "output/overlay.png"}
        ]
    }
    """
    with open(path, 'r') as f:
        return json.load(f)


def apply_instructions(app, instructions):
    """Apply instructions to a MontarisApp instance.

    Args:
        app: MontarisApp instance
        instructions: dict from load_instructions

    Returns:
        list of log messages
    """
    log = []

    # Load image
    image_path = instructions.get('image_path')
    if image_path:
        from montaris.io.image_io import load_image
        from montaris.layers import ImageLayer
        import os
        data = load_image(image_path)
        app.layer_stack.set_image(ImageLayer(os.path.basename(image_path), data))
        app.canvas.refresh_image()
        log.append(f"Loaded image: {image_path}")

    # Load ROIs
    roi_path = instructions.get('roi_path')
    if roi_path:
        from montaris.io.roi_io import load_roi_set
        rois = load_roi_set(roi_path)
        for roi in rois:
            app.layer_stack.add_roi(roi)
        app.canvas.refresh_overlays()
        log.append(f"Loaded {len(rois)} ROIs from {roi_path}")

    # Apply adjustments
    adj_dict = instructions.get('adjustments')
    if adj_dict:
        from montaris.core.adjustments import ImageAdjustments
        adj = ImageAdjustments(**adj_dict)
        if hasattr(app, 'adjustments_panel'):
            app.adjustments_panel._adjustments = adj
            app.adjustments_panel._sync_sliders()
            app.adjustments_panel.adjustments_changed.emit(adj)
        log.append(f"Applied adjustments: {adj_dict}")

    # Execute operations
    for op in instructions.get('operations', []):
        op_type = op.get('type')

        if op_type == 'fix_overlaps':
            from montaris.core.roi_ops import fix_overlaps
            priority = op.get('priority', 'later_wins')
            fix_overlaps(app.layer_stack.roi_layers, priority)
            app.canvas.refresh_overlays()
            log.append(f"Fixed overlaps ({priority})")

        elif op_type == 'export':
            fmt = op.get('format', 'npz')
            path = op.get('path', '')

            if fmt == 'npz':
                from montaris.io.roi_io import save_roi_set
                save_roi_set(path, app.layer_stack.roi_layers)
                log.append(f"Exported NPZ to {path}")

            elif fmt == 'imagej':
                from montaris.io.imagej_roi import mask_to_imagej_roi, write_imagej_roi
                out_dir = Path(path)
                out_dir.mkdir(parents=True, exist_ok=True)
                for i, roi in enumerate(app.layer_stack.roi_layers):
                    roi_dict = mask_to_imagej_roi(roi.mask, roi.name)
                    if roi_dict:
                        write_imagej_roi(roi_dict, out_dir / f"{roi.name}.roi")
                log.append(f"Exported ImageJ ROIs to {path}")

            elif fmt == 'png':
                app.export_roi_png_to(path)
                log.append(f"Exported PNG to {path}")

    return log
