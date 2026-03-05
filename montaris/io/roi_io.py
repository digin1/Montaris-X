import json
import numpy as np
from montaris.layers import ROILayer, ROI_COLORS


def save_roi_set(path, roi_layers):
    data = {}
    metadata = []
    for i, roi in enumerate(roi_layers):
        data[f"mask_{i}"] = roi.mask
        metadata.append({
            'name': roi.name,
            'color': list(roi.color),
            'opacity': roi.opacity,
        })
    data['_metadata'] = np.array(json.dumps(metadata))
    np.savez_compressed(str(path), **data)


def load_roi_set(path):
    data = np.load(str(path), allow_pickle=False)

    metadata = []
    if '_metadata' in data:
        metadata = json.loads(str(data['_metadata']))

    roi_layers = []
    mask_keys = sorted([k for k in data.files if k.startswith('mask_')])

    for i, key in enumerate(mask_keys):
        mask = data[key]
        h, w = mask.shape

        if i < len(metadata):
            meta = metadata[i]
            name = meta.get('name', f'ROI {i + 1}')
            color = tuple(meta.get('color', ROI_COLORS[i % len(ROI_COLORS)]))
            opacity = meta.get('opacity', 128)
        else:
            name = f'ROI {i + 1}'
            color = ROI_COLORS[i % len(ROI_COLORS)]
            opacity = 128

        roi = ROILayer(name, w, h, color)
        roi.mask = mask
        roi.opacity = opacity
        roi_layers.append(roi)

    return roi_layers
