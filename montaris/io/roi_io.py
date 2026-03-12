import json
import numpy as np
from montaris.layers import ROILayer, _generate_color


def save_roi_set(path, roi_layers, progress_callback=None, mask_transform=None):
    import zipfile
    import io as _io
    # Flatten any layer offsets before saving
    for roi in roi_layers:
        if hasattr(roi, 'flatten_offset'):
            roi.flatten_offset()
    metadata = []
    for roi in roi_layers:
        metadata.append({
            'name': roi.name,
            'color': list(roi.color),
            'opacity': roi.opacity,
        })
    with zipfile.ZipFile(str(path), 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, roi in enumerate(roi_layers):
            mask = roi.mask
            if mask_transform is not None:
                mask = mask_transform(mask)
            buf = _io.BytesIO()
            np.save(buf, mask)
            zf.writestr(f"mask_{i}.npy", buf.getvalue())
            if progress_callback:
                progress_callback(i + 1)
        buf = _io.BytesIO()
        np.save(buf, np.array(json.dumps(metadata)))
        zf.writestr("_metadata.npy", buf.getvalue())


def load_roi_set(path):
    data = np.load(str(path), allow_pickle=False)

    metadata = []
    if '_metadata' in data:
        metadata = json.loads(str(data['_metadata']))

    roi_layers = []
    mask_keys = sorted(
        [k for k in data.files if k.startswith('mask_')],
        key=lambda k: int(k.split('_')[1]),
    )

    for i, key in enumerate(mask_keys):
        mask = data[key]
        h, w = mask.shape

        if i < len(metadata):
            meta = metadata[i]
            name = meta.get('name', f'ROI {i + 1}')
            color = tuple(meta.get('color', _generate_color(i)))
            opacity = meta.get('opacity', 128)
        else:
            name = f'ROI {i + 1}'
            color = _generate_color(i)
            opacity = 128

        roi = ROILayer(name, w, h, color)
        roi.mask = mask
        roi.opacity = opacity
        # Compress immediately to minimize peak memory
        roi.compress()
        roi_layers.append(roi)

    return roi_layers
