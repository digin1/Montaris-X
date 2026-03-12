"""Session save/restore I/O helpers.

Pure I/O functions with no GUI dependencies — safe for background threads.
"""

import os
import re
import json
from datetime import datetime


def get_base_stem(image_stem):
    """Strip channel suffixes (_ch0, _c1, etc.) from an image stem.

    Matches the naming convention in image_io.py: ``f"{stem}_ch{i}"``.
    """
    return re.sub(r'_ch?\d+$', '', image_stem)


def build_session_folder_name(image_stem):
    """Return a timestamped session folder name for the given image stem."""
    base = get_base_stem(image_stem)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"session_{base}_{ts}"


def find_sessions(image_dir, image_stem):
    """Find session folders in *image_dir* that match *image_stem*.

    Matches by ``get_base_stem()`` so that ``sample_ch0.tif`` finds sessions
    created from ``sample.tif``.

    Returns ``[(folder_path, metadata_dict)]`` sorted newest-first.
    Silently skips corrupt / unreadable sessions.
    """
    base = get_base_stem(image_stem)
    results = []
    try:
        entries = os.listdir(image_dir)
    except OSError:
        return results

    for entry in entries:
        if not entry.startswith('session_'):
            continue
        folder = os.path.join(image_dir, entry)
        meta_path = os.path.join(folder, 'session.json')
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            # Match by base stem
            session_stem = meta.get('image_stem', '')
            if get_base_stem(session_stem) != base:
                continue
            results.append((folder, meta))
        except (OSError, json.JSONDecodeError, KeyError):
            continue

    # Sort newest-first by timestamp field, falling back to folder name
    results.sort(key=lambda x: x[1].get('timestamp', x[0]), reverse=True)
    return results
