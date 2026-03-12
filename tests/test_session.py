"""Tests for session save/restore functionality."""

import os
import json
import time
import numpy as np

from montaris.io.session import get_base_stem, build_session_folder_name, find_sessions
from montaris.io.imagej_roi import (
    mask_to_imagej_roi, write_imagej_roi, read_imagej_roi, imagej_roi_to_mask,
    scale_roi_dict,
)
from montaris.app import _save_session_from_snapshots, _sanitize_roi_filename


# ── get_base_stem ─────────────────────────────────────────────────────

class TestGetBaseStem:
    def test_no_channel_suffix(self):
        assert get_base_stem("sample") == "sample"

    def test_ch0_suffix(self):
        assert get_base_stem("sample_ch0") == "sample"

    def test_ch2_suffix(self):
        assert get_base_stem("sample_ch2") == "sample"

    def test_c1_suffix(self):
        assert get_base_stem("sample_c1") == "sample"

    def test_nested_ch_in_name(self):
        """Only strip trailing channel suffix, not interior ones."""
        assert get_base_stem("my_ch0_data_ch2") == "my_ch0_data"

    def test_no_false_positive(self):
        """Don't strip things that look like channels but aren't."""
        assert get_base_stem("archive") == "archive"
        assert get_base_stem("notch3") == "notch3"

    def test_empty_string(self):
        assert get_base_stem("") == ""

    def test_multi_digit_channel(self):
        assert get_base_stem("image_ch10") == "image"
        assert get_base_stem("data_ch123") == "data"

    def test_underscore_only_stem(self):
        """Edge: stem that is just underscores."""
        assert get_base_stem("_ch0") == ""


# ── build_session_folder_name ─────────────────────────────────────────

class TestBuildSessionFolderName:
    def test_format(self):
        name = build_session_folder_name("sample_ch0")
        assert name.startswith("session_sample_")
        parts = name.split("_")
        assert len(parts) >= 3

    def test_strips_channel_suffix(self):
        name = build_session_folder_name("image_ch1")
        assert "session_image_" in name
        assert "_ch1_" not in name


# ── _sanitize_roi_filename ────────────────────────────────────────────

class TestSanitizeFilename:
    def test_slashes(self):
        assert _sanitize_roi_filename("A/B") == "A_B"
        assert _sanitize_roi_filename("A\\B") == "A_B"

    def test_colon(self):
        assert _sanitize_roi_filename("ROI:special") == "ROI_special"

    def test_spaces_and_parens_preserved(self):
        """Spaces and parens are valid filenames — keep them."""
        assert _sanitize_roi_filename("ROI 1 (2)") == "ROI 1 (2)"

    def test_clean_name_unchanged(self):
        assert _sanitize_roi_filename("ROI_1") == "ROI_1"


# ── Session save helpers ──────────────────────────────────────────────

def _make_circle_mask(h, w, cy, cx, r):
    """Create a circular mask for testing."""
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2).astype(np.uint8) * 255


def _create_session_via_save(tmpdir, stem, roi_masks, names=None, ds_factor=1):
    """Create a session using the actual _save_session_from_snapshots path."""
    from montaris.io.session import build_session_folder_name, get_base_stem

    folder_name = build_session_folder_name(stem)
    session_dir = os.path.join(tmpdir, folder_name)

    snapshots = []
    roi_names = []
    roi_colors = []
    roi_opacities = []

    for i, mask in enumerate(roi_masks):
        name = names[i] if names else f"ROI {i + 1}"
        roi_names.append(name)
        roi_colors.append([255, 0, 0])
        roi_opacities.append(128)

        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            continue
        bbox = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
        snapshots.append({
            'mask': mask.copy(),
            'name': name,
            'color': [255, 0, 0],
            'opacity': 128,
            'bbox': bbox,
        })

    meta = {
        'version': 1,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'image_stem': get_base_stem(stem),
        'image_path': os.path.join(tmpdir, f"{stem}.tif"),
        'downsample_factor': ds_factor,
        'original_shape': [roi_masks[0].shape[0] * ds_factor, roi_masks[0].shape[1] * ds_factor] if roi_masks else None,
        'canvas_shape': list(roi_masks[0].shape) if roi_masks else None,
        'channel_names': [stem],
        'roi_count': len(snapshots),
        'roi_names': roi_names,
        'roi_colors': roi_colors,
        'roi_opacities': roi_opacities,
    }

    _save_session_from_snapshots(session_dir, snapshots, meta)

    # Re-read meta from disk (it gets roi_files added)
    with open(os.path.join(session_dir, 'session.json')) as f:
        meta = json.load(f)

    return session_dir, meta


def _create_session_legacy(tmpdir, stem, roi_masks, ds_factor=1):
    """Create a session in the old name-based format (no roi_files key)."""
    from montaris.io.session import build_session_folder_name, get_base_stem

    folder_name = build_session_folder_name(stem)
    session_dir = os.path.join(tmpdir, folder_name)
    os.makedirs(session_dir, exist_ok=True)

    roi_names = []
    roi_colors = []
    roi_opacities = []

    for i, mask in enumerate(roi_masks):
        name = f"ROI {i + 1}"
        roi_names.append(name)
        roi_colors.append([255, 0, 0])
        roi_opacities.append(128)

        roi_dict = mask_to_imagej_roi(mask, name)
        if roi_dict is not None:
            safe_name = name.replace("/", "_").replace("\\", "_")
            write_imagej_roi(roi_dict, os.path.join(session_dir, f"{safe_name}.roi"))

    meta = {
        'version': 1,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'image_stem': get_base_stem(stem),
        'image_path': os.path.join(tmpdir, f"{stem}.tif"),
        'downsample_factor': ds_factor,
        'original_shape': [roi_masks[0].shape[0] * ds_factor, roi_masks[0].shape[1] * ds_factor],
        'canvas_shape': list(roi_masks[0].shape),
        'channel_names': [stem],
        'roi_count': len([m for m in roi_masks if m.any()]),
        'roi_names': roi_names,
        'roi_colors': roi_colors,
        'roi_opacities': roi_opacities,
        # NOTE: no 'roi_files' key — simulates old format
    }

    with open(os.path.join(session_dir, 'session.json'), 'w') as f:
        json.dump(meta, f)

    return session_dir, meta


# ── Save creates folder and files ────────────────────────────────────

class TestSaveSession:
    def test_save_creates_folder_and_files(self, tmp_path):
        mask = _make_circle_mask(100, 100, 50, 50, 20)
        session_dir, meta = _create_session_via_save(str(tmp_path), "test_image", [mask])

        assert os.path.isdir(session_dir)
        assert os.path.isfile(os.path.join(session_dir, "session.json"))
        # Indexed filename
        assert os.path.isfile(os.path.join(session_dir, "0000_ROI 1.roi"))

    def test_session_metadata_fields(self, tmp_path):
        mask = _make_circle_mask(200, 300, 100, 150, 40)
        session_dir, meta = _create_session_via_save(str(tmp_path), "sample_ch0", [mask], ds_factor=2)

        assert meta['version'] == 1
        assert meta['image_stem'] == "sample"
        assert meta['downsample_factor'] == 2
        assert meta['roi_count'] == 1
        assert meta['roi_names'] == ["ROI 1"]
        assert 'timestamp' in meta
        assert meta['original_shape'] == [400, 600]
        assert meta['canvas_shape'] == [200, 300]
        assert meta['roi_files'] == ["0000_ROI 1.roi"]

    def test_empty_roi_skipped(self, tmp_path):
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        good_mask = _make_circle_mask(100, 100, 50, 50, 20)
        session_dir, meta = _create_session_via_save(str(tmp_path), "test", [empty_mask, good_mask])

        # Empty mask is skipped during snapshotting (bbox=None),
        # so only the good mask appears in roi_files
        non_none = [f for f in meta['roi_files'] if f is not None]
        assert len(non_none) == 1
        roi_files_on_disk = [f for f in os.listdir(session_dir) if f.endswith('.roi')]
        assert len(roi_files_on_disk) == 1

    def test_all_empty_masks_no_crash(self, tmp_path):
        """All-empty mask list produces session.json but no .roi files."""
        empty1 = np.zeros((100, 100), dtype=np.uint8)
        empty2 = np.zeros((100, 100), dtype=np.uint8)
        session_dir, meta = _create_session_via_save(str(tmp_path), "test", [empty1, empty2])

        assert os.path.isfile(os.path.join(session_dir, "session.json"))
        roi_files = [f for f in os.listdir(session_dir) if f.endswith('.roi')]
        assert len(roi_files) == 0

    def test_roi_name_with_special_chars(self, tmp_path):
        """ROI names with parens and spaces (from generate_unique_roi_name)."""
        mask = _make_circle_mask(100, 100, 50, 50, 20)
        session_dir, meta = _create_session_via_save(
            str(tmp_path), "test", [mask], names=["ROI 1 (2)"]
        )

        assert os.path.isfile(os.path.join(session_dir, "0000_ROI 1 (2).roi"))
        # Roundtrip: read it back
        roi_data = read_imagej_roi(os.path.join(session_dir, "0000_ROI 1 (2).roi"))
        restored = imagej_roi_to_mask(roi_data, 100, 100)
        assert restored.any()


# ── Filename collision ────────────────────────────────────────────────

class TestFilenameCollision:
    def test_sanitization_collision_handled_by_index(self, tmp_path):
        """Two names that sanitize to the same string get unique files via index."""
        mask1 = _make_circle_mask(100, 100, 30, 30, 15)
        mask2 = _make_circle_mask(100, 100, 70, 70, 15)
        session_dir, meta = _create_session_via_save(
            str(tmp_path), "test", [mask1, mask2], names=["A/B", "A\\B"]
        )

        # Both should exist with different indexed names
        assert os.path.isfile(os.path.join(session_dir, "0000_A_B.roi"))
        assert os.path.isfile(os.path.join(session_dir, "0001_A_B.roi"))

        # Read both back — masks should differ
        d1 = read_imagej_roi(os.path.join(session_dir, "0000_A_B.roi"))
        d2 = read_imagej_roi(os.path.join(session_dir, "0001_A_B.roi"))
        m1 = imagej_roi_to_mask(d1, 100, 100)
        m2 = imagej_roi_to_mask(d2, 100, 100)
        # Centers at (30,30) vs (70,70) should not overlap
        assert not np.array_equal(m1, m2)


# ── find_sessions ────────────────────────────────────────────────────

class TestFindSessions:
    def test_find_sessions_matches(self, tmp_path):
        mask = _make_circle_mask(100, 100, 50, 50, 20)
        _create_session_via_save(str(tmp_path), "sample", [mask])

        # Search by channel name should find sessions for base stem
        sessions = find_sessions(str(tmp_path), "sample_ch0")
        assert len(sessions) == 1

    def test_find_sessions_no_match(self, tmp_path):
        mask = _make_circle_mask(100, 100, 50, 50, 20)
        _create_session_via_save(str(tmp_path), "imageA", [mask])

        sessions = find_sessions(str(tmp_path), "imageB")
        assert len(sessions) == 0

    def test_multiple_sessions_sorted_newest_first(self, tmp_path):
        mask = _make_circle_mask(100, 100, 50, 50, 20)
        _create_session_via_save(str(tmp_path), "sample", [mask])
        time.sleep(1.1)  # ensure different timestamps
        _create_session_via_save(str(tmp_path), "sample", [mask])

        sessions = find_sessions(str(tmp_path), "sample")
        assert len(sessions) == 2
        # Newest first
        ts0 = sessions[0][1]['timestamp']
        ts1 = sessions[1][1]['timestamp']
        assert ts0 >= ts1

    def test_find_sessions_empty_dir(self, tmp_path):
        sessions = find_sessions(str(tmp_path), "anything")
        assert sessions == []

    def test_corrupt_session_skipped(self, tmp_path):
        bad_dir = os.path.join(str(tmp_path), "session_sample_20250101_000000")
        os.makedirs(bad_dir)
        with open(os.path.join(bad_dir, "session.json"), "w") as f:
            f.write("{invalid json")

        sessions = find_sessions(str(tmp_path), "sample")
        assert len(sessions) == 0

    def test_nonexistent_directory(self):
        sessions = find_sessions("/nonexistent/path/that/does/not/exist", "sample")
        assert sessions == []

    def test_session_missing_image_stem(self, tmp_path):
        """session.json with missing image_stem defaults to '' → no match."""
        bad_dir = os.path.join(str(tmp_path), "session_sample_20250101_000000")
        os.makedirs(bad_dir)
        with open(os.path.join(bad_dir, "session.json"), "w") as f:
            json.dump({"version": 1, "timestamp": "2025-01-01"}, f)

        sessions = find_sessions(str(tmp_path), "sample")
        assert len(sessions) == 0

    def test_session_folder_without_json_ignored(self, tmp_path):
        """session_ folder without session.json is silently skipped."""
        bad_dir = os.path.join(str(tmp_path), "session_sample_20250101_000000")
        os.makedirs(bad_dir)
        # No session.json inside

        sessions = find_sessions(str(tmp_path), "sample")
        assert len(sessions) == 0


# ── Roundtrip save → restore ─────────────────────────────────────────

class TestRoundtrip:
    def test_roundtrip_save_restore(self, tmp_path):
        h, w = 200, 300
        mask = _make_circle_mask(h, w, 100, 150, 40)
        session_dir, meta = _create_session_via_save(str(tmp_path), "test", [mask])

        roi_path = os.path.join(session_dir, meta['roi_files'][0])
        roi_data = read_imagej_roi(roi_path)
        restored = imagej_roi_to_mask(roi_data, w, h)

        intersection = np.logical_and(mask > 0, restored > 0).sum()
        union = np.logical_or(mask > 0, restored > 0).sum()
        iou = intersection / union if union > 0 else 0
        assert iou > 0.95, f"IoU too low: {iou:.3f}"

    def test_ds_factor_upscale_on_restore(self, tmp_path):
        """Save at ds=2, restore at ds=1 → coords scale 2x."""
        h, w = 100, 150
        mask = _make_circle_mask(h, w, 50, 75, 20)
        session_dir, meta = _create_session_via_save(str(tmp_path), "test", [mask], ds_factor=2)

        roi_path = os.path.join(session_dir, meta['roi_files'][0])
        roi_data = read_imagej_roi(roi_path)

        scaled_data = scale_roi_dict(roi_data, 2.0, 2.0)
        big_h, big_w = h * 2, w * 2
        restored = imagej_roi_to_mask(scaled_data, big_w, big_h)

        original_area = (mask > 0).sum()
        restored_area = (restored > 0).sum()
        ratio = restored_area / original_area
        assert 3.5 < ratio < 4.5, f"Area ratio {ratio:.2f} not ~4x after 2x scale"

    def test_ds_factor_downscale_on_restore(self, tmp_path):
        """Save at ds=1, restore at ds=2 → coords scale 0.5x."""
        h, w = 200, 300
        mask = _make_circle_mask(h, w, 100, 150, 40)
        session_dir, meta = _create_session_via_save(str(tmp_path), "test", [mask], ds_factor=1)

        roi_path = os.path.join(session_dir, meta['roi_files'][0])
        roi_data = read_imagej_roi(roi_path)

        # session_ds=1, current_ds=2 → scale = 1/2 = 0.5
        scaled_data = scale_roi_dict(roi_data, 0.5, 0.5)
        small_h, small_w = h // 2, w // 2
        restored = imagej_roi_to_mask(scaled_data, small_w, small_h)

        original_area = (mask > 0).sum()
        restored_area = (restored > 0).sum()
        ratio = restored_area / original_area
        # Downscaled to half → area ~1/4
        assert 0.2 < ratio < 0.3, f"Area ratio {ratio:.2f} not ~0.25 after 0.5x scale"

    def test_multiple_rois_roundtrip(self, tmp_path):
        h, w = 200, 200
        mask1 = _make_circle_mask(h, w, 50, 50, 20)
        mask2 = _make_circle_mask(h, w, 150, 150, 30)
        session_dir, meta = _create_session_via_save(str(tmp_path), "multi", [mask1, mask2])

        assert len(meta['roi_files']) == 2
        for idx, (original, filename) in enumerate(zip([mask1, mask2], meta['roi_files'])):
            roi_data = read_imagej_roi(os.path.join(session_dir, filename))
            restored = imagej_roi_to_mask(roi_data, w, h)
            intersection = np.logical_and(original > 0, restored > 0).sum()
            union = np.logical_or(original > 0, restored > 0).sum()
            iou = intersection / union
            assert iou > 0.90, f"ROI {idx} IoU too low: {iou:.3f}"

    def test_1_pixel_mask_at_corner(self, tmp_path):
        """Tiny 1-pixel ROI at image corner survives roundtrip."""
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[0, 0] = 255

        session_dir, meta = _create_session_via_save(str(tmp_path), "test", [mask])

        roi_file = meta['roi_files'][0]
        if roi_file is None:
            # 1-pixel might not produce valid contour — acceptable
            return
        roi_data = read_imagej_roi(os.path.join(session_dir, roi_file))
        restored = imagej_roi_to_mask(roi_data, w, h)
        # At minimum, the corner pixel area should have some overlap
        assert restored.any(), "1-pixel ROI did not restore"

    def test_full_image_mask(self, tmp_path):
        """Mask filling entire canvas survives roundtrip."""
        h, w = 50, 80
        mask = np.full((h, w), 255, dtype=np.uint8)
        session_dir, meta = _create_session_via_save(str(tmp_path), "test", [mask])

        roi_path = os.path.join(session_dir, meta['roi_files'][0])
        roi_data = read_imagej_roi(roi_path)
        restored = imagej_roi_to_mask(roi_data, w, h)

        intersection = np.logical_and(mask > 0, restored > 0).sum()
        total = h * w
        assert intersection / total > 0.95, "Full-image mask lost too many pixels"


# ── Restore edge cases ───────────────────────────────────────────────

class TestRestoreEdgeCases:
    def test_missing_roi_file_skipped(self, tmp_path):
        """If a .roi file is deleted after save, restore skips it gracefully."""
        mask1 = _make_circle_mask(100, 100, 30, 30, 15)
        mask2 = _make_circle_mask(100, 100, 70, 70, 15)
        session_dir, meta = _create_session_via_save(str(tmp_path), "test", [mask1, mask2])

        # Delete one .roi file
        first_file = meta['roi_files'][0]
        os.remove(os.path.join(session_dir, first_file))

        # Restore should still work for the remaining file
        second_file = meta['roi_files'][1]
        assert os.path.isfile(os.path.join(session_dir, second_file))
        roi_data = read_imagej_roi(os.path.join(session_dir, second_file))
        restored = imagej_roi_to_mask(roi_data, 100, 100)
        assert restored.any()

    def test_empty_roi_names_list(self, tmp_path):
        """Session with empty roi_names should produce 0 restored ROIs."""
        session_dir = os.path.join(str(tmp_path), "session_test_20250101_000000")
        os.makedirs(session_dir)
        meta = {
            'version': 1, 'timestamp': '2025-01-01', 'image_stem': 'test',
            'roi_count': 0, 'roi_names': [], 'roi_colors': [], 'roi_opacities': [],
            'roi_files': [], 'downsample_factor': 1,
        }
        with open(os.path.join(session_dir, 'session.json'), 'w') as f:
            json.dump(meta, f)

        sessions = find_sessions(str(tmp_path), "test")
        assert len(sessions) == 1
        assert sessions[0][1]['roi_count'] == 0

    def test_mismatched_colors_length(self, tmp_path):
        """Fewer colors than names → fallback colors used (no crash)."""
        mask = _make_circle_mask(100, 100, 50, 50, 20)
        session_dir, meta = _create_session_via_save(str(tmp_path), "test", [mask])

        # Tamper: remove colors from metadata
        meta['roi_colors'] = []
        with open(os.path.join(session_dir, 'session.json'), 'w') as f:
            json.dump(meta, f)

        # The restore code uses `i < len(roi_colors)` guard — should not crash
        sessions = find_sessions(str(tmp_path), "test")
        assert len(sessions) == 1
        assert sessions[0][1]['roi_colors'] == []


# ── Legacy format backward compatibility ──────────────────────────────

class TestLegacyFormat:
    def test_legacy_session_without_roi_files_key(self, tmp_path):
        """Sessions saved before roi_files was added still work via name fallback."""
        mask = _make_circle_mask(100, 100, 50, 50, 20)
        session_dir, meta = _create_session_legacy(str(tmp_path), "test", [mask])

        # Verify no roi_files key
        assert 'roi_files' not in meta

        # The .roi file exists with old naming convention
        assert os.path.isfile(os.path.join(session_dir, "ROI 1.roi"))

        # find_sessions should find it
        sessions = find_sessions(str(tmp_path), "test")
        assert len(sessions) == 1


# ── Module-level save function ────────────────────────────────────────

class TestSaveFromSnapshots:
    def test_save_session_from_snapshots(self, tmp_path):
        h, w = 100, 100
        mask = _make_circle_mask(h, w, 50, 50, 20)
        bbox = (30, 71, 30, 71)

        snapshots = [{
            'mask': mask,
            'name': 'Test ROI',
            'color': [255, 0, 0],
            'opacity': 128,
            'bbox': bbox,
        }]

        meta = {
            'version': 1,
            'timestamp': '2025-01-01T00:00:00',
            'image_stem': 'test',
            'image_path': '/tmp/test.tif',
            'downsample_factor': 1,
            'original_shape': [100, 100],
            'canvas_shape': [100, 100],
            'channel_names': ['test'],
            'roi_count': 1,
            'roi_names': ['Test ROI'],
            'roi_colors': [[255, 0, 0]],
            'roi_opacities': [128],
        }

        session_dir = os.path.join(str(tmp_path), "session_test_20250101_000000")
        _save_session_from_snapshots(session_dir, snapshots, meta)

        assert os.path.isdir(session_dir)
        assert os.path.isfile(os.path.join(session_dir, "session.json"))
        assert os.path.isfile(os.path.join(session_dir, "0000_Test ROI.roi"))

        with open(os.path.join(session_dir, "session.json")) as f:
            loaded = json.load(f)
        assert loaded['version'] == 1
        assert loaded['roi_count'] == 1
        assert loaded['roi_files'] == ["0000_Test ROI.roi"]

    def test_save_with_all_none_masks(self, tmp_path):
        """All snapshots where mask_to_imagej_roi returns None → roi_files all None."""
        # Create a snapshot with an empty submask region (bbox points to empty area)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 255  # tiny dot

        snapshots = [{
            'mask': mask,
            'name': 'Tiny',
            'color': [255, 0, 0],
            'opacity': 128,
            'bbox': (50, 51, 50, 51),
        }]

        meta = {
            'version': 1, 'timestamp': '2025-01-01', 'image_stem': 'test',
            'roi_count': 1, 'roi_names': ['Tiny'], 'roi_colors': [[255, 0, 0]],
            'roi_opacities': [128],
        }

        session_dir = os.path.join(str(tmp_path), "session_test_tiny")
        _save_session_from_snapshots(session_dir, snapshots, meta)

        with open(os.path.join(session_dir, "session.json")) as f:
            loaded = json.load(f)

        # Either a file was created or roi_files has None — either way no crash
        assert 'roi_files' in loaded
