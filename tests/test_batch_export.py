import json
import numpy as np
import pytest
from montaris.io.instructions import load_instructions
from montaris.io.roi_io import save_roi_set, load_roi_set
from montaris.layers import ROILayer


class TestInstructions:
    def test_load_instructions(self, tmp_path):
        instr = {
            "version": 1,
            "adjustments": {"brightness": 0.1, "contrast": 1.2},
            "operations": [
                {"type": "fix_overlaps", "priority": "later_wins"},
            ],
        }
        path = tmp_path / "instructions.json"
        with open(path, 'w') as f:
            json.dump(instr, f)
        loaded = load_instructions(str(path))
        assert loaded['version'] == 1
        assert loaded['adjustments']['brightness'] == 0.1


class TestBatchExport:
    def test_export_npz(self, tmp_path):
        roi1 = ROILayer("A", 100, 80, color=(255, 0, 0))
        roi1.mask[10:30, 10:30] = 255
        roi2 = ROILayer("B", 100, 80, color=(0, 255, 0))
        roi2.mask[40:60, 40:60] = 128
        path = tmp_path / "batch_export.npz"
        save_roi_set(str(path), [roi1, roi2])
        loaded = load_roi_set(str(path))
        assert len(loaded) == 2

    def test_export_imagej(self, tmp_path):
        from montaris.io.imagej_roi import mask_to_imagej_roi, write_imagej_roi
        roi = ROILayer("TestROI", 100, 80)
        roi.mask[20:60, 20:80] = 255
        roi_dict = mask_to_imagej_roi(roi.mask, roi.name)
        assert roi_dict is not None
        path = tmp_path / "test.roi"
        write_imagej_roi(roi_dict, str(path))
        assert path.exists()

    def test_save_with_progress_callback(self, tmp_path):
        """Progress callback fires once per ROI."""
        roi1 = ROILayer("X", 50, 50, color=(255, 0, 0))
        roi1.mask[5:15, 5:15] = 255
        roi2 = ROILayer("Y", 50, 50, color=(0, 255, 0))
        roi2.mask[20:30, 20:30] = 128
        roi3 = ROILayer("Z", 50, 50, color=(0, 0, 255))
        roi3.mask[35:45, 35:45] = 200
        path = tmp_path / "progress.npz"
        calls = []
        save_roi_set(str(path), [roi1, roi2, roi3], progress_callback=lambda i: calls.append(i))
        assert calls == [1, 2, 3]

    def test_npz_roundtrip_after_refactor(self, tmp_path):
        """Manual ZIP format is loadable by load_roi_set."""
        roi1 = ROILayer("Alpha", 80, 60, color=(100, 200, 50))
        roi1.mask[10:30, 10:30] = 255
        roi1.opacity = 200
        roi2 = ROILayer("Beta", 80, 60, color=(50, 100, 200))
        roi2.mask[40:55, 40:55] = 128
        roi2.opacity = 100
        path = tmp_path / "roundtrip.npz"
        save_roi_set(str(path), [roi1, roi2])
        loaded = load_roi_set(str(path))
        assert len(loaded) == 2
        assert loaded[0].name == "Alpha"
        assert loaded[1].name == "Beta"
        assert loaded[0].opacity == 200
        assert loaded[1].opacity == 100
        np.testing.assert_array_equal(loaded[0].mask, roi1.mask)
        np.testing.assert_array_equal(loaded[1].mask, roi2.mask)

    def test_save_with_mask_transform(self, tmp_path):
        """mask_transform callback upscales masks on save — pixel-perfect verification."""
        roi = ROILayer("Small", 50, 40, color=(255, 0, 0))
        roi.mask[10:20, 10:20] = 255
        path = tmp_path / "transformed.npz"

        def upscale_2x(mask):
            return np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)

        # Build expected upscaled mask pixel by pixel
        expected = upscale_2x(roi.mask)

        save_roi_set(str(path), [roi], mask_transform=upscale_2x)
        loaded = load_roi_set(str(path))
        assert len(loaded) == 1
        assert loaded[0].mask.shape == (80, 100)
        # Pixel-perfect comparison against expected
        np.testing.assert_array_equal(loaded[0].mask, expected)

    def test_save_with_mask_transform_multiple_rois(self, tmp_path):
        """mask_transform applied to each ROI independently — pixel-perfect."""
        roi1 = ROILayer("A", 60, 40, color=(255, 0, 0))
        roi1.mask[5:15, 10:30] = 255
        roi2 = ROILayer("B", 60, 40, color=(0, 255, 0))
        roi2.mask[20:35, 20:50] = 128
        path = tmp_path / "multi_transform.npz"

        def upscale_3x(mask):
            return np.repeat(np.repeat(mask, 3, axis=0), 3, axis=1)

        expected1 = upscale_3x(roi1.mask)
        expected2 = upscale_3x(roi2.mask)

        save_roi_set(str(path), [roi1, roi2], mask_transform=upscale_3x)
        loaded = load_roi_set(str(path))
        assert len(loaded) == 2
        assert loaded[0].mask.shape == (120, 180)
        assert loaded[1].mask.shape == (120, 180)
        np.testing.assert_array_equal(loaded[0].mask, expected1)
        np.testing.assert_array_equal(loaded[1].mask, expected2)
