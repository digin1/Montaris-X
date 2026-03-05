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
