import numpy as np
import pytest
from montaris.layers import ROILayer
from montaris.core.roi_ops import fix_overlaps, compute_overlap_map, find_overlapping_pairs


class TestFixOverlaps:
    def test_no_overlap(self):
        roi1 = ROILayer("A", 100, 100)
        roi2 = ROILayer("B", 100, 100)
        roi1.mask[10:30, 10:30] = 255
        roi2.mask[50:70, 50:70] = 255
        fix_overlaps([roi1, roi2])
        assert roi1.mask.sum() > 0
        assert roi2.mask.sum() > 0

    def test_later_wins(self):
        roi1 = ROILayer("A", 100, 100)
        roi2 = ROILayer("B", 100, 100)
        roi1.mask[20:40, 20:40] = 255
        roi2.mask[30:50, 30:50] = 255
        fix_overlaps([roi1, roi2], priority="later_wins")
        # Overlapping region (30:40, 30:40) should belong to roi2
        overlap_region = roi1.mask[30:40, 30:40]
        assert overlap_region.sum() == 0

    def test_earlier_wins(self):
        roi1 = ROILayer("A", 100, 100)
        roi2 = ROILayer("B", 100, 100)
        roi1.mask[20:40, 20:40] = 255
        roi2.mask[30:50, 30:50] = 255
        fix_overlaps([roi1, roi2], priority="earlier_wins")
        # Overlapping region should belong to roi1
        overlap_region = roi2.mask[30:40, 30:40]
        assert overlap_region.sum() == 0

    def test_single_roi(self):
        roi = ROILayer("A", 100, 100)
        roi.mask[10:30, 10:30] = 255
        before = roi.mask.copy()
        fix_overlaps([roi])
        assert np.array_equal(roi.mask, before)


class TestOverlapMap:
    def test_no_overlap(self):
        roi1 = ROILayer("A", 100, 100)
        roi2 = ROILayer("B", 100, 100)
        roi1.mask[10:20, 10:20] = 255
        roi2.mask[50:60, 50:60] = 255
        overlap = compute_overlap_map([roi1, roi2])
        assert overlap.max() == 1

    def test_overlap(self):
        roi1 = ROILayer("A", 100, 100)
        roi2 = ROILayer("B", 100, 100)
        roi1.mask[20:40, 20:40] = 255
        roi2.mask[30:50, 30:50] = 255
        overlap = compute_overlap_map([roi1, roi2])
        assert overlap.max() == 2


class TestFindOverlappingPairs:
    def test_finds_pair(self):
        roi1 = ROILayer("A", 100, 100)
        roi2 = ROILayer("B", 100, 100)
        roi1.mask[20:40, 20:40] = 255
        roi2.mask[30:50, 30:50] = 255
        pairs = find_overlapping_pairs([roi1, roi2])
        assert (0, 1) in pairs

    def test_no_pair(self):
        roi1 = ROILayer("A", 100, 100)
        roi2 = ROILayer("B", 100, 100)
        roi1.mask[10:20, 10:20] = 255
        roi2.mask[50:60, 50:60] = 255
        pairs = find_overlapping_pairs([roi1, roi2])
        assert len(pairs) == 0
