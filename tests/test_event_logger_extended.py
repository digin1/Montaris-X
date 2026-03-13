"""Extended tests for montaris.core.event_logger — memory logging, export, app stats."""

import time
import pytest

from montaris.core.event_logger import EventLogger, _get_rss_mb


@pytest.fixture
def logger():
    """Fresh EventLogger instance (not the singleton)."""
    return EventLogger(max_events=5000)


# ---------------------------------------------------------------------------
# 1. log_mem — memory snapshot logging
# ---------------------------------------------------------------------------

class TestLogMem:
    def test_log_mem_records_event(self, logger):
        logger.log_mem("test_checkpoint")
        assert len(logger._events) == 1
        event = logger._events[0]
        assert event['cat'] == 'memory'
        assert event['name'] == 'test_checkpoint'

    def test_log_mem_includes_rss(self, logger):
        logger.log_mem("snapshot")
        event = logger._events[0]
        # rss_mb is passed as a keyword to log(), so it's in meta
        assert 'meta' in event
        assert 'rss_mb' in event['meta']
        assert isinstance(event['meta']['rss_mb'], float)

    def test_log_mem_tracks_hwm(self, logger):
        logger.log_mem("first")
        event = logger._events[0]
        assert 'hwm_mb' in event['meta']
        assert event['meta']['hwm_mb'] >= event['meta']['rss_mb']

    def test_log_mem_returns_mb(self, logger):
        mb = logger.log_mem("check")
        assert isinstance(mb, float)
        assert mb >= 0.0

    def test_log_mem_hwm_never_decreases(self, logger):
        logger.log_mem("a")
        hwm1 = logger._hwm_mb
        logger.log_mem("b")
        hwm2 = logger._hwm_mb
        assert hwm2 >= hwm1

    def test_log_mem_extra_metadata(self, logger):
        logger.log_mem("with_extra", roi_count=42)
        event = logger._events[0]
        assert event['meta']['roi_count'] == 42


# ---------------------------------------------------------------------------
# 2. export_json — dict structure
# ---------------------------------------------------------------------------

class TestExportJsonStructure:
    def test_top_level_keys(self, logger):
        result = logger.export_json()
        assert 'version' in result
        assert 'session' in result
        assert 'memory_timeline' in result
        assert 'transform_ops' in result
        assert 'events' in result

    def test_version_is_2(self, logger):
        result = logger.export_json()
        assert result['version'] == 2

    def test_session_keys(self, logger):
        result = logger.export_json()
        session = result['session']
        required_keys = [
            'start_time', 'duration_s', 'platform', 'python',
            'cpu_count', 'total_events', 'current_rss_mb', 'hwm_rss_mb',
        ]
        for key in required_keys:
            assert key in session, f"Missing session key: {key}"

    def test_events_list_matches(self, logger):
        logger.log("io", "load")
        logger.log("render", "refresh", duration_ms=5.0)
        result = logger.export_json()
        assert len(result['events']) == 2

    def test_memory_timeline_populated(self, logger):
        logger.log_mem("snap1")
        logger.log_mem("snap2")
        result = logger.export_json()
        assert len(result['memory_timeline']) == 2
        for entry in result['memory_timeline']:
            assert 'ts' in entry
            assert 'label' in entry

    def test_transform_ops_populated(self, logger):
        logger.log("transform", "rotate", duration_ms=10.0, angle=90)
        result = logger.export_json()
        assert len(result['transform_ops']) == 1
        assert result['transform_ops'][0]['name'] == 'rotate'


# ---------------------------------------------------------------------------
# 3. export_json with app object — ROI stats
# ---------------------------------------------------------------------------

class _FakeROI:
    def __init__(self, compressed, mask_nbytes, rle_len):
        self.is_compressed = compressed
        self._mask = None if compressed else _FakeMask(mask_nbytes)
        self._rle_data = b'\x00' * rle_len if compressed else None


class _FakeMask:
    def __init__(self, nbytes):
        self.nbytes = nbytes


class _FakeImageLayer:
    def __init__(self):
        self.shape = (512, 1024)


class _FakeLayerStack:
    def __init__(self):
        self.image_layer = _FakeImageLayer()
        self.roi_layers = [
            _FakeROI(compressed=False, mask_nbytes=1024 * 1024, rle_len=0),
            _FakeROI(compressed=True, mask_nbytes=0, rle_len=5000),
            _FakeROI(compressed=True, mask_nbytes=0, rle_len=3000),
        ]


class _FakeApp:
    def __init__(self):
        self.layer_stack = _FakeLayerStack()


class TestExportJsonWithApp:
    def test_app_stats_included(self, logger):
        app = _FakeApp()
        result = logger.export_json(app=app)
        session = result['session']
        assert session['total_rois'] == 3
        assert session['rois_compressed'] == 2
        assert session['rois_decompressed'] == 1
        assert 'image_dimensions' in session
        assert session['image_dimensions'] == [1024, 512]

    def test_app_mask_memory(self, logger):
        app = _FakeApp()
        result = logger.export_json(app=app)
        session = result['session']
        expected_mb = round(1024 * 1024 / (1024 * 1024), 1)  # 1.0
        assert session['decompressed_mask_mb'] == expected_mb

    def test_app_rle_data_size(self, logger):
        app = _FakeApp()
        result = logger.export_json(app=app)
        session = result['session']
        expected = round((5000 + 3000) / (1024 * 1024), 2)
        assert session['rle_data_mb'] == expected

    def test_export_without_app(self, logger):
        """Exporting without app should not include ROI stats."""
        result = logger.export_json()
        session = result['session']
        assert 'total_rois' not in session
        assert 'image_dimensions' not in session


# ---------------------------------------------------------------------------
# 4. Session timing
# ---------------------------------------------------------------------------

class TestSessionTiming:
    def test_session_start_recorded(self, logger):
        assert logger._session_start > 0
        assert logger._session_start <= time.time()

    def test_duration_positive(self, logger):
        time.sleep(0.01)
        result = logger.export_json()
        assert result['session']['duration_s'] >= 0.0

    def test_start_time_formatted(self, logger):
        result = logger.export_json()
        start = result['session']['start_time']
        # Should be ISO-like format: YYYY-MM-DDTHH:MM:SS
        assert 'T' in start
        assert len(start) == 19


# ---------------------------------------------------------------------------
# 5. timed_mem context manager
# ---------------------------------------------------------------------------

class TestTimedMem:
    def test_timed_mem_records_duration_and_memory(self, logger):
        with logger.timed_mem("test", "mem_op"):
            time.sleep(0.01)
        assert len(logger._events) == 1
        event = logger._events[0]
        assert event['cat'] == 'test'
        assert event['name'] == 'mem_op'
        assert event['dur_ms'] > 0
        assert 'meta' in event
        assert 'mem_before_mb' in event['meta']
        assert 'mem_after_mb' in event['meta']
        assert 'mem_delta_mb' in event['meta']
