"""Tests for the EventLogger diagnostics system."""

import time
import pytest
from montaris.core.event_logger import EventLogger


@pytest.fixture
def logger():
    """Fresh EventLogger instance (not the singleton)."""
    return EventLogger(max_events=5000)


class TestRingBuffer:
    def test_capacity(self, logger):
        """Events beyond max_events evict oldest."""
        logger = EventLogger(max_events=5000)
        for i in range(6000):
            logger.log("test", f"event_{i}")
        assert len(logger._events) == 5000
        # Oldest events should be gone
        assert logger._events[0]['name'] == 'event_1000'

    def test_empty(self, logger):
        assert len(logger._events) == 0


class TestEventStructure:
    def test_basic_event(self, logger):
        logger.log("io", "load_image")
        event = logger._events[0]
        assert event['cat'] == 'io'
        assert event['name'] == 'load_image'
        assert 'ts' in event
        assert 'dur_ms' not in event
        assert 'meta' not in event

    def test_event_with_duration(self, logger):
        logger.log("render", "refresh", duration_ms=42.567)
        event = logger._events[0]
        assert event['dur_ms'] == 42.57

    def test_event_with_metadata(self, logger):
        logger.log("io", "import_zip", duration_ms=100.0, count=5, format="roi")
        event = logger._events[0]
        assert event['meta'] == {'count': 5, 'format': 'roi'}

    def test_event_with_duration_zero(self, logger):
        logger.log("test", "fast_op", duration_ms=0.0)
        assert logger._events[0]['dur_ms'] == 0.0


class TestTimedContextManager:
    def test_timed_logs_duration(self, logger):
        with logger.timed("test", "sleep_op"):
            time.sleep(0.01)
        event = logger._events[0]
        assert event['cat'] == 'test'
        assert event['name'] == 'sleep_op'
        assert event['dur_ms'] > 0

    def test_timed_with_metadata(self, logger):
        with logger.timed("io", "export", format="png"):
            pass
        event = logger._events[0]
        assert event['meta'] == {'format': 'png'}


class TestExportJson:
    def test_export_format(self, logger):
        logger.log("test", "event1")
        logger.log("test", "event2", duration_ms=1.5)
        result = logger.export_json()
        assert result['version'] == 1
        assert 'session' in result
        assert 'events' in result
        assert len(result['events']) == 2

    def test_session_fields(self, logger):
        result = logger.export_json()
        session = result['session']
        assert 'start_time' in session
        assert 'duration_s' in session
        assert 'platform' in session
        assert 'python' in session
        assert 'cpu_count' in session
        assert 'total_events' in session

    def test_export_empty(self, logger):
        result = logger.export_json()
        assert result['events'] == []
        assert result['session']['total_events'] == 0


class TestSingleton:
    def test_singleton(self):
        # Reset singleton for test
        EventLogger._instance = None
        a = EventLogger.instance()
        b = EventLogger.instance()
        assert a is b
        # Clean up
        EventLogger._instance = None


class TestBusyCursorLogAs:
    def test_log_as_parameter(self, qapp, logger):
        """busy_cursor with log_as logs an event."""
        # Monkey-patch singleton for this test
        old = EventLogger._instance
        EventLogger._instance = logger
        try:
            from montaris.core.busy import busy_cursor
            with busy_cursor(log_as="io.test_op"):
                time.sleep(0.005)
            assert len(logger._events) == 1
            event = logger._events[0]
            assert event['cat'] == 'io'
            assert event['name'] == 'test_op'
            assert event['dur_ms'] > 0
        finally:
            EventLogger._instance = old

    def test_no_log_without_log_as(self, qapp, logger):
        """busy_cursor without log_as does not log."""
        old = EventLogger._instance
        EventLogger._instance = logger
        try:
            from montaris.core.busy import busy_cursor
            with busy_cursor():
                pass
            assert len(logger._events) == 0
        finally:
            EventLogger._instance = old
