import time
from contextlib import contextmanager
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt


@contextmanager
def busy_cursor(status_msg=None, window=None):
    """Show WaitCursor and optional statusbar message during a blocking operation."""
    QApplication.setOverrideCursor(Qt.WaitCursor)
    if status_msg and window and hasattr(window, 'statusbar'):
        window.statusbar.showMessage(status_msg)
    QApplication.processEvents()
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()
        if status_msg and window and hasattr(window, 'statusbar'):
            window.statusbar.clearMessage()


def should_process_events(last_time, interval=0.1):
    """Return True and new timestamp if interval elapsed, else False and same timestamp."""
    now = time.monotonic()
    if now - last_time >= interval:
        QApplication.processEvents()
        return now
    return last_time
