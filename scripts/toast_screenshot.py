"""Capture a standalone screenshot of each toast level."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from PySide6.QtWidgets import QApplication

from montaris.widgets.toast import ToastNotification

SHOT_DIR = ROOT / "tests" / "_screenshots" / "toasts"
SHOT_DIR.mkdir(parents=True, exist_ok=True)

app = QApplication.instance() or QApplication(sys.argv)
for level in ("success", "error", "warning", "info"):
    t = ToastNotification(f"This is a {level} toast — filename truncated.", level=level)
    t.setFixedWidth(340)
    t.adjustSize()
    t.show()
    # Let the style engine lay out before grabbing.
    for _ in range(10):
        app.processEvents()
    time.sleep(0.1)
    pix = t.grab()
    out = SHOT_DIR / f"{level}.png"
    pix.save(str(out))
    print(f"saved {out}  ({pix.width()}x{pix.height()})")
    t.close()
    t.deleteLater()
