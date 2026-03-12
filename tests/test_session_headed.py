#!/usr/bin/env python3
"""Headed GUI test for session save/restore.

Run directly:
    LD_LIBRARY_PATH="" python3 tests/test_session_headed.py

Resource-capped: 2 GB memory, 30s wall-clock timeout.
Auto-quits — no manual interaction needed.
"""

import sys
import os
import signal
import resource
import shutil
import traceback
import faulthandler

# ── Resource caps ─────────────────────────────────────────────────────
# 4 GB data segment (heap) limit — avoids capping virtual address space
# which Qt's thread pool needs for stack reservations
_FOUR_GB = 4 * 1024 * 1024 * 1024
try:
    resource.setrlimit(resource.RLIMIT_DATA, (_FOUR_GB, _FOUR_GB))
except (ValueError, resource.error):
    pass  # some systems don't support RLIMIT_DATA

# 30 second wall-clock kill switch
signal.signal(signal.SIGALRM, lambda *_: os._exit(99))
signal.alarm(30)

# Catch segfaults
faulthandler.enable(all_threads=True)

# ── Imports ───────────────────────────────────────────────────────────
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_project_root)
# Ensure local source takes priority over pip-installed package
sys.path.insert(0, _project_root)

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from montaris.app import apply_dark_theme, MontarisApp, _save_session_from_snapshots
from montaris.layers import ROILayer
from montaris.io.image_io import load_image_stack
from montaris.io.session import find_sessions, get_base_stem

# ── State ─────────────────────────────────────────────────────────────
TIF_PATH = os.path.join(os.getcwd(), "test.tif")
FAILURES = []
_step = [0]


def _fail(msg):
    FAILURES.append(msg)
    print(f"  FAIL: {msg}", file=sys.stderr)


def _ok(msg):
    print(f"  OK: {msg}", file=sys.stderr)


# ── Test steps (driven by QTimer) ────────────────────────────────────

def run_steps(w, app):
    folder = os.path.dirname(TIF_PATH)

    def cleanup_sessions():
        for d in os.listdir(folder):
            if d.startswith("session_"):
                shutil.rmtree(os.path.join(folder, d), ignore_errors=True)

    def step_load_image():
        """Step 1: Load test.tif into the app."""
        print("[Step 1] Loading image...", file=sys.stderr)
        try:
            if not os.path.exists(TIF_PATH):
                _fail(f"test.tif not found at {TIF_PATH}")
                app.quit()
                return
            cleanup_sessions()
            channels = load_image_stack(TIF_PATH)
            skipped = []
            w._initial_session_saved = False
            for name, data in channels:
                w._load_single_channel(name, data, 4, skipped, image_path=TIF_PATH)
            if not w._documents:
                _fail("No documents after load")
            elif w.layer_stack.image_layer is None:
                _fail("image_layer is None after load")
            else:
                _ok(f"Loaded {len(w._documents)} doc(s), shape={w.layer_stack.image_layer.data.shape}")
        except Exception:
            _fail(f"Load crashed:\n{traceback.format_exc()}")
        QTimer.singleShot(200, step_add_rois)

    def step_add_rois():
        """Step 2: Add test ROIs."""
        print("[Step 2] Adding ROIs...", file=sys.stderr)
        try:
            img = w.layer_stack.image_layer
            h, wd = img.data.shape[:2]
            for i in range(3):
                roi = ROILayer(f"Test ROI {i+1}", wd, h)
                y0, x0 = 20 + i * 40, 20 + i * 40
                roi.mask[y0:y0+30, x0:x0+30] = 255
                w.layer_stack.insert_roi(i, roi)
            w.canvas.refresh_overlays()
            w.layer_panel.refresh()
            n = len(w.layer_stack.roi_layers)
            if n != 3:
                _fail(f"Expected 3 ROIs, got {n}")
            else:
                _ok(f"Added {n} ROIs")
        except Exception:
            _fail(f"Add ROIs crashed:\n{traceback.format_exc()}")
        QTimer.singleShot(200, step_save_session)

    def step_save_session():
        """Step 3: Save session progress."""
        print("[Step 3] Saving session...", file=sys.stderr)
        try:
            w.save_session_progress()
            _ok("save_session_progress() returned")
        except Exception:
            _fail(f"Save crashed:\n{traceback.format_exc()}")
        # Wait for background save to finish
        QTimer.singleShot(3000, step_verify_save)

    def step_verify_save():
        """Step 4: Verify session folder was created."""
        print("[Step 4] Verifying save...", file=sys.stderr)
        try:
            sessions = find_sessions(folder, "test")
            if not sessions:
                _fail("No session folder found after save")
            else:
                sf, meta = sessions[0]
                roi_files = meta.get("roi_files", [])
                non_none = [f for f in roi_files if f is not None]
                actual_files = [f for f in os.listdir(sf) if f.endswith(".roi")]
                if len(actual_files) == 0:
                    _fail(f"Session folder has 0 .roi files")
                else:
                    _ok(f"Session saved: {len(actual_files)} .roi files, meta roi_count={meta.get('roi_count')}")
        except Exception:
            _fail(f"Verify save crashed:\n{traceback.format_exc()}")
        QTimer.singleShot(200, step_restore_session)

    def step_restore_session():
        """Step 5: Restore ROIs from the saved session."""
        print("[Step 5] Restoring session...", file=sys.stderr)
        try:
            sessions = find_sessions(folder, "test")
            if not sessions:
                _fail("No sessions to restore")
                QTimer.singleShot(200, step_done)
                return
            sf, meta = sessions[0]
            # Clear existing ROIs first
            w.canvas._selection.clear()
            w.layer_stack.roi_layers.clear()
            w.canvas._active_layer = None
            n_before = len(w.layer_stack.roi_layers)
            w._restore_session_rois(sf, meta)
            n_after = len(w.layer_stack.roi_layers)
            if n_after == 0:
                _fail("Restore produced 0 ROIs")
            else:
                _ok(f"Restored {n_after} ROIs (was {n_before})")
        except Exception:
            _fail(f"Restore crashed:\n{traceback.format_exc()}")
        QTimer.singleShot(200, step_save_again)

    def step_save_again():
        """Step 6: Save again after restore (tests re-save of restored ROIs)."""
        print("[Step 6] Re-saving after restore...", file=sys.stderr)
        try:
            w.save_session_progress()
            _ok("Re-save returned")
        except Exception:
            _fail(f"Re-save crashed:\n{traceback.format_exc()}")
        QTimer.singleShot(3000, step_verify_resave)

    def step_verify_resave():
        """Step 7: Verify second save."""
        print("[Step 7] Verifying re-save...", file=sys.stderr)
        try:
            sessions = find_sessions(folder, "test")
            if len(sessions) < 2:
                _fail(f"Expected 2+ sessions, found {len(sessions)}")
            else:
                _ok(f"Found {len(sessions)} sessions after re-save")
        except Exception:
            _fail(f"Verify re-save crashed:\n{traceback.format_exc()}")
        QTimer.singleShot(200, step_done)

    def step_done():
        """Final: cleanup and report."""
        print("\n[Cleanup] Removing session folders...", file=sys.stderr)
        cleanup_sessions()

        print("\n" + "=" * 50, file=sys.stderr)
        if FAILURES:
            print(f"FAILED — {len(FAILURES)} failure(s):", file=sys.stderr)
            for f in FAILURES:
                print(f"  - {f}", file=sys.stderr)
        else:
            print("ALL STEPS PASSED", file=sys.stderr)
        print("=" * 50, file=sys.stderr)

        app.quit()

    # Kick off
    QTimer.singleShot(500, step_load_image)


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50, file=sys.stderr)
    print("Session headed test — resource capped", file=sys.stderr)
    print(f"  Memory limit: 4 GB (heap)", file=sys.stderr)
    print(f"  Timeout: 30s", file=sys.stderr)
    print(f"  test.tif: {TIF_PATH}", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    app = QApplication(sys.argv)
    apply_dark_theme(app)
    w = MontarisApp()
    w.show()

    run_steps(w, app)
    rc = app.exec()

    signal.alarm(0)  # cancel alarm
    sys.exit(1 if FAILURES else rc)
