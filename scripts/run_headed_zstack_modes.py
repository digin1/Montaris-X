"""Drive headed_zstack_modes.py once per import mode in separate processes.

Fresh process per mode so each run starts with no cached memory. Prints a
summary table at the end.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"
DRIVER = ROOT / "scripts" / "headed_zstack_modes.py"

MODES = ["max", "slice", "synced"]


def run(mode: str) -> dict:
    env = {**os.environ, "QT_QPA_PLATFORM": "xcb"}
    proc = subprocess.run(
        [str(PY), str(DRIVER), mode],
        env=env, cwd=str(ROOT),
        capture_output=True, text=True, timeout=600,
    )
    # Find the RESULT line.
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT::"):
            return json.loads(line[len("RESULT::"):])
    sys.stderr.write(f"\n--- {mode} failed ---\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n")
    return {"mode": mode, "error": "no result", "returncode": proc.returncode}


def main():
    rows = []
    for mode in MODES:
        print(f"[{mode}] running…", flush=True)
        r = run(mode)
        rows.append(r)
        print(f"[{mode}] done: {r}", flush=True)

    print("\n=== Summary ===")
    headers = [
        "mode", "n_docs", "sync_groups", "slider_max",
        "z_scrub_ok", "composite_on", "load_s", "mem_peak_mb", "cpu_peak_pct",
    ]
    print(" | ".join(h.rjust(12) for h in headers))
    for r in rows:
        if "error" in r:
            print(f"{r['mode']:>12} | ERROR: {r['error']}")
            continue
        cells = [
            r["mode"], r["n_docs"], r["sync_groups"], r["slider_max"],
            r["z_scrub_ok"], r["composite_on"], r["load_seconds"],
            r["mem_peak_mb"], r["cpu_peak_pct"],
        ]
        print(" | ".join(str(c).rjust(12) for c in cells))


if __name__ == "__main__":
    main()
