"""Entry point for ``python -m montaris``.

The GPU + platform pre-pass lives in ``montaris.app.main`` so every launch
path (pip console_script, ``python -m montaris``, direct import) gets it.
"""
import sys

from montaris.app import main

if __name__ == "__main__":
    sys.exit(main() or 0)
