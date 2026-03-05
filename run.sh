#!/bin/bash
# Launch Montaris-X (strips MATLAB Runtime from LD_LIBRARY_PATH to avoid libstdc++ conflicts)
DIR="$(cd "$(dirname "$0")" && pwd)"
LD_LIBRARY_PATH="" exec python3 "$DIR/main.py" "$@"
