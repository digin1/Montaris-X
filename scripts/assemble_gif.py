"""Assemble individual PNG frames into an optimized demo GIF.

Run from the repository root:
    python scripts/assemble_gif.py
"""
import os
import glob
from PIL import Image

# Ensure working directory is the repo root so relative paths work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
os.chdir(REPO_ROOT)

FRAME_DIR = os.path.join("docs", "frames")
OUTPUT = os.path.join("docs", "demo.gif")
# Target width for GIF frames (smaller = smaller file)
TARGET_WIDTH = 800


def main():
    # Load frame files in sorted order
    pngs = sorted(glob.glob(os.path.join(FRAME_DIR, "*.png")))
    if not pngs:
        print("No PNG frames found in", FRAME_DIR)
        return

    # Load durations
    dur_path = os.path.join(FRAME_DIR, "durations.txt")
    durations = []
    with open(dur_path) as f:
        for line in f:
            line = line.strip()
            if line:
                durations.append(int(line))

    if len(durations) != len(pngs):
        print(f"Warning: {len(pngs)} PNGs but {len(durations)} durations")
        # Pad or trim
        while len(durations) < len(pngs):
            durations.append(1200)
        durations = durations[:len(pngs)]

    # First pass: determine consistent target height from first frame
    first = Image.open(pngs[0])
    fw, fh = first.size
    TARGET_HEIGHT = int(TARGET_WIDTH * fh / fw)
    first.close()

    print(f"Loading {len(pngs)} frames (target: {TARGET_WIDTH}x{TARGET_HEIGHT})...")
    frames = []
    for i, path in enumerate(pngs):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
        frames.append(img)
        name = os.path.basename(path)
        print(f"  {i+1:>2}. {name} ({w}x{h} -> {TARGET_WIDTH}x{TARGET_HEIGHT})")

    print(f"\nQuantizing to 256 colors...")
    quantized = []
    for f in frames:
        quantized.append(f.quantize(colors=256, method=Image.MEDIANCUT))

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    print(f"Saving GIF to {OUTPUT}...")
    quantized[0].save(
        OUTPUT,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )

    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    total_dur = sum(durations) / 1000
    print(f"\nDone!")
    print(f"  {len(frames)} frames, {total_dur:.1f}s total")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Output: {OUTPUT}")


if __name__ == "__main__":
    main()
