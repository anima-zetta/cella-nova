#!/usr/bin/env python3
"""Compare Rust and Python PNG frames pixel-by-pixel."""
import os
import sys
import numpy as np
from PIL import Image

PNG_DIR = "pngs"
PY_PREFIX = "py_frame_"
RS_PREFIX = "rs_frame_"
NUM_FRAMES = 51


def load_frame(prefix: str, step: int):
    path = os.path.join(PNG_DIR, f"{prefix}{step:04d}.png")
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return None
    return np.array(Image.open(path), dtype=np.float64)


def main() -> None:
    if not os.path.isdir(PNG_DIR):
        print(f"Error: '{PNG_DIR}/' directory not found.")
        print("Run the generators first:")
        print("  cargo run --bin save_pngs")
        print("  python3 train/save_frames_png.py")
        sys.exit(1)

    max_pixel_diff = 0
    max_diff_step = -1
    total_pixels = 0
    differing_pixels = 0

    print(f"Comparing {NUM_FRAMES} frames in '{PNG_DIR}/'...\n")
    print(f"{'Step':>5}  {'Py sum':>10}  {'Rs sum':>10}  {'Max diff':>9}  {'Px off':>7}  {'Status'}")

    for step in range(NUM_FRAMES):
        py = load_frame(PY_PREFIX, step)
        rs = load_frame(RS_PREFIX, step)
        if py is None or rs is None:
            continue

        diff = np.abs(py - rs)
        step_max = diff.max()
        step_off = (diff > 0).sum()

        if step_max > max_pixel_diff:
            max_pixel_diff = step_max
            max_diff_step = step

        total_pixels += py.size
        differing_pixels += step_off

        status = "OK" if step_max <= 1 else "DIFF"
        print(f"{step:5d}  {py.sum():10.1f}  {rs.sum():10.1f}  {step_max:9.1f}  {step_off:7d}  {status}")

    match_pct = (1 - differing_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    print("\n--- Summary ---")
    print(f"Total pixels compared: {total_pixels}")
    print(f"Identical pixels:      {total_pixels - differing_pixels} ({match_pct:.2f}%)")
    print(f"Differing pixels:     {differing_pixels} ({100 - match_pct:.2f}%)")
    print(f"Max pixel difference:  {max_pixel_diff:.0f} (at step {max_diff_step})")

    if max_pixel_diff <= 1:
        print("\n✅ Frames match (max diff <= 1 gray level)")
    else:
        print(f"\n⚠️  Frames differ by up to {max_pixel_diff:.0f} gray levels")


if __name__ == "__main__":
    main()
