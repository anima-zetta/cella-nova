#!/usr/bin/env python3
"""Compare PNG frames from Python, reference Rust, and fl-rs Rust."""
import os
import sys
import numpy as np
from PIL import Image

PNG_DIR = "pngs"

SETS = {
    "py":    {"prefix": "py_frame_",    "label": "Python"},
    "rs":    {"prefix": "rs_frame_",    "label": "Reference"},
    "fl_rs": {"prefix": "fl_rs_frame_", "label": "fl-rs"},
}


def load_frame(prefix: str, step: int):
    path = os.path.join(PNG_DIR, f"{prefix}{step:04d}.png")
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path), dtype=np.float64)


def count_frames(prefix: str) -> int:
    n = 0
    while load_frame(prefix, n) is not None:
        n += 1
    return n


def compare_pair(prefix_a: str, label_a: str, prefix_b: str, label_b: str, num_frames: int):
    print(f"\n--- {label_a} vs {label_b} ---")

    # Check sizes of first frame
    a0 = load_frame(prefix_a, 0)
    b0 = load_frame(prefix_b, 0)
    if a0 is None or b0 is None:
        print("  Missing frames, skipping")
        return

    same_size = a0.shape == b0.shape
    target_size = a0.shape
    if not same_size:
        print(f"  Note: different sizes {a0.shape} vs {b0.shape} — resizing B to {target_size}")

    print(f"{'Step':>5}  {'A sum':>10}  {'B sum':>10}  {'Max diff':>9}  {'Px off':>7}  {'Status'}")

    max_pixel_diff = 0
    total_pixels = 0
    differing_pixels = 0

    for step in range(num_frames):
        a = load_frame(prefix_a, step)
        b = load_frame(prefix_b, step)
        if a is None or b is None:
            continue

        if not same_size:
            b = np.array(Image.fromarray(b.astype(np.uint8)).resize(
                (target_size[1], target_size[0]), 0  # 0 = nearest neighbor
            ), dtype=np.float64)

        diff = np.abs(a - b)
        step_max = diff.max()
        step_off = (diff > 0).sum()

        if step_max > max_pixel_diff:
            max_pixel_diff = step_max

        total_pixels += a.size
        differing_pixels += step_off

        status = "OK" if step_max <= 1 else "DIFF"
        print(f"{step:5d}  {a.sum():10.1f}  {b.sum():10.1f}  {step_max:9.1f}  {step_off:7d}  {status}")

    match_pct = (1 - differing_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    print(f"  Total pixels: {total_pixels}, identical: {total_pixels - differing_pixels} ({match_pct:.2f}%)")
    if max_pixel_diff <= 1:
        print(f"  ✅ Match (max diff = {max_pixel_diff:.0f} gray level)")
    else:
        print(f"  ⚠️  Differ by up to {max_pixel_diff:.0f} gray levels")


def main() -> None:
    if not os.path.isdir(PNG_DIR):
        print(f"Error: '{PNG_DIR}/' directory not found.")
        print("Run the generators first:")
        print("  cargo run --bin save_pngs")
        print("  cargo run --bin fl-rs-save-pngs")
        print("  python3 train/save_frames_png.py")
        sys.exit(1)

    # Count available frames for each set
    counts = {name: count_frames(info["prefix"]) for name, info in SETS.items()}
    for name, count in counts.items():
        print(f"{SETS[name]['label']}: {count} frames")

    # Compare each pair using the smaller frame count
    names = list(SETS.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            num = min(counts[na], counts[nb])
            if num > 0:
                compare_pair(
                    SETS[na]["prefix"], SETS[na]["label"],
                    SETS[nb]["prefix"], SETS[nb]["label"],
                    num,
                )


if __name__ == "__main__":
    main()
