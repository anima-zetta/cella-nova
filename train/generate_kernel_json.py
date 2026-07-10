#!/usr/bin/env python3
"""
Generate kernel and seed data for the glider creature.

Creates:
  kernels/glider.bin   — the glider's FFT convolution kernels (its "genome")
  seed/glider.json     — the glider's initial state (its "body")

The Rust simulation loads both files instead of recomputing at startup.
"""

import json
import math
import os
import struct

import numpy as np
import numpy.typing as npt

GRID_SIZE: int = 512
NUM_KERNELS: int = 9
NUM_CHANNELS: int = 3

GLOBAL_R = 42.0
RADII = np.array([0.8, 0.6, 1.0, 0.7, 0.5, 0.9, 0.65, 0.55, 0.85])

A = np.array([
    [0.0, 0.6, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.7, 0.0],
    [0.0, 0.55, 0.0],
    [0.0, 0.45, 0.0],
    [0.0, 0.65, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.6, 0.0],
    [0.0, 0.55, 0.0],
])

W = np.array([
    [0.08, 0.06, 0.01],
    [0.07, 0.05, 0.01],
    [0.09, 0.07, 0.01],
    [0.08, 0.06, 0.01],
    [0.07, 0.05, 0.01],
    [0.09, 0.07, 0.01],
    [0.08, 0.06, 0.01],
    [0.07, 0.05, 0.01],
    [0.08, 0.06, 0.01],
])

B = np.array([
    [0.8, -0.3, 0.0],
    [0.7, -0.25, 0.0],
    [0.9, -0.35, 0.0],
    [0.75, -0.3, 0.0],
    [0.65, -0.2, 0.0],
    [0.85, -0.35, 0.0],
    [0.7, -0.25, 0.0],
    [0.6, -0.2, 0.0],
    [0.8, -0.3, 0.0],
])

# Seed channel config (matches seed/glider.json)
SEED_CHANNELS = [
    {"sigma": 0.25, "offset_x": 0.0, "offset_y": 0.0},
    {"sigma": 0.22, "offset_x": 0.04, "offset_y": 0.0},
    {"sigma": 0.28, "offset_x": 0.0, "offset_y": 0.04},
]


def generate_kernels_fft() -> list[npt.NDArray[np.complex64]]:
    """Generate pre-FFT'd kernels using vectorized numpy."""
    size = GRID_SIZE
    mid = size // 2

    i, j = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    dist = np.sqrt((i - mid) ** 2 + (j - mid) ** 2)

    all_kernels: list[npt.NDArray[np.complex64]] = []

    for k in range(NUM_KERNELS):
        d_scaled = dist / ((GLOBAL_R + 15.0) * RADII[k])
        sig = 0.5 * (np.tanh((-d_scaled + 1.0) * 5.0) + 1.0)

        ker_val = np.zeros_like(d_scaled)
        for p in range(3):
            diff = d_scaled - A[k, p]
            ker_val += B[k, p] * np.exp(-(diff * diff) / W[k, p])

        kernel_real = sig * ker_val

        total = kernel_real.sum()
        if total > 0.0:
            kernel_real /= total

        kernel_shifted = np.fft.ifftshift(kernel_real)
        kfft = np.fft.fft2(kernel_shifted).astype(np.complex64)
        all_kernels.append(kfft)

    return all_kernels


def generate_seed_channels() -> list[list[float]]:
    """Generate 3-channel Gaussian seed matching generate_glider_seed() in Rust."""
    size = GRID_SIZE
    coords = [(-1.0 + 2.0 * i / (size - 1)) for i in range(size)]
    channels: list[list[float]] = [[0.0] * (size * size) for _ in range(NUM_CHANNELS)]

    for iy in range(size):
        for ix in range(size):
            gx = coords[ix]
            gy = coords[iy]
            idx = iy * size + ix
            for c, ch in enumerate(SEED_CHANNELS):
                dx = gx - ch["offset_x"]
                dy = gy - ch["offset_y"]
                val = math.exp(-(dx * dx + dy * dy) / (2.0 * ch["sigma"] * ch["sigma"]))
                channels[c][idx] = max(0.0, min(1.0, val))

    return channels


def save_kernels_bin(kernels: list[npt.NDArray[np.complex64]], path: str) -> None:
    """Save FFT kernels as raw f32 interleaved real/imag pairs."""
    with open(path, "wb") as f:
        for kfft in kernels:
            flat = kfft.ravel()
            for val in flat:
                _ = f.write(struct.pack("ff", val.real.item(), val.imag.item()))
    size_mb = os.path.getsize(path) / 1024.0 / 1024.0
    print(f"Saved FFT kernels to {path} ({size_mb:.1f} MB)")


def main() -> None:
    print(f"Generating {NUM_KERNELS} kernels for {GRID_SIZE}x{GRID_SIZE} grid...")
    kernels: list[npt.NDArray[np.complex64]] = generate_kernels_fft()
    print(f"Done. Each kernel: {GRID_SIZE}x{GRID_SIZE} complex64")

    print(f"Generating {NUM_CHANNELS} seed channels...")
    seed_channels: list[list[float]] = generate_seed_channels()
    print(f"Done. Each channel: {GRID_SIZE}x{GRID_SIZE} f64")

    os.makedirs("kernels", exist_ok=True)

    # Save seed channels as JSON
    params: dict[str, object] = {
        "grid_size": GRID_SIZE,
        "num_channels": NUM_CHANNELS,
        "seed_channels": seed_channels,
    }
    os.makedirs("seed", exist_ok=True)
    with open("seed/glider.json", "w") as f:
        json.dump(params, f, indent=2)
    print("Saved seed channels to seed/glider.json")

    # Save FFT data as binary
    save_kernels_bin(kernels, "kernels/glider.bin")


if __name__ == "__main__":
    main()
