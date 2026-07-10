#!/usr/bin/env python3
"""
Generate kernel configuration for Flow Lenia.

Creates:
  kernels/glider.json       — generation parameters (global_r, radii, a, w, b)
  kernels/kernels_fft.bin   — pre-computed FFT kernel weights (binary f32)

The Rust simulation loads both files instead of recomputing at startup.
"""

import json
import os
import struct

import numpy as np
import numpy.typing as npt

GRID_SIZE: int = 512
NUM_KERNELS: int = 9

GLOBAL_R: float = 42.0
RADII: npt.NDArray[np.float64] = np.array([0.8, 0.6, 1.0, 0.7, 0.5, 0.9, 0.65, 0.55, 0.85])

A: npt.NDArray[np.float64] = np.array([
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

W: npt.NDArray[np.float64] = np.array([
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

B: npt.NDArray[np.float64] = np.array([
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

    os.makedirs("kernels", exist_ok=True)

    # Save parameters as JSON
    params: dict[str, object] = {
        "grid_size": GRID_SIZE,
        "num_kernels": NUM_KERNELS,
        "global_r": GLOBAL_R,
        "radii": RADII.tolist(),
        "a": A.tolist(),
        "w": W.tolist(),
        "b": B.tolist(),
    }
    with open("kernels/glider.json", "w") as f:
        json.dump(params, f, indent=2)
    print("Saved parameters to kernels/glider.json")

    # Save FFT data as binary
    save_kernels_bin(kernels, "kernels/kernels_fft.bin")


if __name__ == "__main__":
    main()
