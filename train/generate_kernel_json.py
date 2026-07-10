#!/usr/bin/env python3
"""
Generate kernel and seed data for Flow Lenia creatures.

Each creature has:
  kernels/<name>.bin   — its FFT convolution kernels (its "genome")
  seed/<name>.json     — its initial state (its "body")

The Rust simulation loads these files instead of recomputing at startup.
"""

import json
import math
import os
import struct
from typing import Any

import numpy as np
import numpy.typing as npt

GRID_SIZE: int = 512
NUM_CHANNELS: int = 3


# ---------------------------------------------------------------------------
# Kernel generation
# ---------------------------------------------------------------------------

def generate_kernels_fft(
    size: int,
    num_kernels: int,
    global_r: float,
    radii: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> list[npt.NDArray[np.complex64]]:
    """Generate pre-FFT'd kernels for a creature."""
    mid = size // 2
    i, j = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    dist = np.sqrt((i - mid) ** 2 + (j - mid) ** 2)

    kernels: list[npt.NDArray[np.complex64]] = []

    for k in range(num_kernels):
        d_scaled = dist / ((global_r + 15.0) * radii[k])
        sig = 0.5 * (np.tanh((-d_scaled + 1.0) * 5.0) + 1.0)

        ker_val = np.zeros_like(d_scaled)
        for p in range(3):
            diff = d_scaled - a[k, p]
            ker_val += b[k, p] * np.exp(-(diff * diff) / w[k, p])

        kernel_real = sig * ker_val

        total = kernel_real.sum()
        if total > 0.0:
            kernel_real /= total

        kernel_shifted = np.fft.ifftshift(kernel_real)
        kfft = np.fft.fft2(kernel_shifted).astype(np.complex64)
        kernels.append(kfft)

    return kernels


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------

def generate_seed_channels(
    size: int,
    num_channels: int,
    channel_configs: list[dict[str, float]],
) -> list[list[float]]:
    """Generate multi-channel Gaussian seed."""
    coords = [(-1.0 + 2.0 * i / (size - 1)) for i in range(size)]
    channels: list[list[float]] = [[0.0] * (size * size) for _ in range(num_channels)]

    for iy in range(size):
        for ix in range(size):
            gx = coords[ix]
            gy = coords[iy]
            idx = iy * size + ix
            for c, ch in enumerate(channel_configs):
                dx = gx - ch["offset_x"]
                dy = gy - ch["offset_y"]
                val = math.exp(-(dx * dx + dy * dy) / (2.0 * ch["sigma"] * ch["sigma"]))
                channels[c][idx] = max(0.0, min(1.0, val))

    return channels


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_kernels_bin(kernels: list[npt.NDArray[np.complex64]], path: str) -> None:
    """Save FFT kernels as raw f32 interleaved real/imag pairs."""
    with open(path, "wb") as f:
        for kfft in kernels:
            flat = kfft.ravel()
            for val in flat:
                _ = f.write(struct.pack("ff", val.real.item(), val.imag.item()))
    size_mb = os.path.getsize(path) / 1024.0 / 1024.0
    print(f"  Saved {path} ({size_mb:.1f} MB)")


def save_seed_json(seed_channels: list[list[float]], path: str) -> None:
    """Save seed channels as JSON."""
    data: dict[str, object] = {
        "grid_size": GRID_SIZE,
        "num_channels": NUM_CHANNELS,
        "seed_channels": seed_channels,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    size_kb = os.path.getsize(path) / 1024.0
    print(f"  Saved {path} ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# Creature definitions
# ---------------------------------------------------------------------------

def generate_creature(name: str, config: dict[str, Any]) -> None:
    """Generate kernel and seed files for a single creature."""
    print(f"\n=== {name} ===")

    size = config.get("grid_size", GRID_SIZE)
    num_kernels = config["num_kernels"]
    num_channels = config.get("num_channels", NUM_CHANNELS)

    # Generate kernels
    kernels = generate_kernels_fft(
        size,
        num_kernels,
        config["global_r"],
        np.array(config["radii"]),
        np.array(config["a"]),
        np.array(config["w"]),
        np.array(config["b"]),
    )
    print(f"  Kernels: {num_kernels} × {size}×{size} complex64")

    # Generate seed
    seed_channels = generate_seed_channels(size, num_channels, config["seed_params"])
    print(f"  Seed: {num_channels} channels × {size}×{size} f64")

    # Save
    os.makedirs("kernels", exist_ok=True)
    os.makedirs("seed", exist_ok=True)
    save_kernels_bin(kernels, f"kernels/{name}.bin")
    save_seed_json(seed_channels, f"seed/{name}.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    glider = {
        "num_kernels": 9,
        "global_r": 42.0,
        "radii": [0.8, 0.6, 1.0, 0.7, 0.5, 0.9, 0.65, 0.55, 0.85],
        "a": [
            [0.0, 0.6, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.7, 0.0],
            [0.0, 0.55, 0.0],
            [0.0, 0.45, 0.0],
            [0.0, 0.65, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.6, 0.0],
            [0.0, 0.55, 0.0],
        ],
        "w": [
            [0.08, 0.06, 0.01],
            [0.07, 0.05, 0.01],
            [0.09, 0.07, 0.01],
            [0.08, 0.06, 0.01],
            [0.07, 0.05, 0.01],
            [0.09, 0.07, 0.01],
            [0.08, 0.06, 0.01],
            [0.07, 0.05, 0.01],
            [0.08, 0.06, 0.01],
        ],
        "b": [
            [0.8, -0.3, 0.0],
            [0.7, -0.25, 0.0],
            [0.9, -0.35, 0.0],
            [0.75, -0.3, 0.0],
            [0.65, -0.2, 0.0],
            [0.85, -0.35, 0.0],
            [0.7, -0.25, 0.0],
            [0.6, -0.2, 0.0],
            [0.8, -0.3, 0.0],
        ],
        "seed_params": [
            {"sigma": 0.25, "offset_x": 0.0, "offset_y": 0.0},
            {"sigma": 0.22, "offset_x": 0.04, "offset_y": 0.0},
            {"sigma": 0.28, "offset_x": 0.0, "offset_y": 0.04},
        ],
    }

    generate_creature("glider", glider)


if __name__ == "__main__":
    main()
