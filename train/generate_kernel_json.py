#!/usr/bin/env python3
"""
Generate kernel and seed data for a single 64x64 Flow Lenia creature.

The creature is stored grid-size independently:
  kernels/<name>_512.bin  — FFT kernels at 512x512 (for Rust simulator)
  kernels/<name>_256.bin  — spatial kernels at 128x128 (for Python training)
  seed/<name>.json        — 64x64 seed + bump params + growth params

Both the Python training (256x256) and Rust simulation (512x512) load the
same seed and pad it to their respective grid sizes.
"""

import json
import math
import os
import struct
from typing import Any

import numpy as np
import numpy.typing as npt

GRID_512: int = 512          # simulation grid
GRID_256: int = 256          # training grid (spatial kernels stored at this size)
SEED_SIZE: int = 64
NUM_CHANNELS: int = 3


def generate_kernels_fft(
    size: int, num_kernels: int, global_r: float,
    radii: npt.NDArray[np.float64], a: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64], b: npt.NDArray[np.float64],
) -> list[npt.NDArray[np.complex64]]:
    mid = size // 2
    i, j = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    dist = np.sqrt((i - mid) ** 2 + (j - mid) ** 2)
    kernels = []
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


def generate_kernels_spatial(
    size: int, num_kernels: int, global_r: float,
    radii: npt.NDArray[np.float64], a: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64], b: npt.NDArray[np.float64],
) -> list[npt.NDArray[np.float32]]:
    mid = size // 2
    i, j = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    dist = np.sqrt((i - mid) ** 2 + (j - mid) ** 2)
    kernels = []
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
        kernels.append(kernel_real.astype(np.float32))
    return kernels


def generate_seed_channels(
    size: int, num_channels: int,
    channel_configs: list[dict[str, float]],
) -> list[list[float]]:
    coords = [(-1.0 + 2.0 * i / (size - 1)) for i in range(size)]
    channels = [[0.0] * (size * size) for _ in range(num_channels)]
    for iy in range(size):
        for ix in range(size):
            gx = coords[ix]
            gy = coords[iy]
            idx = iy * size + ix
            for c, ch in enumerate(channel_configs):
                dx = gx - ch.get("offset_x", 0.0)
                dy = gy - ch.get("offset_y", 0.0)
                val = math.exp(-(dx*dx + dy*dy) / (2.0 * ch["sigma"] * ch["sigma"]))
                amp = ch.get("amplitude", 1.0)
                channels[c][idx] = max(0.0, min(1.0, val * amp))
    return channels


def save_kernels_fft_bin(kernels: list[npt.NDArray[np.complex64]], path: str) -> None:
    with open(path, "wb") as f:
        for kfft in kernels:
            flat = kfft.ravel()
            for val in flat:
                _ = f.write(struct.pack("ff", val.real.item(), val.imag.item()))
    print(f"  Saved {path} ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")


def save_kernels_spatial_bin(kernels: list[npt.NDArray[np.float32]], path: str) -> None:
    with open(path, "wb") as f:
        for kspatial in kernels:
            flat = kspatial.ravel()
            for val in flat:
                _ = f.write(struct.pack("f", val.item()))
    print(f"  Saved {path} ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")


def save_seed_json(seed_channels, bump_params, growth_params, path):
    data = {
        "seed_size": SEED_SIZE,
        "num_channels": NUM_CHANNELS,
        "seed_channels": seed_channels,
        "bump_params": bump_params,
        "growth_params": growth_params,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {path} ({os.path.getsize(path) / 1024:.1f} KB)")


def generate_creature(name: str, config: dict[str, Any]) -> None:
    print(f"\n=== {name} ===")
    num_kernels = config["num_kernels"]

    kernels_fft = generate_kernels_fft(
        GRID_512, num_kernels, config["global_r"],
        np.array(config["radii"]), np.array(config["a"]),
        np.array(config["w"]), np.array(config["b"]),
    )
    print(f"  FFT kernels: {num_kernels} x {GRID_512}x{GRID_512} complex64")

    kernels_spatial = generate_kernels_spatial(
        GRID_256, num_kernels, config["global_r"],
        np.array(config["radii"]), np.array(config["a"]),
        np.array(config["w"]), np.array(config["b"]),
    )
    print(f"  Spatial kernels: {num_kernels} x {GRID_256}x{GRID_256} float32")

    seed_channels = generate_seed_channels(SEED_SIZE, NUM_CHANNELS, config["seed_params"])
    print(f"  Seed: {NUM_CHANNELS} channels x {SEED_SIZE}x{SEED_SIZE} f64")

    os.makedirs("kernels", exist_ok=True)
    os.makedirs("seed", exist_ok=True)
    save_kernels_fft_bin(kernels_fft, f"kernels/{name}_512.bin")
    save_kernels_spatial_bin(kernels_spatial, f"kernels/{name}_256.bin")
    bump_params = {
        "num_kernels": num_kernels,
        "global_r": config["global_r"],
        "radii": config["radii"],
        "a": config["a"], "w": config["w"], "b": config["b"],
    }
    growth_params = {
        "m": config["growth_m"], "s": config["growth_s"], "h": config["growth_h"],
    }
    save_seed_json(seed_channels, bump_params, growth_params, f"seed/{name}.json")


def main() -> None:
    # A 64x64 creature with rich, asymmetric patterns
    # Uses the original glider's bump parameters (global_r=42) for complex dynamics
    glider = {
        "num_kernels": 9,
        "global_r": 42.0,
        "radii": [0.8, 0.6, 1.0, 0.7, 0.5, 0.9, 0.65, 0.55, 0.85],
        "growth_m": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "growth_s": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        "growth_h": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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
            {"sigma": 0.50, "offset_x": 0.0, "offset_y": 0.0},
            {"sigma": 0.44, "offset_x": 0.08, "offset_y": 0.0},
            {"sigma": 0.56, "offset_x": 0.0, "offset_y": 0.08},
        ],
    }

    generate_creature("glider", glider)

    # A simple 3-kernel creature matching the test parameters from save_frames_png.py
    test = {
        "num_kernels": 3,
        "global_r": 10.0,
        "radii": [0.5, 0.8, 0.65],
        "growth_m": [0.1, 0.15, 0.12],
        "growth_s": [0.05, 0.08, 0.065],
        "growth_h": [0.5, 0.8, 0.65],
        "a": [
            [0.0, 0.5, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.45, 0.0],
        ],
        "w": [
            [0.1, 0.05, 0.01],
            [0.08, 0.06, 0.01],
            [0.09, 0.055, 0.01],
        ],
        "b": [
            [0.5, 0.3, 0.0],
            [0.7, 0.2, 0.0],
            [0.6, 0.25, 0.0],
        ],
        "seed_params": [
            {"sigma": 0.25, "offset_x": 0.0, "offset_y": 0.0, "amplitude": 0.5},
            {"sigma": 0.25, "offset_x": 0.0, "offset_y": 0.0, "amplitude": 0.6667},
            {"sigma": 0.25, "offset_x": 0.0, "offset_y": 0.0, "amplitude": 0.8333},
        ],
    }

    generate_creature("test", test)


if __name__ == "__main__":
    main()
