#!/usr/bin/env python3
"""
Generate kernel and seed data for a MaceLenia creature at a specific grid size.

MaceLenia uses FFT-based convolution with ring-based Gaussian kernels,
a bump growth function, and multi-channel dynamics.

Outputs:
  kernels/{name}_{grid_size}.bin  -- FFT kernels at grid_size x grid_size
  seed/{name}.json                -- seed + MaceLenia params (mu, sigma, weights, etc.)
"""

import argparse
import json
import math
import os
import struct
from typing import Any

import numpy as np
import numpy.typing as npt

NUM_CHANNELS: int = 3
NUM_KERNELS: int = NUM_CHANNELS * NUM_CHANNELS  # 9 for 3 channels

# ---------------------------------------------------------------------------
# Reference creature config (optimized for DiffusionLenia)
# ---------------------------------------------------------------------------

def get_reference_config() -> dict[str, Any]:
    """Return a DiffusionLenia-optimized creature configuration.

    Key design principles for DiffusionLenia:
    - Kernels span a wide range of radii to detect features at multiple scales
    - Growth mu/sigma create diverse affinity responses across channel pairs
    - Seed has multiple offset blobs to trigger directional pattern formation
    - Weights are non-uniform to create channel specialization
    """
    # Kernel generation params (Gaussian rings) -- wider range for diversity
    global_r = 10.0
    radii = [0.25 + 0.6 * (k / NUM_KERNELS) for k in range(NUM_KERNELS)]
    widths = [0.03 + 0.10 * (k / NUM_KERNELS) for k in range(NUM_KERNELS)]

    # Growth function params (per kernel, using permuted indexing to match
    # Python's state[:,:,None] channel ordering)
    # perm(k) = (k % C) * C + (k // C)
    perm = [((k % NUM_CHANNELS) * NUM_CHANNELS + (k // NUM_CHANNELS)) for k in range(NUM_KERNELS)]

    # Mu values: spread across a wider range for diverse affinity responses.
    # Lower mu = cells flow at lower densities (sparser patterns).
    # Higher mu = cells flow at higher densities (denser patterns).
    # Mix both for interesting dynamics.
    growth_mu = [0.05 + 0.20 * (perm[k] / NUM_KERNELS) for k in range(NUM_KERNELS)]

    # Sigma values: proportional to mu, with some variation.
    # Narrow sigma = sharp affinity transitions (crisp edges).
    # Wide sigma = smooth affinity transitions (diffuse boundaries).
    growth_sigma = [0.03 + 0.12 * (perm[k] / NUM_KERNELS) for k in range(NUM_KERNELS)]

    # Non-uniform weights: diagonal (same channel) pairs have higher weight,
    # cross-channel pairs have lower weight. This creates channel specialization.
    growth_weights = []
    for k in range(NUM_KERNELS):
        in_ch = k % NUM_CHANNELS
        out_ch = k // NUM_CHANNELS
        if in_ch == out_ch:
            growth_weights.append(0.5)  # same-channel: strong self-influence
        else:
            growth_weights.append(0.25)  # cross-channel: weaker influence

    # C0/C1 channel mapping: c0[k] = input channel, c1[k] = output channel
    c0 = [k % NUM_CHANNELS for k in range(NUM_KERNELS)]
    c1 = [k // NUM_CHANNELS for k in range(NUM_KERNELS)]

    # Seed: multiple offset blobs with different sizes per channel.
    # This creates asymmetry that drives directional pattern formation.
    seed_params = [
        {"radius": 0.35, "offset_x": -0.15, "offset_y": 0.0,   "amplitude": 0.5, "edge_width": 0.08},
        {"radius": 0.30, "offset_x": 0.1,   "offset_y": -0.1,  "amplitude": 0.45, "edge_width": 0.06},
        {"radius": 0.25, "offset_x": 0.05,  "offset_y": 0.15,  "amplitude": 0.4, "edge_width": 0.1},
    ]

    return {
        "num_kernels": NUM_KERNELS,
        "num_channels": NUM_CHANNELS,
        "global_r": global_r,
        "radii": radii,
        "widths": widths,
        "growth_mu": growth_mu,
        "growth_sigma": growth_sigma,
        "growth_weights": growth_weights,
        "c0": c0,
        "c1": c1,
        "seed_params": seed_params,
    }


# ---------------------------------------------------------------------------
# Kernel generation -- MaceLenia style (Gaussian rings, matches ml-rs)
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    return 0.5 * (math.tanh(x / 2.0) + 1.0)


def generate_kernels_fft(
    size: int, num_kernels: int, global_r: float,
    radii: list[float], widths: list[float],
) -> list[npt.NDArray[np.complex64]]:
    """Generate MaceLenia ring-based Gaussian kernels and return their FFT.

    Matches the Rust `generate_kernels_fft` in save_pngs.rs.
    Kernels are returned in natural order (k=0..num_kernels-1).
    """
    mid = size // 2
    kernels = []
    for k in range(num_kernels):
        # Build spatial kernel with a simple Gaussian ring
        spatial = np.zeros((size, size), dtype=np.float32)
        radius = radii[k]
        width = widths[k]

        for i in range(size):
            for j in range(size):
                di = i - mid
                dj = j - mid
                dist = math.sqrt(float(di * di + dj * dj))
                d_scaled = dist / (global_r * radius)
                sig = sigmoid(-(d_scaled - 1.0) * 10.0)
                diff = d_scaled - 0.5
                ker_val = math.exp(-(diff * diff) / (2.0 * width * width))
                spatial[i, j] = sig * ker_val

        # Normalize
        total = spatial.sum()
        if total > 0.0:
            spatial /= total

        # FFT shift: swap quadrants so center is at top-left
        shifted = np.fft.ifftshift(spatial)

        # 2D FFT
        kfft = np.fft.fft2(shifted).astype(np.complex64)
        kernels.append(kfft)

    return kernels


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------

def generate_seed_channels(
    size: int, num_channels: int,
    channel_configs: list[dict[str, float]],
) -> list[list[float]]:
    """Generate seed channels as flat-top blobs with smooth edges."""
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
                d = math.sqrt(dx*dx + dy*dy)
                radius = ch.get("radius", 0.5)
                edge_width = ch.get("edge_width", 0.1)
                # Smooth step: flat top inside radius, tanh transition at edge
                val = 0.5 * (1.0 - math.tanh((d - radius) / edge_width))
                amp = ch.get("amplitude", 1.0)
                channels[c][idx] = max(0.0, min(1.0, val * amp))
    return channels


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_kernels_fft_bin(kernels: list[npt.NDArray[np.complex64]], path: str) -> None:
    """Save FFT kernels to a binary file.

    Format: for each kernel, [re, im, re, im, ...] as f32 little-endian.
    """
    with open(path, "wb") as f:
        for kfft in kernels:
            flat = kfft.ravel()
            for val in flat:
                f.write(struct.pack("ff", val.real.item(), val.imag.item()))
    print(f"  Saved {path} ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")


def save_seed_json(
    seed_channels: list[list[float]],
    mcl_params: dict[str, Any],
    seed_size: int,
    path: str,
) -> None:
    """Save seed and MaceLenia parameters to a JSON file."""
    data = {
        "seed_size": seed_size,
        "num_channels": NUM_CHANNELS,
        "num_kernels": mcl_params["num_kernels"],
        "seed_channels": seed_channels,
        "c0": mcl_params["c0"],
        "c1": mcl_params["c1"],
        "growth_mu": mcl_params["growth_mu"],
        "growth_sigma": mcl_params["growth_sigma"],
        "growth_weights": mcl_params["growth_weights"],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {path} ({os.path.getsize(path) / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Creature generation
# ---------------------------------------------------------------------------

def generate_creature(name: str, config: dict[str, Any], grid_size: int) -> None:
    """Generate a complete MaceLenia creature at the given grid size."""
    print(f"\n=== {name} @ {grid_size}x{grid_size} ===")
    num_kernels = config["num_kernels"]

    # Generate FFT kernels (natural order, Rust will permute on upload)
    kernels_fft = generate_kernels_fft(
        grid_size, num_kernels, config["global_r"],
        config["radii"], config["widths"],
    )
    print(f"  FFT kernels: {num_kernels} x {grid_size}x{grid_size} complex64")

    # Generate seed
    seed_channels = generate_seed_channels(grid_size, NUM_CHANNELS, config["seed_params"])
    print(f"  Seed: {NUM_CHANNELS} channels x {grid_size}x{grid_size} f64")

    os.makedirs("kernels", exist_ok=True)
    os.makedirs("seed", exist_ok=True)

    # Save FFT kernels
    save_kernels_fft_bin(kernels_fft, f"kernels/{name}_{grid_size}.bin")

    # Save seed + MaceLenia params
    mcl_params = {
        "num_kernels": num_kernels,
        "num_channels": config["num_channels"],
        "c0": config["c0"],
        "c1": config["c1"],
        "growth_mu": config["growth_mu"],
        "growth_sigma": config["growth_sigma"],
        "growth_weights": config["growth_weights"],
        "global_r": config["global_r"],
        "radii": config["radii"],
        "widths": config["widths"],
    }
    save_seed_json(seed_channels, mcl_params, grid_size, f"seed/{name}.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a MaceLenia creature (kernels + seed)"
    )
    parser.add_argument("--grid-size", type=int, default=512,
                        choices=[64, 128, 256, 512, 1024],
                        help="Grid size (default: 512)")
    parser.add_argument("--name", type=str, default="mcl_creature",
                        help="Creature name (default: mcl_creature)")
    args = parser.parse_args()

    config = get_reference_config()
    generate_creature(args.name, config, args.grid_size)


if __name__ == "__main__":
    main()
