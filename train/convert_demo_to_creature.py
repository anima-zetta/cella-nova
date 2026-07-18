#!/usr/bin/env python3
"""
Convert demo_params/*.pt files to MaceLenia creatures (seed JSON + kernel binaries).

For each .pt file in demo_params/, this script:
  1. Loads the PyTorch parameter dict (mu, sigma, beta, mu_k, sigma_k, weights, k_size)
  2. Builds spatial kernels from the ring-based Gaussian parameters
  3. Pads kernels to the target grid size and computes their FFT
  4. Saves FFT kernels as kernels/{name}_{grid_size}.bin
  5. Extracts growth params (mu, sigma, weights) and c0/c1 mapping
  6. Generates a default seed (Gaussian blobs)
  7. Saves the creature config as seed/{name}.json

Usage:
  python3 train/convert_demo_to_creature.py
  python3 train/convert_demo_to_creature.py --grid-size 256
  python3 train/convert_demo_to_creature.py --name "abbreviated_nonworker"
  python3 train/convert_demo_to_creature.py --all
"""

import argparse
import json
import math
import os
import struct
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Kernel generation — matches MCLenia.kernel_slice + compute_kernel
# ---------------------------------------------------------------------------

def build_spatial_kernel(
    k_size: int,
    beta: np.ndarray,   # (C_out, C_in, num_rings)
    mu_k: np.ndarray,   # (C_out, C_in, num_rings)
    sigma_k: np.ndarray, # (C_out, C_in, num_rings)
) -> np.ndarray:
    """Build spatial kernel from ring-based Gaussian parameters.

    Matches MCLenia.kernel_slice() + compute_kernel() in lenia_org.py.

    Returns:
        kernel: (C_out, C_in, k_size, k_size) float32, normalized so sum = 1
    """
    C_out, C_in, num_rings = beta.shape

    # Coordinate grid in [-1, 1]
    xyrange = np.linspace(-1.0, 1.0, k_size)
    X, Y = np.meshgrid(xyrange, xyrange, indexing="xy")
    r = np.sqrt(X ** 2 + Y ** 2)  # (k_size, k_size)

    # Expand r to (1, 1, 1, k_size, k_size) then to (C_out, C_in, num_rings, k_size, k_size)
    r_exp = r[np.newaxis, np.newaxis, np.newaxis, :, :]
    r_exp = np.broadcast_to(r_exp, (C_out, C_in, num_rings, k_size, k_size))

    # Gaussian rings
    mu_k_exp = mu_k[:, :, :, np.newaxis, np.newaxis]   # (C_out, C_in, num_rings, 1, 1)
    sigma_k_exp = sigma_k[:, :, :, np.newaxis, np.newaxis]  # (C_out, C_in, num_rings, 1, 1)
    beta_exp = beta[:, :, :, np.newaxis, np.newaxis]   # (C_out, C_in, num_rings, 1, 1)

    K = np.exp(-((r_exp - mu_k_exp) / sigma_k_exp) ** 2 / 2)  # (C_out, C_in, num_rings, k_size, k_size)
    K = np.sum(beta_exp * K, axis=2)  # (C_out, C_in, k_size, k_size)

    # Normalize so integral = 1
    summed = K.sum(axis=(-1, -2), keepdims=True)
    summed = np.where(summed < 1e-6, 1.0, summed)
    K = K / summed

    return K.astype(np.float32)


def kernel_to_fft(kernel: np.ndarray, grid_size: int) -> np.ndarray:
    """Pad a spatial kernel to grid_size and compute its FFT.

    Matches MCLenia.kernel_to_fft() in lenia_org.py.

    Args:
        kernel: (C_out, C_in, k_size, k_size) float32 spatial kernel
        grid_size: target grid size (power of two)

    Returns:
        fft_kernel: (C_out, C_in, grid_size, grid_size) complex64
    """
    C_out, C_in, k_size, _ = kernel.shape
    pad_w = grid_size - k_size
    pad_h = grid_size - k_size

    # Pad: left/right, top/bottom
    K = np.pad(kernel, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="constant")

    # Center the kernel on the top-left corner for FFT (fftshift equivalent)
    half = k_size // 2
    K = np.roll(K, (-half, -half), axis=(-2, -1))

    # 2D FFT
    K_fft = np.fft.fft2(K, axes=(-2, -1)).astype(np.complex64)

    return K_fft


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------

def generate_seed_channels(
    grid_size: int, num_channels: int
) -> list[list[float]]:
    """Generate seed channels as smooth flat-top blobs with offsets.

    Matches the style from generate_kernel_json.py.
    """
    coords = [-1.0 + 2.0 * i / (grid_size - 1) for i in range(grid_size)]
    channels = [[0.0] * (grid_size * grid_size) for _ in range(num_channels)]

    seed_params = [
        {"radius": 0.35, "offset_x": -0.15, "offset_y": 0.0, "amplitude": 0.5, "edge_width": 0.08},
        {"radius": 0.30, "offset_x": 0.1, "offset_y": -0.1, "amplitude": 0.45, "edge_width": 0.06},
        {"radius": 0.25, "offset_x": 0.05, "offset_y": 0.15, "amplitude": 0.4, "edge_width": 0.1},
    ]

    for iy in range(grid_size):
        for ix in range(grid_size):
            gx = coords[ix]
            gy = coords[iy]
            idx = iy * grid_size + ix
            for c, ch in enumerate(seed_params):
                dx = gx - ch.get("offset_x", 0.0)
                dy = gy - ch.get("offset_y", 0.0)
                d = math.sqrt(dx * dx + dy * dy)
                radius = ch.get("radius", 0.5)
                edge_width = ch.get("edge_width", 0.1)
                val = 0.5 * (1.0 - math.tanh((d - radius) / edge_width))
                amp = ch.get("amplitude", 1.0)
                channels[c][idx] = max(0.0, min(1.0, val * amp))

    return channels


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_kernels_fft_bin(
    kernels_fft: np.ndarray, path: str
) -> None:
    """Save FFT kernels to a binary file.

    Format: for each kernel, [re, im, re, im, ...] as f32 little-endian.
    kernels_fft shape: (K, H, W) complex64
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        for k in range(kernels_fft.shape[0]):
            flat = kernels_fft[k].ravel()
            for val in flat:
                f.write(struct.pack("ff", val.real.item(), val.imag.item()))
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Saved {path} ({size_mb:.1f} MB)")


def save_seed_json(
    seed_channels: list[list[float]],
    params: dict,
    grid_size: int,
    path: str,
) -> None:
    """Save seed and MaceLenia parameters to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "seed_size": grid_size,
        "num_channels": params["num_channels"],
        "num_kernels": params["num_kernels"],
        "seed_channels": seed_channels,
        "c0": params["c0"],
        "c1": params["c1"],
        "growth_mu": params["growth_mu"],
        "growth_sigma": params["growth_sigma"],
        "growth_weights": params["growth_weights"],
        "global_r": params.get("global_r", 10.0),
        "radii": params.get("radii", []),
        "widths": params.get("widths", []),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved {path} ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_pt_to_creature(
    pt_path: str,
    grid_size: int,
    output_dir: str = ".",
) -> str:
    """Convert a single .pt parameter file to a MaceLenia creature.

    Returns the creature name (stem of the .pt file).
    """
    name = os.path.splitext(os.path.basename(pt_path))[0]
    print(f"\n=== {name} @ {grid_size}x{grid_size} ===")

    # Load params
    device = "cpu"
    param_dict = torch.load(pt_path, map_location=device)

    # Validate expected keys
    expected_keys = {"k_size", "mu", "sigma", "beta", "mu_k", "sigma_k", "weights"}
    actual_keys = set(param_dict.keys())
    if not expected_keys.issubset(actual_keys):
        missing = expected_keys - actual_keys
        print(f"  ⚠️  Skipping {name}: missing keys {missing}")
        return None

    k_size = param_dict["k_size"]
    if isinstance(k_size, torch.Tensor):
        k_size = k_size.item()
    k_size = int(k_size)
    if k_size % 2 == 0:
        k_size += 1  # ensure odd

    # Extract tensors, squeeze batch dim
    mu = param_dict["mu"].detach().cpu().numpy()       # (B, C_out, C_in)
    sigma = param_dict["sigma"].detach().cpu().numpy()  # (B, C_out, C_in)
    beta = param_dict["beta"].detach().cpu().numpy()    # (B, C_out, C_in, rings)
    mu_k = param_dict["mu_k"].detach().cpu().numpy()    # (B, C_out, C_in, rings)
    sigma_k = param_dict["sigma_k"].detach().cpu().numpy()  # (B, C_out, C_in, rings)
    weights = param_dict["weights"].detach().cpu().numpy()  # (B, C_out, C_in)

    # Squeeze batch dimension (assume B=1)
    mu = mu[0]          # (C_out, C_in)
    sigma = sigma[0]    # (C_out, C_in)
    beta = beta[0]      # (C_out, C_in, rings)
    mu_k = mu_k[0]      # (C_out, C_in, rings)
    sigma_k = sigma_k[0]  # (C_out, C_in, rings)
    weights = weights[0]  # (C_out, C_in)

    C_out, C_in = mu.shape
    num_channels = C_out
    num_kernels = C_out * C_in

    print(f"  {num_channels} channels, {num_kernels} kernels, k_size={k_size}")

    # Build spatial kernels from ring params
    spatial_kernel = build_spatial_kernel(k_size, beta, mu_k, sigma_k)
    print(f"  Spatial kernel: {spatial_kernel.shape}")

    # Pad and FFT
    fft_kernel = kernel_to_fft(spatial_kernel, grid_size)
    print(f"  FFT kernel: {fft_kernel.shape}")

    # Flatten to (K, H, W) with c0/c1 mapping
    c0 = []
    c1 = []
    kernels_flat = []
    for out_ch in range(C_out):
        for in_ch in range(C_in):
            c0.append(in_ch)
            c1.append(out_ch)
            kernels_flat.append(fft_kernel[out_ch, in_ch])
    kernels_flat = np.array(kernels_flat)  # (K, H, W) complex64

    # Save FFT kernels
    save_kernels_fft_bin(kernels_flat, f"{output_dir}/kernels/{name}_{grid_size}.bin")

    # Extract growth params in c0/c1 order
    growth_mu = []
    growth_sigma = []
    growth_weights = []
    for out_ch in range(C_out):
        for in_ch in range(C_in):
            growth_mu.append(float(mu[out_ch, in_ch]))
            growth_sigma.append(float(sigma[out_ch, in_ch]))
            growth_weights.append(float(weights[out_ch, in_ch]))

    # Dummy radii/widths (metadata only — not used by Rust simulation)
    global_r = 10.0
    radii = [0.25 + 0.6 * (k / num_kernels) for k in range(num_kernels)]
    widths = [0.03 + 0.10 * (k / num_kernels) for k in range(num_kernels)]

    # Generate seed
    seed_channels = generate_seed_channels(grid_size, num_channels)

    # Save JSON config
    params_out = {
        "num_channels": num_channels,
        "num_kernels": num_kernels,
        "c0": c0,
        "c1": c1,
        "growth_mu": growth_mu,
        "growth_sigma": growth_sigma,
        "growth_weights": growth_weights,
        "global_r": global_r,
        "radii": radii,
        "widths": widths,
    }
    save_seed_json(seed_channels, params_out, grid_size, f"{output_dir}/seed/{name}.json")

    print(f"  ✅ Created creature '{name}'")
    return name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert demo_params/*.pt to MaceLenia creatures"
    )
    parser.add_argument("--grid-size", type=int, default=512,
                        choices=[64, 128, 256, 512, 1024],
                        help="Grid size (default: 512)")
    parser.add_argument("--name", type=str, default=None,
                        help="Convert a single .pt file by name (without .pt extension)")
    parser.add_argument("--all", action="store_true",
                        help="Convert ALL .pt files in demo_params/")
    parser.add_argument("--demo-dir", type=str, default="demo_params",
                        help="Directory containing .pt files (default: demo_params)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory (default: current dir, creates seed/ and kernels/ subdirs)")
    args = parser.parse_args()

    if args.name:
        # Convert a single file
        pt_path = os.path.join(args.demo_dir, f"{args.name}.pt")
        if not os.path.exists(pt_path):
            print(f"Error: {pt_path} not found")
            sys.exit(1)
        convert_pt_to_creature(pt_path, args.grid_size, args.output_dir)

    elif args.all:
        # Convert all .pt files
        if not os.path.isdir(args.demo_dir):
            print(f"Error: {args.demo_dir}/ directory not found")
            sys.exit(1)
        pt_files = sorted(f for f in os.listdir(args.demo_dir) if f.endswith(".pt"))
        if not pt_files:
            print(f"No .pt files found in {args.demo_dir}/")
            sys.exit(1)
        print(f"Found {len(pt_files)} .pt files in {args.demo_dir}/")
        success = 0
        for fname in pt_files:
            pt_path = os.path.join(args.demo_dir, fname)
            result = convert_pt_to_creature(pt_path, args.grid_size, args.output_dir)
            if result is not None:
                success += 1
        print(f"\n{'='*50}")
        print(f"Converted {success}/{len(pt_files)} creatures successfully")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
