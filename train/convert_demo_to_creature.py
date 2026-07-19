#!/usr/bin/env python3
"""
Convert demo_params/*.pt files to MaceLenia creatures (seed JSON + kernel binaries).

For each .pt file in demo_params/, this script:
  1. Loads the PyTorch parameter dict (mu, sigma, beta, mu_k, sigma_k, weights, k_size)
  2. Builds spatial kernels from the ring-based Gaussian parameters
  3. Pads kernels to the target grid size and computes their FFT
  4. Saves FFT kernels as kernels/{name}_{grid_size}.bin
  5. Extracts growth params (mu, sigma, weights) and c0/c1 mapping
  6. Generates a fractal Perlin noise seed
  7. Saves the creature config as seed/{name}.json

Usage:
  python3 train/convert_demo_to_creature.py
  python3 train/convert_demo_to_creature.py --grid-size 256
  python3 train/convert_demo_to_creature.py --name "abbreviated_nonworker"
  python3 train/convert_demo_to_creature.py --all
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
from pyperlin import FractalPerlin2D

# ---------------------------------------------------------------------------
# Perlin noise seed constants
# ---------------------------------------------------------------------------

# Dominant feature size in pixels (default: grid_size // 6, set at call site).
PERLIN_WAVELENGTH: int | None = None
# Amplitude falloff per octave.
PERLIN_PERSISTENCE: float = 0.5
# Fraction of seed pixels clamped to 0.
PERLIN_BLACK_PROP: float = 0.3
# RNG seed for reproducible noise (None = random).
PERLIN_SEED: int | None = None

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
# Seed generation — Perlin noise  (via pyperlin library)
# ---------------------------------------------------------------------------

def _fractal_noise_2d(
    shape: tuple[int, int],
    max_wavelength: int,
    persistence: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate fractal Perlin noise using the pyperlin library.

    Matches the approach in train/utils/noise_gen.py: each octave is
    generated separately via FractalPerlin2D and summed.

    Args:
        shape: (H, W) of the output grid.
        max_wavelength: largest feature size in pixels.
        persistence: amplitude scaling per octave (default 0.5).
        rng: numpy random generator (used for torch seed).

    Returns:
        (H, W) float64 numpy array in approximately [-1, 1].
    """
    H, W = shape
    max_octaves = min(6, int(np.log2(max_wavelength)))
    if max_octaves < 1:
        max_octaves = 1

    # Seed the torch generator from the numpy rng state.
    if rng is not None:
        torch_seed = rng.integers(0, 2 ** 31 - 1)
    else:
        torch_seed = None

    norm = sum(persistence ** (i + 1) for i in range(max_octaves))
    noise = np.zeros((H, W), dtype=np.float64)

    for i in range(max_octaves):
        wl = int(max_wavelength * 0.5 ** i)
        if wl < 1:
            wl = 1

        # Pad grid to a multiple of the wavelength (required by pyperlin).
        pad_h = int(np.ceil(H / wl) * wl)
        pad_w = int(np.ceil(W / wl) * wl)
        freq_y = pad_h // wl
        freq_x = pad_w // wl

        gen = torch.Generator(device="cpu")
        if torch_seed is not None:
            gen.manual_seed(int(torch_seed + i))

        fp = FractalPerlin2D(
            (1, pad_h, pad_w),
            [[freq_y, freq_x]],
            [1.0 / 0.7053],
            generator=gen,
        )()
        octave = fp[0, :H, :W].cpu().numpy()
        noise += persistence ** (i + 1) * octave

    return noise / norm


def _apply_black_prop(noise: np.ndarray, black_prop: float) -> np.ndarray:
    """Scale and clamp noise to [0, 1] with a given black proportion.

    Values below the black_prop threshold are clamped to 0, creating
    scattered "empty" regions that help seed formation.
    """
    scaled = (noise + (0.5 - black_prop) * 2.0) / (2.0 * (1.0 - black_prop))
    return np.clip(scaled, 0.0, 1.0)


def generate_seed_perlin(
    grid_size: int,
    num_channels: int,
    *,
    wavelength: int | None = None,
    persistence: float = 0.5,
    black_prop: float = 0.3,
    seed: int | None = None,
) -> list[list[float]]:
    """Generate seed channels as circular blobs filled with fractal Perlin noise.

    Each channel gets a circular region (with per-channel position, radius,
    and edge softness) filled with independent 2D Perlin noise.  This
    combines the localised structure of the original Gaussian-blob seed
    with the rich texture of fractal noise.

    Args:
        grid_size: grid width/height in pixels.
        num_channels: number of channels.
        wavelength: dominant feature size (default: grid_size // 10).
        persistence: amplitude falloff per octave (default 0.5).
        black_prop: fraction of pixels clamped to 0 (default 0.3).
        seed: rng seed for reproducible noise.

    Returns:
        List of ``num_channels`` flat arrays, each ``grid_size * grid_size``
        floats in [0, 1].
    """
    if wavelength is None:
        wavelength = max(4, grid_size // 6)
    rng = np.random.default_rng(seed)

    # Per-channel blob parameters (same layout as the original Gaussian seed).
    blob_params = [
        {"radius": 0.65, "offset_x": -0.05, "offset_y": 0.0,   "amplitude": 0.5, "edge_width": 0.08},
        {"radius": 0.60, "offset_x": 0.03,  "offset_y": -0.03, "amplitude": 0.45, "edge_width": 0.06},
        {"radius": 0.55, "offset_x": 0.02,  "offset_y": 0.04,  "amplitude": 0.4,  "edge_width": 0.1},
    ]

    # Normalised pixel coordinates in [-1, 1].
    coords = np.linspace(-1.0, 1.0, grid_size)
    X, Y = np.meshgrid(coords, coords, indexing="xy")

    shape = (grid_size, grid_size)
    channels: list[list[float]] = []

    for c in range(num_channels):
        bp = blob_params[c % len(blob_params)]
        cx = bp["offset_x"]
        cy = bp["offset_y"]
        radius = bp["radius"]
        edge = bp["edge_width"]
        amp = bp["amplitude"]

        # --- Circular mask (smooth step, same as original Gaussian seed) ---
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = 0.5 * (1.0 - np.tanh((dist - radius) / edge))
        mask = np.clip(mask, 0.0, 1.0)

        # --- 2D fractal Perlin noise ---
        raw = _fractal_noise_2d(shape, wavelength, persistence, rng)

        # Normalise noise to [-1, 1] so _apply_black_prop thresholds correctly.
        lo, hi = raw.min(), raw.max()
        if hi > lo:
            norm = 2.0 * (raw - lo) / (hi - lo) - 1.0
        else:
            norm = raw - lo

        noise_ch = _apply_black_prop(norm, black_prop)

        # --- Combine: noise inside the circle, zero outside ---
        ch = noise_ch * mask * amp
        channels.append(ch.ravel().tolist())

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
    skip_existing: bool = False,
) -> str:
    """Convert a single .pt parameter file to a MaceLenia creature.

    Returns the creature name (stem of the .pt file), or None if skipped.
    """
    name = os.path.splitext(os.path.basename(pt_path))[0]

    # Check if both output files already exist.
    if skip_existing:
        json_path = os.path.join(output_dir, "seed", f"{name}.json")
        bin_path = os.path.join(output_dir, "kernels", f"{name}_{grid_size}.bin")
        if os.path.exists(json_path) and os.path.exists(bin_path):
            print(f"  Skipping {name} @ {grid_size}x{grid_size} (already exists)")
            return None

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

    # Generate seed (fractal Perlin noise)
    seed_channels = generate_seed_perlin(grid_size, num_channels)
    print(f"  Seed: Perlin noise ({num_channels} channels)")

    # Save JSON config
    params_out = {
        "num_channels": num_channels,
        "num_kernels": num_kernels,
        "c0": c0,
        "c1": c1,
        "growth_mu": growth_mu,
        "growth_sigma": growth_sigma,
        "growth_weights": growth_weights,
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
                        choices=[64, 128, 256, 512, 1024, 2048],
                        help="Grid size (default: 512)")
    parser.add_argument("--name", type=str, default=None,
                        help="Convert a single .pt file by name (without .pt extension)")
    parser.add_argument("--all", action="store_true",
                        help="Convert ALL .pt files in demo_params/")
    parser.add_argument("--demo-dir", type=str, default="demo_params",
                        help="Directory containing .pt files (default: demo_params)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory (default: current dir, creates seed/ and kernels/ subdirs)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip creatures that already have seed JSON and kernel bin files")
    args = parser.parse_args()

    if args.name:
        # Convert a single file
        pt_path = os.path.join(args.demo_dir, f"{args.name}.pt")
        if not os.path.exists(pt_path):
            print(f"Error: {pt_path} not found")
            sys.exit(1)
        convert_pt_to_creature(pt_path, args.grid_size, args.output_dir,
                                skip_existing=args.skip_existing)

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
            result = convert_pt_to_creature(pt_path, args.grid_size, args.output_dir,
                                             skip_existing=args.skip_existing)
            if result is not None:
                success += 1
        print(f"\n{'='*50}")
        print(f"Converted {success}/{len(pt_files)} creatures successfully")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
