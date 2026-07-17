#!/usr/bin/env python3
"""Generate PNG frames matching ml-rs GPU implementation exactly."""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from lenia_org import MCLenia
from PIL import Image

os.makedirs("pngs", exist_ok=True)

parser = argparse.ArgumentParser(description="Generate MaceLenia PNG frames (matching ml-rs)")
parser.add_argument("--grid-size", type=int, default=64, choices=[64, 128, 256, 512],
                    help="Grid size (default: 64)")
args = parser.parse_args()

GRID = args.grid_size
C = 3  # number of channels
K = C * C  # number of kernels
DT = 0.2
NUM_STEPS = 50

# ---------------------------------------------------------------------------
# Kernel generation — matches ml-rs/save_pngs.rs generate_kernels_fft()
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 0.5 * (np.tanh(x / 2.0) + 1.0)

def generate_kernels_fft(size, num_kernels):
    """Generate spatial kernels matching the Rust implementation, return FFT'd."""
    mid = size // 2
    global_r = 10.0

    kernels_spatial = np.zeros((num_kernels, size, size), dtype=np.float32)

    for k in range(num_kernels):
        radius = 0.5 + 0.3 * (k / num_kernels)
        width = 0.05 + 0.03 * (k / num_kernels)

        for i in range(size):
            for j in range(size):
                di = i - mid
                dj = j - mid
                dist = np.sqrt(float(di * di + dj * dj))
                d_scaled = dist / (global_r * radius)
                sig = sigmoid(-(d_scaled - 1.0) * 10.0)
                diff = d_scaled - 0.5
                ker_val = np.exp(-(diff * diff) / (2.0 * width * width))
                kernels_spatial[k, i, j] = sig * ker_val

        # Normalize
        total = kernels_spatial[k].sum()
        if total > 0:
            kernels_spatial[k] /= total

        # FFT shift: swap quadrants
        shifted = np.zeros_like(kernels_spatial[k])
        half = size // 2
        for i in range(size):
            for j in range(size):
                ni = (i + half) % size
                nj = (j + half) % size
                shifted[ni, nj] = kernels_spatial[k, i, j]
        kernels_spatial[k] = shifted

    # 2D FFT using numpy
    kernels_fft = np.fft.fft2(kernels_spatial, axes=(-2, -1))
    return kernels_fft  # (K, H, W) complex64


# ---------------------------------------------------------------------------
# Build parameters matching ml-rs
# ---------------------------------------------------------------------------

# mu/sigma/weights: flat per-kernel in Rust, shaped (B, C, C) in Python
mu = np.zeros((1, C, C), dtype=np.float32)
sigma = np.zeros((1, C, C), dtype=np.float32)
weights = np.zeros((1, C, C), dtype=np.float32)

for k in range(K):
    out_ch = k // C
    in_ch = k % C
    mu[0, out_ch, in_ch] = 0.1 + 0.05 * (k / K)
    sigma[0, out_ch, in_ch] = 0.05 + 0.03 * (k / K)
    weights[0, out_ch, in_ch] = 1.0 / C

# Dummy ring params (needed for MCLenia init but will be overridden)
rings = 3
k_size = 25
beta = np.zeros((1, C, C, rings), dtype=np.float32)
mu_k = np.zeros((1, C, C, rings), dtype=np.float32)
sigma_k = np.ones((1, C, C, rings), dtype=np.float32)  # wide rings = flat kernel

params = {
    'k_size': k_size,
    'mu': torch.from_numpy(mu).float(),
    'sigma': torch.from_numpy(sigma).float(),
    'beta': torch.from_numpy(beta).float(),
    'mu_k': torch.from_numpy(mu_k).float(),
    'sigma_k': torch.from_numpy(sigma_k).float(),
    'weights': torch.from_numpy(weights).float(),
}

# ---------------------------------------------------------------------------
# Initialize state with a Gaussian blob (matches ml-rs generate_seed)
# ---------------------------------------------------------------------------
A = np.zeros((1, C, GRID, GRID), dtype=np.float64)
cx, cy = GRID / 2.0, GRID / 2.0
variance = (GRID * GRID) / 64.0
for i in range(GRID):
    for j in range(GRID):
        dx, dy = i - cx, j - cy
        dist = np.sqrt(dx * dx + dy * dy)
        val = np.exp(-dist * dist / variance)
        for c in range(C):
            A[0, c, i, j] = val * (0.5 + 0.5 * c / C)

state_init = torch.from_numpy(A).float()

# ---------------------------------------------------------------------------
# Create MaceLenia automaton
# ---------------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

lenia = MCLenia(
    size=(1, GRID, GRID),
    dt=DT,
    num_channels=C,
    params=params,
    state_init=state_init,
    device=device,
)

# ---------------------------------------------------------------------------
# Override FFT kernels with ml-rs compatible ones
# ---------------------------------------------------------------------------
kernels_fft_np = generate_kernels_fft(GRID, K)  # (K, H, W) complex64

# Reshape to (B, C, C, H, W) — kernel[out, in] = kernel for in->out
kernels_fft_torch = torch.zeros((1, C, C, GRID, GRID), dtype=torch.complex64)
for k in range(K):
    out_ch = k // C
    in_ch = k % C
    kernels_fft_torch[0, out_ch, in_ch] = torch.from_numpy(kernels_fft_np[k])

lenia.fft_kernel = kernels_fft_torch.to(device)

# ---------------------------------------------------------------------------
# Save frames
# ---------------------------------------------------------------------------
def save_frame(arr, step):
    """Save a frame as PNG. arr shape (C, H, W)."""
    img = np.sum(arr, axis=0)  # (H, W)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo) * 255
    else:
        img = np.zeros_like(img)
    img = img.astype(np.uint8)
    Image.fromarray(img).convert('L').save(f'pngs/py_frame_{step:04d}.png')

# Save initial state
initial_state = lenia.state.detach().cpu().numpy()[0]
save_frame(initial_state, 0)

# Run steps and save frames
for step in range(1, NUM_STEPS + 1):
    lenia.step()
    state = lenia.state.detach().cpu().numpy()[0]
    save_frame(state, step)

print(f"Saved {NUM_STEPS + 1} Python frames (step 0 + {NUM_STEPS} steps) to pngs/ ({GRID}x{GRID})")
print("  py_frame_0000.png through py_frame_{:04d}.png".format(NUM_STEPS))
