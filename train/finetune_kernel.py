#!/usr/bin/env python3
"""
Fine-tune a creature to move.

Full Flow Lenia forward pass matching the Rust/WGPU shader 1:1.
Loads grid-size-independent spatial kernels + seed from files,
pads to training grid size (256x256), and trains bump + growth params.
"""

import argparse
import json
import math
import os
import struct
import numpy as np
import torch
import torch.nn.functional as F

GRID_SIZE = 256
SIM_GRID_SIZE = 512  # simulation grid (for exporting Rust-compatible kernels)
NUM_KERNELS = 9
NUM_CHANNELS = 3
NUM_BUMPS = 3
BPTT_STEPS = 10
DT = 0.2
DD = 5
SIGMA_ADV = 0.65
BASAL_RATE = 0.001
KINETIC_COST = 0.0005

C0 = [0, 0, 0, 1, 1, 1, 2, 2, 2]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# --- Constants matching generate_kernel_json.py ---
KERNEL_SIZE = 256  # spatial kernels stored at this size (matches GRID_256)


# --- Distance map ---
def _dist_grid(size):
    mid = size // 2
    i, j = torch.meshgrid(torch.arange(size, device=DEVICE), torch.arange(size, device=DEVICE), indexing="ij")
    return torch.sqrt((i - mid) ** 2 + (j - mid) ** 2).unsqueeze(0).unsqueeze(0)

DIST_MAP = _dist_grid(GRID_SIZE)
SIM_DIST_MAP = _dist_grid(SIM_GRID_SIZE)  # for exporting kernels at sim resolution


# --- Load spatial kernels from file, pad to GRID_SIZE, FFT ---
def load_kernels_fft(creature):
    path = f"kernels/{creature}_256.bin"
    data = np.fromfile(path, dtype=np.float32)
    kernel_elems = KERNEL_SIZE * KERNEL_SIZE
    expected = NUM_KERNELS * kernel_elems
    assert len(data) == expected, f"Spatial kernel file size mismatch: {len(data)} vs {expected}"

    pad = (GRID_SIZE - KERNEL_SIZE) // 2
    kernels = []
    for k in range(NUM_KERNELS):
        # Read spatial kernel
        spatial = data[k * kernel_elems:(k + 1) * kernel_elems].reshape(KERNEL_SIZE, KERNEL_SIZE)
        # Pad to GRID_SIZE
        padded = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        padded[pad:pad + KERNEL_SIZE, pad:pad + KERNEL_SIZE] = spatial
        # IFFT shift (swap quadrants so kernel is centered for FFT)
        shifted = np.fft.ifftshift(padded)
        # FFT
        kfft = np.fft.fft2(shifted).astype(np.complex64)
        kernels.append(kfft)

    # Stack into [K, H, W] complex64
    stack = np.stack(kernels, axis=0)
    return torch.from_numpy(stack.view(np.float32).reshape(NUM_KERNELS, GRID_SIZE, GRID_SIZE, 2)).to(DEVICE)


# --- Build kernel from bump params (for training) ---
def build_kernels(ba, bw, bb, br, global_r, size=None):
    K = ba.shape[0]
    dist = SIM_DIST_MAP if size == SIM_GRID_SIZE else DIST_MAP
    kernels = []
    for k in range(K):
        d_scaled = dist / ((global_r + 15.0) * br[k] + 1e-6)
        sig = 0.5 * (torch.tanh((-d_scaled + 1.0) * 5.0) + 1.0)
        ker_val = torch.zeros_like(dist)
        for i in range(NUM_BUMPS):
            diff = d_scaled - ba[k, i]
            ker_val = ker_val + bb[k, i] * torch.exp(-(diff * diff) / (bw[k, i] + 1e-6))
        kernel_real = sig * ker_val
        total = kernel_real.sum()
        if total > 0:
            kernel_real = kernel_real / total
        kernels.append(kernel_real)
    return torch.cat(kernels, dim=0)


# --- Load creature data (grid-size independent) ---
def load_creature(creature):
    with open(f"seed/{creature}.json") as f:
        data = json.load(f)

    seed_size = data["seed_size"]

    # Load seed channels and pad to GRID_SIZE
    pad = (GRID_SIZE - seed_size) // 2
    chs = []
    for ch in data["seed_channels"]:
        arr = np.array(ch, dtype=np.float32).reshape(seed_size, seed_size)
        padded = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        padded[pad:pad + seed_size, pad:pad + seed_size] = arr
        chs.append(torch.from_numpy(padded[None, None, :, :]))
    seed = torch.cat(chs, dim=1).to(DEVICE)

    bp = data["bump_params"]
    ba = torch.tensor(bp["a"], device=DEVICE, dtype=torch.float32)
    bw = torch.tensor(bp["w"], device=DEVICE, dtype=torch.float32)
    bb = torch.tensor(bp["b"], device=DEVICE, dtype=torch.float32)
    br = torch.tensor(bp["radii"], device=DEVICE, dtype=torch.float32)
    global_r = float(bp["global_r"])
    return seed, ba, bw, bb, br, global_r


# --- Sobel gradient ---
def sobel_gradient(field):
    gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=field.dtype, device=field.device).view(1, 1, 3, 3).float()
    gy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=field.dtype, device=field.device).view(1, 1, 3, 3).float()
    B, C, H, W = field.shape
    padded = F.pad(field, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(padded.view(B * C, 1, H + 2, W + 2), gx).view(B, C, H, W)
    grad_y = F.conv2d(padded.view(B * C, 1, H + 2, W + 2), gy).view(B, C, H, W)
    return grad_x, grad_y


# --- Reintegration tracking ---
def _reintegrate_impl(channels, flow_x, flow_y):
    B, C, H, W = channels.shape
    ma = DD - SIGMA_ADV
    max_sz = min(1.0, 2.0 * SIGMA_ADV)
    area_norm = 4.0 * SIGMA_ADV * SIGMA_ADV

    flow_x = torch.clamp(flow_x, -ma, ma)
    flow_y = torch.clamp(flow_y, -ma, ma)

    pad = DD
    channels_pad = F.pad(channels, (pad, pad, pad, pad), mode='constant', value=0.0)
    flow_x_pad = F.pad(flow_x, (pad, pad, pad, pad), mode='constant', value=0.0)
    flow_y_pad = F.pad(flow_y, (pad, pad, pad, pad), mode='constant', value=0.0)

    pos_x = torch.arange(W, device=channels.device).float() + 0.5
    pos_y = torch.arange(H, device=channels.device).float() + 0.5

    new_channels = torch.zeros_like(channels)

    for dx in range(-DD, DD + 1):
        for dy in range(-DD, DD + 1):
            nx = dx + pad
            ny = dy + pad

            a = channels_pad[:, :, ny:ny + H, nx:nx + W]
            n_pos_x = (torch.arange(W, device=channels.device).float() + dx) + 0.5
            n_pos_y = (torch.arange(H, device=channels.device).float() + dy) + 0.5

            fx = flow_x_pad[:, :, ny:ny + H, nx:nx + W]
            fy = flow_y_pad[:, :, ny:ny + H, nx:nx + W]

            mu_x = torch.clamp(n_pos_x[None, None, None, :] + fx * DT, SIGMA_ADV, W - SIGMA_ADV)
            mu_y = torch.clamp(n_pos_y[None, None, :, None] + fy * DT, SIGMA_ADV, H - SIGMA_ADV)

            dpx = torch.abs(pos_x[None, None, None, :] - mu_x)
            dpy = torch.abs(pos_y[None, None, :, None] - mu_y)

            sz_x = torch.clamp(0.5 - dpx + SIGMA_ADV, 0.0, max_sz)
            sz_y = torch.clamp(0.5 - dpy + SIGMA_ADV, 0.0, max_sz)

            area = (sz_x * sz_y) / area_norm
            new_channels = new_channels + a * area

    flow_mag = torch.sqrt(flow_x ** 2 + flow_y ** 2 + 1e-8)
    new_channels = new_channels * (1.0 - BASAL_RATE * DT) - KINETIC_COST * flow_mag * DT
    return torch.clamp(new_channels, min=0.0)


def reintegrate(channels, flow_x, flow_y):
    return torch.utils.checkpoint.checkpoint(
        _reintegrate_impl, channels, flow_x, flow_y, use_reentrant=False
    )


# --- Shape metrics ---
def compute_radius(field):
    B, C, H, W = field.shape
    mass = field.sum()
    if mass < 1e-6:
        return torch.tensor(0.0, device=field.device)
    total = field.sum(dim=1, keepdim=True)
    grid_x = torch.arange(W, device=field.device).float()[None, None, None, :]
    grid_y = torch.arange(H, device=field.device).float()[None, None, :, None]
    cx = (total * grid_x).sum() / mass
    cy = (total * grid_y).sum() / mass
    dx2 = (grid_x - cx) ** 2
    dy2 = (grid_y - cy) ** 2
    dist2 = dx2 + dy2
    radius = torch.sqrt((total * dist2).sum() / mass)
    return radius


def compute_channel_fractions(field):
    ch_mass = field.sum(dim=(2, 3))
    total = ch_mass.sum(dim=1, keepdim=True)
    return ch_mass / (total + 1e-8)


def compute_center_of_mass(field):
    size = field.shape[-1]
    grid = torch.arange(size, device=field.device).float()
    mass = field.sum()
    if mass < 1e-6:
        return torch.tensor([size / 2.0], device=field.device), torch.tensor([size / 2.0], device=field.device)
    total = field.sum(dim=1, keepdim=True)
    cx = (total * grid[None, None, None, :]).sum() / mass
    cy = (total * grid[None, None, :, None]).sum() / mass
    return cx, cy


# --- Export ---
def save_kernels_fft(kfft, path):
    with open(path, "wb") as f:
        for k in range(NUM_KERNELS):
            flat = kfft[k, 0].cpu().numpy().ravel()
            for val in flat:
                f.write(struct.pack("ff", val.real.item(), val.imag.item()))
    print(f"  Saved {path} ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")


def export_seed(state, name, h, m, s):
    """Export seed at SEED_SIZE (64) with seed_size field and growth params for Rust."""
    os.makedirs("seed", exist_ok=True)
    pad = (GRID_SIZE - 64) // 2
    chs = []
    for c in range(NUM_CHANNELS):
        ch = state[0, c].detach().cpu().numpy()
        cropped = ch[pad:pad + 64, pad:pad + 64]
        chs.append(cropped.ravel().tolist())
    with open(f"seed/{name}.json", "w") as f:
        json.dump({
            "seed_size": 64,
            "num_channels": NUM_CHANNELS,
            "seed_channels": chs,
            "growth_params": {
                "m": m.detach().cpu().tolist(),
                "s": s.detach().cpu().tolist(),
                "h": h.detach().cpu().tolist(),
            },
        }, f, indent=2)


# --- Full Flow Lenia forward step ---
def forward_step(channels, params):
    C = channels.shape[1]
    ba, bw, bb, br, gr, hh, mm, ss = params

    kernels = build_kernels(ba, bw, bb, br, gr)
    kernels = torch.fft.ifftshift(kernels, dim=(-2, -1))
    K_fft = torch.fft.fft2(kernels)

    K = kernels.shape[0]
    u_all = torch.zeros(1, K, GRID_SIZE, GRID_SIZE, device=channels.device)
    for k in range(K):
        src_c = C0[k]
        A_src = channels[:, src_c:src_c+1, :, :]
        A_fft = torch.fft.fft2(A_src)
        U = torch.real(torch.fft.ifft2(A_fft * K_fft[k:k+1]))
        g = 2.0 * torch.exp(-((U - mm[k]) ** 2) / (2.0 * ss[k] ** 2 + 1e-6)) - 1.0
        u_all[:, k:k+1, :, :] = hh[k] * g

    growth_channels = torch.zeros(1, C, GRID_SIZE, GRID_SIZE, device=channels.device)
    for c in range(C):
        mask = torch.tensor([C0[k] == c for k in range(K)], device=channels.device, dtype=torch.bool)
        growth_channels[:, c:c+1, :, :] = u_all[:, mask, :, :].sum(dim=1, keepdim=True)

    A_total = channels.sum(dim=1, keepdim=True)

    nabla_u_x, nabla_u_y = sobel_gradient(growth_channels)
    nabla_a_x, nabla_a_y = sobel_gradient(A_total)

    alpha = torch.clamp((channels / C) ** 2, 0.0, 1.0)
    flow_x = nabla_u_x * (1.0 - alpha) - nabla_a_x * alpha
    flow_y = nabla_u_y * (1.0 - alpha) - nabla_a_y * alpha

    channels_next = reintegrate(channels, flow_x, flow_y)
    return channels_next


# --- Training ---
def run_epoch(init_state, params, opt, init_frozen, start_cx, start_cy,
              init_radius, init_ch_frac):
    A1 = init_state
    for _ in range(BPTT_STEPS):
        A1 = forward_step(A1, params)

    cx, cy = compute_center_of_mass(A1)
    dist = torch.sqrt((cx - start_cx) ** 2 + (cy - start_cy) ** 2 + 1e-8)
    movement_loss = -dist / GRID_SIZE

    mass = A1.sum()
    size_ratio = mass / init_frozen.sum()
    size_loss = torch.clamp(size_ratio - 2.0, min=0.0)
    mass_loss = torch.clamp(5000.0 - mass, min=0.0) / 5000.0

    radius = compute_radius(A1)
    radius_loss = torch.clamp(radius / (init_radius + 1e-8) - 1.1, min=0.0) * 3.0

    ch_frac = compute_channel_fractions(A1)
    channel_loss = torch.sum((ch_frac - init_ch_frac) ** 2) * NUM_CHANNELS * 3.0

    loss = (movement_loss + size_loss + mass_loss * 0.5
            + radius_loss + channel_loss)

    opt.zero_grad()
    loss.backward()
    opt.step()

    return cx, cy, dist, loss, A1


def train(args):
    print(f"Device: {DEVICE}")
    print(f"Fine-tuning '{args.creature}' to maximize movement from start\n")

    init_state, ba, bw, bb, br, global_r = load_creature(args.creature)

    ba = ba.clone().requires_grad_(True)
    bw = bw.clone().requires_grad_(True)
    bb = bb.clone().requires_grad_(True)
    br = br.clone().requires_grad_(True)
    gr = torch.tensor(global_r, device=DEVICE, dtype=torch.float32).requires_grad_(True)
    h = torch.ones(NUM_KERNELS, device=DEVICE).requires_grad_(True)
    m = torch.zeros(NUM_KERNELS, device=DEVICE).requires_grad_(True)
    s = torch.ones(NUM_KERNELS, device=DEVICE) * 5.0
    s.requires_grad_(True)

    init_frozen = init_state.clone()

    opt = torch.optim.Adam([
        {"params": [ba, bw, bb, br, gr, h, m, s], "lr": args.lr},
    ])

    with torch.no_grad():
        start_cx, start_cy = compute_center_of_mass(init_state)
        init_radius = compute_radius(init_frozen)
        init_ch_frac = compute_channel_fractions(init_frozen)
    params = (ba, bw, bb, br, gr, h, m, s)

    if args.epochs > 0:
        for epoch in range(args.epochs):
            cx, cy, dist, loss, A1 = run_epoch(
                init_state, params, opt, init_frozen,
                start_cx, start_cy, init_radius, init_ch_frac,
            )

            with torch.no_grad():
                ba.clamp_(0.01, 1.0); bw.clamp_(0.01, 0.3); bb.clamp_(-2.0, 2.0); br.clamp_(0.1, 2.0)
                gr.clamp_(10.0, 100.0); h.clamp_(0.0, 5.0); m.clamp_(0.05, 0.35); s.clamp_(0.1, 10.0)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"epoch {epoch + 1:4d}/{args.epochs}  COM=({cx.item():6.1f}, {cy.item():6.1f})  dist={dist.item():.1f}  loss={loss.item():.4e}  mass={A1.sum().item():.0f}")

        print(f"\nFinal COM: ({cx.item():.1f}, {cy.item():.1f})  distance moved: {dist.item():.1f}")
    else:
        print("0 epochs: exporting original parameters unchanged")

    # Export FFT kernels at simulation resolution (512x512) for Rust
    kernels_sim = build_kernels(ba.detach(), bw.detach(), bb.detach(), br.detach(), gr.detach(), size=SIM_GRID_SIZE)
    kernels_sim = torch.fft.ifftshift(kernels_sim, dim=(-2, -1))
    save_kernels_fft(torch.fft.fft2(kernels_sim), f"kernels/{args.creature}_finetuned_512.bin")
    # Export seed at 64x64 with seed_size field
    export_seed(init_state, f"{args.creature}_finetuned", h, m, s)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--creature", type=str, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    train(p.parse_args())
