#!/usr/bin/env python3
"""
Fine-tune a creature to move toward a goal.

Full Flow Lenia forward pass matching the Rust/WGPU shader 1:1.
Loads seed + bump params from seed/{creature}.json,
learns bump params + growth function + seed to move toward GOAL.
"""

import argparse
import json
import os
import struct
import numpy as np
import torch
import torch.nn.functional as F

GRID_SIZE = 512
NUM_KERNELS = 9
NUM_CHANNELS = 3
NUM_BUMPS = 3
BPTT_STEPS = 50
T = 10.0
DD = 5
SIGMA_ADV = 2.0
BASAL_RATE = 0.01
KINETIC_COST = 0.001

C0 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
GOAL = (400.0, 400.0)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


# --- Distance map ---
def _dist_grid(size):
    mid = size // 2
    i, j = torch.meshgrid(torch.arange(size, device=DEVICE), torch.arange(size, device=DEVICE), indexing="ij")
    return torch.sqrt((i - mid) ** 2 + (j - mid) ** 2).unsqueeze(0).unsqueeze(0)

DIST_MAP = _dist_grid(GRID_SIZE)


# --- Build kernel from bump params (matching generate_kernel_json.py) ---
def build_kernels(ba, bw, bb, br, global_r):
    K = ba.shape[0]
    dist = DIST_MAP
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


# --- Load creature data ---
def load_creature(creature):
    with open(f"seed/{creature}.json") as f:
        data = json.load(f)
    chs = []
    for ch in data["seed_channels"]:
        arr = np.array(ch, dtype=np.float32).reshape(1, 1, GRID_SIZE, GRID_SIZE)
        chs.append(torch.from_numpy(arr))
    seed = torch.cat(chs, dim=1).to(DEVICE)
    bp = data["bump_params"]
    ba = torch.tensor(bp["a"], device=DEVICE, dtype=torch.float32)
    bw = torch.tensor(bp["w"], device=DEVICE, dtype=torch.float32)
    bb = torch.tensor(bp["b"], device=DEVICE, dtype=torch.float32)
    br = torch.tensor(bp["radii"], device=DEVICE, dtype=torch.float32)
    global_r = float(bp["global_r"])
    return seed, ba, bw, bb, br, global_r


# --- Sobel gradient (3x3, clamp-to-edge, multi-channel) ---
def sobel_gradient(field):
    gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=field.dtype, device=field.device).view(1, 1, 3, 3).float()
    gy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=field.dtype, device=field.device).view(1, 1, 3, 3).float()
    B, C, H, W = field.shape
    padded = F.pad(field, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(padded.view(B * C, 1, H + 2, W + 2), gx).view(B, C, H, W)
    grad_y = F.conv2d(padded.view(B * C, 1, H + 2, W + 2), gy).view(B, C, H, W)
    return grad_x, grad_y


# --- Manual bilinear interpolation (MPS-compatible) ---
def _bilinear_sample(field, x, y):
    _, C, H, W = field.shape
    x = torch.clamp(x, 0.0, W - 1.0)
    y = torch.clamp(y, 0.0, H - 1.0)
    x0 = torch.clamp(torch.floor(x).long(), 0, W - 1)
    x1 = torch.clamp(x0 + 1, 0, W - 1)
    y0 = torch.clamp(torch.floor(y).long(), 0, H - 1)
    y1 = torch.clamp(y0 + 1, 0, H - 1)
    wx = x - x0.float()
    wy = y - y0.float()
    result = torch.zeros_like(field)
    for c in range(C):
        f = field[0, c]
        tl = f[y0[0, c], x0[0, c]]
        tr = f[y0[0, c], x1[0, c]]
        bl = f[y1[0, c], x0[0, c]]
        br = f[y1[0, c], x1[0, c]]
        top = tl + (tr - tl) * wx[0, c]
        bottom = bl + (br - bl) * wx[0, c]
        result[0, c] = top + (bottom - top) * wy[0, c]
    return result


# --- Semi-Lagrangian advection ---
def advect(field, flow_x, flow_y, dt):
    B, C, H, W = field.shape
    gy, gx = torch.meshgrid(torch.arange(H, device=field.device).float(), torch.arange(W, device=field.device).float(), indexing="ij")
    gx = gx.unsqueeze(0).unsqueeze(0)
    gy = gy.unsqueeze(0).unsqueeze(0)
    src_x = gx - flow_x * dt
    src_y = gy - flow_y * dt
    result = _bilinear_sample(field, src_x, src_y)
    flow_mag = torch.sqrt(flow_x ** 2 + flow_y ** 2 + 1e-8)
    result = result * (1.0 - BASAL_RATE * dt) - KINETIC_COST * flow_mag * dt
    return torch.clamp(result, min=0.0)


# --- Center of mass ---
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


# --- Target ---
# --- Export ---
def save_kernels_fft(kfft, path):
    with open(path, "wb") as f:
        for k in range(NUM_KERNELS):
            flat = kfft[k, 0].cpu().numpy().ravel()
            for val in flat:
                f.write(struct.pack("ff", val.real.item(), val.imag.item()))
    print(f"  Saved {path} ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")


def export_seed(state, name):
    os.makedirs("seed", exist_ok=True)
    chs = [state[0, c].detach().cpu().numpy().ravel().tolist() for c in range(NUM_CHANNELS)]
    with open(f"seed/{name}.json", "w") as f:
        json.dump({"grid_size": GRID_SIZE, "num_channels": NUM_CHANNELS, "seed_channels": chs}, f, indent=2)


# --- Full Flow Lenia forward step (matches Rust shader) ---
def forward_step(channels, params):
    """Single Flow Lenia step: conv → growth → sobel → flow → advect."""
    C = channels.shape[1]
    dt = 1.0 / T
    ba, bw, bb, br, gr, hh, mm, ss = params

    # Build kernels
    kernels = build_kernels(ba, bw, bb, br, gr)
    kernels = torch.fft.ifftshift(kernels, dim=(-2, -1))
    K_fft = torch.fft.fft2(kernels)

    # Per-kernel convolution + growth
    K = kernels.shape[0]
    u_all = torch.zeros(1, K, GRID_SIZE, GRID_SIZE, device=channels.device)
    for k in range(K):
        src_c = C0[k]
        A_src = channels[:, src_c:src_c+1, :, :]
        A_fft = torch.fft.fft2(A_src)
        U = torch.real(torch.fft.ifft2(A_fft * K_fft[k:k+1]))
        g = 2.0 * torch.exp(-((U - mm[k]) ** 2) / (2.0 * ss[k] ** 2 + 1e-6)) - 1.0
        u_all[:, k:k+1, :, :] = hh[k] * g

    # Channel aggregation
    growth_channels = torch.zeros(1, C, GRID_SIZE, GRID_SIZE, device=channels.device)
    for c in range(C):
        mask = torch.tensor([C0[k] == c for k in range(K)], device=channels.device, dtype=torch.bool)
        growth_channels[:, c:c+1, :, :] = u_all[:, mask, :, :].sum(dim=1, keepdim=True)

    # Total mass field
    A_total = channels.sum(dim=1, keepdim=True)

    # Sobel gradients
    nabla_u_x, nabla_u_y = sobel_gradient(growth_channels)
    nabla_a_x, nabla_a_y = sobel_gradient(A_total)

    # Alpha-blended flow field
    alpha = torch.clamp((channels / C) ** 2, 0.0, 1.0)
    flow_x = nabla_u_x * (1.0 - alpha) - nabla_a_x * alpha
    flow_y = nabla_u_y * (1.0 - alpha) - nabla_a_y * alpha

    # Semi-Lagrangian advection
    channels_adv = advect(channels, flow_x, flow_y, dt)

    # Add growth
    channels_next = channels_adv + growth_channels * dt
    return torch.clamp(channels_next, 0.0, 1.0)


# --- Training ---
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
    s.requires_grad_(True)  # matches Rust default
    init_opt = init_state.clone().requires_grad_(True)
    init_frozen = init_state.clone()  # frozen copy for shape loss

    opt = torch.optim.Adam([
        {"params": [ba, bw, bb, br, gr, h, m, s], "lr": args.lr},
        {"params": [init_opt], "lr": args.lr * 10},
    ])

    with torch.no_grad():
        start_cx, start_cy = compute_center_of_mass(init_opt)
    params = (ba, bw, bb, br, gr, h, m, s)

    for epoch in range(args.epochs):
        A1 = init_opt
        for _ in range(BPTT_STEPS):
            A1 = forward_step(A1, params)

        cx, cy = compute_center_of_mass(A1)
        dist = torch.sqrt((cx - start_cx) ** 2 + (cy - start_cy) ** 2 + 1e-8)
        movement_loss = -dist / GRID_SIZE
        # Soft shape preservation: penalize mass far from the original blob area
        size_ratio = A1.sum() / init_frozen.sum()
        size_loss = torch.clamp(size_ratio - 2.0, min=0.0)  # penalize if more than 2x original mass
        mass = A1.sum()
        mass_loss = torch.clamp(5000.0 - mass, min=0.0) / 5000.0
        loss = movement_loss + size_loss + mass_loss * 0.5

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            ba.clamp_(0.01, 1.0); bw.clamp_(0.01, 0.3); bb.clamp_(-2.0, 2.0); br.clamp_(0.1, 2.0)
            gr.clamp_(10.0, 100.0); h.clamp_(0.0, 5.0); m.clamp_(0.05, 0.35); s.clamp_(0.1, 10.0)
            init_opt.clamp_(0.0, 1.0)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:4d}/{args.epochs}  COM=({cx.item():6.1f}, {cy.item():6.1f})  dist={dist.item():.1f}  loss={loss.item():.4e}  mass={A1.sum().item():.0f}")

    print(f"\nFinal COM: ({cx.item():.1f}, {cy.item():.1f})  distance moved: {dist.item():.1f}")
    kernels = build_kernels(ba.detach(), bw.detach(), bb.detach(), br.detach(), gr.detach())
    kernels = torch.fft.ifftshift(kernels, dim=(-2, -1))
    save_kernels_fft(torch.fft.fft2(kernels), f"kernels/{args.creature}_finetuned.bin")
    export_seed(init_opt.detach(), f"{args.creature}_finetuned")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--creature", type=str, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    train(p.parse_args())
