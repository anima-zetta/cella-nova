#!/usr/bin/env python3
"""
Glider Speed Training — Flow Lenia (PyTorch GPU)

Generates a multi-channel asymmetric seed and trains the Flow Lenia kernels
to move it faster or make it more resilient.

Usage:
    python train/train_lenia.py
    python train/train_lenia.py --epochs 200
"""

import argparse
import json
import math
import os
import random
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# =========================================================================
# Constants
# =========================================================================

GRID_SIZE = 512
NUM_CHANNELS = 3
NUM_KERNELS = 9
NUM_STEPS = 40
DT = 0.2
DD = 5
SIGMA = 0.65

C0 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
C1 = [[0, 1, 6], [2, 3, 4], [5, 7, 8]]

GLOBAL_R = 42.0
RADII = [0.8, 0.6, 1.0, 0.7, 0.5, 0.9, 0.65, 0.55, 0.85]
A_FLAT = [
    0.0, 0.6, 0.0,
    0.0, 0.5, 0.0,
    0.0, 0.7, 0.0,
    0.0, 0.55, 0.0,
    0.0, 0.45, 0.0,
    0.0, 0.65, 0.0,
    0.0, 0.5, 0.0,
    0.0, 0.6, 0.0,
    0.0, 0.55, 0.0,
]
W_FLAT = [
    0.08, 0.06, 0.01,
    0.07, 0.05, 0.01,
    0.09, 0.07, 0.01,
    0.08, 0.06, 0.01,
    0.07, 0.05, 0.01,
    0.09, 0.07, 0.01,
    0.08, 0.06, 0.01,
    0.07, 0.05, 0.01,
    0.08, 0.06, 0.01,
]
B_FLAT = [
    0.8, -0.3, 0.0,
    0.7, -0.25, 0.0,
    0.9, -0.35, 0.0,
    0.75, -0.3, 0.0,
    0.65, -0.2, 0.0,
    0.85, -0.35, 0.0,
    0.7, -0.25, 0.0,
    0.6, -0.2, 0.0,
    0.8, -0.3, 0.0,
]

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# =========================================================================
# Sobel kernels
# =========================================================================

SOBEL_X = torch.tensor(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device
)
SOBEL_Y = torch.tensor(
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device
)


# =========================================================================
# PyTorch Flow Lenia (fully batched)
# =========================================================================


class FlowLeniaTorch(torch.nn.Module):
    """PyTorch Flow Lenia with fully batched GPU operations."""

    def __init__(
        self,
        grid_size,
        num_channels,
        num_kernels,
        c0,
        c1,
        dt=0.2,
        dd=5,
        sigma=0.65,
        basal_rate=0.0,
        kinetic_cost=0.0,
    ):
        super().__init__()
        self.size = grid_size
        self.nc = num_channels
        self.nk = num_kernels
        self.dt = dt
        self.dd = dd
        self.sigma = sigma
        self.basal_rate = basal_rate
        self.kinetic_cost = kinetic_cost

        # Channel mapping buffers
        self.register_buffer("c0_idx", torch.tensor(c0, dtype=torch.long))
        # Channel aggregate weight matrix: [nc, nk] where M[c,k]=1 if k contributes to c
        agg = torch.zeros(num_channels, num_kernels)
        for c, ks in enumerate(c1):
            agg[c, ks] = 1.0
        self.register_buffer("channel_agg_weight", agg)

        # Growth params (fixed: μ=0, σ=5, h=1)
        self.register_buffer("kernel_m", torch.zeros(num_kernels, 1, 1))
        self.register_buffer("kernel_s", torch.full((num_kernels, 1, 1), 5.0))
        self.register_buffer("kernel_h", torch.ones(num_kernels, 1, 1))

        # Trainable kernel FFT weights: [nk, H, W, 2] (real, imag)
        self.kernels_fft = torch.nn.Parameter(
            torch.zeros(
                num_kernels, grid_size, grid_size, 2, dtype=torch.float32, device=device
            )
        )

        # Sobel kernels as conv2d weights: [1, 1, 3, 3]
        self.register_buffer("sobel_x", SOBEL_X.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", SOBEL_Y.view(1, 1, 3, 3))

    def generate_kernels(self, global_r, radii, a, w, b):
        """Generate initial kernel FFT weights."""
        kernels_np = np.zeros((self.nk, self.size, self.size), dtype=np.complex64)
        for k in range(self.nk):
            kernel_real = np.zeros((self.size, self.size), dtype=np.float64)
            mid = self.size // 2
            for i in range(self.size):
                for j in range(self.size):
                    dx = i - mid
                    dy = j - mid
                    dist = math.sqrt(dx * dx + dy * dy)
                    d_scaled = dist / ((global_r + 15.0) * radii[k])
                    sig = 0.5 * (math.tanh((-d_scaled + 1.0) * 5.0) + 1.0)
                    ker_val = 0.0
                    for p in range(3):
                        diff = d_scaled - a[k * 3 + p]
                        ker_val += b[k * 3 + p] * math.exp(
                            -(diff * diff) / w[k * 3 + p]
                        )
                    kernel_real[i, j] = sig * ker_val
            total = np.sum(kernel_real)
            if total > 0.0:
                kernel_real /= total
            kernel_real = np.fft.ifftshift(kernel_real)
            kernels_np[k] = np.fft.fft2(kernel_real).astype(np.complex64)

        kt = torch.from_numpy(np.stack([kernels_np.real, kernels_np.imag], axis=-1))
        self.kernels_fft.data.copy_(kt.to(device))

    # ------------------------------------------------------------------
    # Forward pass (all batched)
    # ------------------------------------------------------------------

    def _growth(self, x):
        """Batched growth: x [nk, H, W], params broadcast [nk, 1, 1]."""
        diff = x - self.kernel_m
        g = torch.exp(-(diff * diff) / (2.0 * self.kernel_s * self.kernel_s))
        return (2.0 * g - 1.0) * self.kernel_h

    def _sobel(self, field):
        """Batched Sobel gradient. field: [C, H, W]."""
        C, H, W = field.shape
        pad = 1
        field_pad = F.pad(field, (pad, pad, pad, pad), mode="circular").unsqueeze(
            0
        )  # [1, C, H+2, W+2]
        kx = self.sobel_x.expand(C, 1, 3, 3)  # [C, 1, 3, 3]
        ky = self.sobel_y.expand(C, 1, 3, 3)
        gx = F.conv2d(field_pad, kx, groups=C).squeeze(0)  # [C, H, W]
        gy = F.conv2d(field_pad, ky, groups=C).squeeze(0)
        return gx, gy

    def _reintegrate(self, channels, flow_x, flow_y):
        """Batched semi-Lagrangian advection via manual bilinear interpolation.

        Uses only basic tensor ops (floor, clamp, indexing, arithmetic) that
        are all natively supported on MPS, avoiding grid_sampler_2d_backward.

        channels: [nc, H, W]
        flow_x, flow_y: [nc, H, W]
        """
        nc, H, W = channels.shape
        ma = self.dd - self.sigma

        fx = flow_x.clamp(-ma, ma)
        fy = flow_y.clamp(-ma, ma)

        # Pixel center coordinates [H, W]
        y = torch.arange(H, dtype=torch.float32, device=channels.device)
        x = torch.arange(W, dtype=torch.float32, device=channels.device)
        gy, gx = torch.meshgrid(y, x, indexing="ij")

        # Back-track: source = destination - flow * dt  [nc, H, W]
        src_x = (gx + 0.5) - fx * self.dt
        src_y = (gy + 0.5) - fy * self.dt

        # Integer neighbors (floor/ceil of source coords)
        x0 = torch.floor(src_x)  # [nc, H, W]
        y0 = torch.floor(src_y)
        x1 = x0 + 1.0
        y1 = y0 + 1.0

        # Bilinear weights (fractional part)
        xw = src_x - x0  # weight for x1
        yw = src_y - y0  # weight for y1

        # Clamp coordinates to valid range [0, W-1] / [0, H-1]
        x0c = x0.clamp(0, W - 1).long()
        x1c = x1.clamp(0, W - 1).long()
        y0c = y0.clamp(0, H - 1).long()
        y1c = y1.clamp(0, H - 1).long()

        # Validity masks: 1 if original coord is in-bounds, 0 otherwise
        in_x0 = ((x0 >= 0) & (x0 < W)).float()
        in_x1 = ((x1 >= 0) & (x1 < W)).float()
        in_y0 = ((y0 >= 0) & (y0 < H)).float()
        in_y1 = ((y1 >= 0) & (y1 < H)).float()

        # Gather values at the 4 corners using advanced indexing
        batch_idx = (
            torch.arange(nc, device=channels.device).view(nc, 1, 1).expand(nc, H, W)
        )

        v00 = channels[batch_idx, y0c, x0c] * in_x0 * in_y0
        v01 = channels[batch_idx, y0c, x1c] * in_x1 * in_y0
        v10 = channels[batch_idx, y1c, x0c] * in_x0 * in_y1
        v11 = channels[batch_idx, y1c, x1c] * in_x1 * in_y1

        # Bilinear interpolation
        new_ch = (
            v00 * (1.0 - xw) * (1.0 - yw)
            + v01 * xw * (1.0 - yw)
            + v10 * (1.0 - xw) * yw
            + v11 * xw * yw
        )

        # Metabolic costs (batched)
        flow_mag = torch.sqrt(fx**2 + fy**2 + 1e-8)
        new_ch = (
            new_ch * (1.0 - self.basal_rate * self.dt)
            - self.kinetic_cost * flow_mag * self.dt
        )
        new_ch = new_ch.clamp(min=0.0)

        return new_ch

    def forward(self, channels, num_steps):
        """Run forward pass for num_steps.

        Args:
            channels: [nc, H, W] initial state.
            num_steps: number of timesteps.

        Returns:
            [nc, H, W] final state.
        """
        ch = channels
        H, W = self.size, self.size

        for _ in range(num_steps):
            # --- Batched per-kernel convolution ---
            src = ch[self.c0_idx]  # [nk, H, W]
            kfft = torch.view_as_complex(self.kernels_fft)  # [nk, H, W]
            conv_fft = torch.fft.fft2(src) * kfft  # [nk, H, W]
            conv = torch.fft.ifft2(conv_fft).real / (H * W)  # [nk, H, W]

            # Batched growth
            u = self._growth(conv)  # [nk, H, W]

            # --- Batched channel aggregate ---
            u_channel = (self.channel_agg_weight @ u.reshape(self.nk, -1)).reshape(
                self.nc, H, W
            )

            # --- Sum channels ---
            sum_a = ch.sum(dim=0)  # [H, W]

            # --- Batched Sobel ---
            nabla_ux, nabla_uy = self._sobel(u_channel)  # [nc, H, W]
            nabla_ax, nabla_ay = self._sobel(sum_a.unsqueeze(0))  # [1, H, W]
            nabla_ax, nabla_ay = nabla_ax[0], nabla_ay[0]  # [H, W]

            # --- Flow field ---
            alpha = torch.clamp((ch / self.nc) ** 2, 0.0, 1.0)  # [nc, H, W]
            flow_x = nabla_ux * (1.0 - alpha) - nabla_ax * alpha
            flow_y = nabla_uy * (1.0 - alpha) - nabla_ay * alpha

            # --- Batched reintegration ---
            ch = self._reintegrate(ch, flow_x, flow_y)

        return ch


# =========================================================================
# Helpers
# =========================================================================


def center_of_mass(state, width):
    total = width * width
    ch0 = state[:total]
    cx = 0.0
    cy = 0.0
    s = 0.0
    for i in range(total):
        v = ch0[i]
        if v > 0.001:
            x = i % width
            y = i // width
            cx += x * v
            cy += y * v
            s += v
    if s > 0.0:
        return (cx / s, cy / s)
    return (0.0, 0.0)


def save_kernels(filename, model, num_kernels, grid_size):
    all_kernels = []
    for k in range(num_kernels):
        kfft = torch.view_as_complex(model.kernels_fft[k])
        kflat = kfft.detach().cpu().numpy().ravel()
        for v in kflat:
            all_kernels.append(v.real)
            all_kernels.append(v.imag)

    total = grid_size * grid_size
    header = struct.pack("III", num_kernels, grid_size, total)
    data = struct.pack(f"{len(all_kernels)}f", *all_kernels)
    with open(filename, "wb") as f:
        f.write(header)
        f.write(data)
    print(f"Saved {num_kernels} kernels ({total} elements each) to {filename}")


def save_kernels_png(model, num_kernels, grid_size, out_dir="train/kernels_png"):
    """Save each kernel as a PNG by converting from FFT domain to spatial domain."""
    if not HAS_PIL:
        print("PIL not available, skipping kernel PNGs")
        return
    os.makedirs(out_dir, exist_ok=True)

    for k in range(num_kernels):
        # Convert from FFT domain to spatial domain
        kfft = torch.view_as_complex(model.kernels_fft[k])  # [H, W] complex
        k_spatial = torch.fft.ifft2(kfft).real  # [H, W] real-valued

        # Normalize to [0, 255]
        k_min = k_spatial.min()
        k_max = k_spatial.max()
        if k_max > k_min:
            k_norm = (k_spatial - k_min) / (k_max - k_min)
        else:
            k_norm = torch.zeros_like(k_spatial)
        k_img = (k_norm * 255).cpu().numpy().astype(np.uint8)

        path = os.path.join(out_dir, f"kernel_{k}.png")
        Image.fromarray(k_img, mode="L").save(path)

    print(f"Saved {num_kernels} kernel PNGs to {out_dir}/")


# =========================================================================
# Initial seed generation
# =========================================================================


def generate_initial_glider_seed(size=512, device=device, seed_path="seed/glider.json"):
    """Generates a multi-channel asymmetric localized seed canvas for Flow Lenia.

    Reads channel parameters (sigma, offset_x, offset_y) from a JSON file.
    Creates a 3-channel (RGB) tensor of shape [3, size, size] normalized
    between 0.0 and 1.0.
    """
    with open(seed_path) as f:
        config = json.load(f)

    # Create coordinate mesh grid scaled between -1.0 and 1.0
    y = torch.linspace(-1, 1, size, device=device)
    x = torch.linspace(-1, 1, size, device=device)
    gy, gx = torch.meshgrid(y, x, indexing="ij")

    channels = []
    for ch in config["channels"]:
        sigma = ch["sigma"]
        ox = ch["offset_x"]
        oy = ch["offset_y"]
        channel = torch.exp(-((gx - ox)**2 + (gy - oy)**2) / (2 * sigma**2))
        channels.append(channel)

    initial_state = torch.stack(channels, dim=0)
    return torch.clamp(initial_state, 0.0, 1.0)


# =========================================================================
# Glider Training
# =========================================================================


def run_train_glider(args):
    """Train a glider to move faster using Flow Lenia.

    Approach:
      1. Generate a multi-channel asymmetric seed as the initial state.
      2. Use FlowLeniaTorch with trainable kernel FFT weights.
      3. Place a target at increasing distances along a random direction.
      4. Loss = MSE between final state and target (supervised).
    """
    grid_size = args.grid_size
    num_channels = args.num_channels
    num_kernels = args.num_kernels
    num_steps = args.num_steps
    lr = args.lr
    num_epochs = args.epochs

    print("=" * 60, flush=True)
    print(f"Glider Speed Training — Flow Lenia [{device}]", flush=True)
    print("=" * 60, flush=True)
    print(f"Grid: {grid_size}, Channels: {num_channels}, Kernels: {num_kernels}", flush=True)
    print(f"Steps: {num_steps}, Epochs: {num_epochs}, LR: {lr}", flush=True)
    print(flush=True)

    # Generate initial seed
    seed = generate_initial_glider_seed(grid_size, device)  # [3, H, W]
    center = grid_size // 2
    initial_mass = seed[0].sum()
    print(f"Initial mass (ch0): {initial_mass.item():.1f}", flush=True)

    # Create Flow Lenia model
    model = FlowLeniaTorch(
        grid_size,
        num_channels,
        num_kernels,
        C0,
        C1,
        DT,
        DD,
        SIGMA,
        0.0,
        0.0,
    ).to(device)

    model.generate_kernels(GLOBAL_R, RADII, A_FLAT, W_FLAT, B_FLAT)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Curriculum: increasing target distances
    stage_distances = [0, 8, 16, 24, 32, 48, 64]
    steps_per_stage = max(1, num_epochs // len(stage_distances))

    angle = random.random() * 2.0 * math.pi
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)
    print(f"Direction: ({dir_x:.2f}, {dir_y:.2f})", flush=True)
    print(f"Stages: {len(stage_distances)} distances, ~{steps_per_stage} epochs each", flush=True)
    print(flush=True)

    best_disp = 0.0
    epoch = 0

    for stage_idx, distance in enumerate(stage_distances):
        tcx = int(round(center + dir_x * distance))
        tcy = int(round(center + dir_y * distance))
        tcx = max(16, min(grid_size - 16, tcx))
        tcy = max(16, min(grid_size - 16, tcy))

        # Create target: seed's channel 0 placed at target position
        target = torch.zeros(grid_size, grid_size, device=device)
        # Extract the non-zero region of channel 0
        nonzero = seed[0] > 0.01
        rows = torch.any(nonzero, dim=1).nonzero(as_tuple=True)[0]
        cols = torch.any(nonzero, dim=0).nonzero(as_tuple=True)[0]
        if len(rows) > 0 and len(cols) > 0:
            y0_src = rows[0].item()
            y1_src = rows[-1].item() + 1
            x0_src = cols[0].item()
            x1_src = cols[-1].item() + 1
            patch = seed[0, y0_src:y1_src, x0_src:x1_src]
            ph = patch.shape[0]
            pw = patch.shape[1]
            ty0 = max(0, tcy - ph // 2)
            tx0 = max(0, tcx - pw // 2)
            ty1 = min(grid_size, ty0 + ph)
            tx1 = min(grid_size, tx0 + pw)
            # Adjust patch slice to match
            py0 = 0 if ty0 == 0 else (ph - (ty1 - ty0))
            px0 = 0 if tx0 == 0 else (pw - (tx1 - tx0))
            target[ty0:ty1, tx0:tx1] = patch[py0:py0+ty1-ty0, px0:px0+tx1-tx0]

        for s in range(steps_per_stage):
            if epoch >= num_epochs:
                break

            # Reset state with seed plus tiny noise
            channels = seed + torch.randn_like(seed) * 0.005
            channels = channels.clamp(0.0, 1.0)

            # Forward + loss + backward
            optimizer.zero_grad()
            final = model(channels, num_steps)
            loss = F.mse_loss(final[0], target)
            loss.backward()
            optimizer.step()

            # Evaluate
            with torch.no_grad():
                state = final[0].detach().cpu().numpy().ravel()
                com_x, com_y = center_of_mass(state, grid_size)
                disp = math.sqrt((com_x - center) ** 2 + (com_y - center) ** 2)
                total_mass = float(final[0].sum().item())

                if disp > best_disp and total_mass > 0.3 * initial_mass.item():
                    best_disp = disp
                    save_kernels(args.output, model, num_kernels, grid_size)
                    if args.save_png:
                        save_kernels_png(model, num_kernels, grid_size)

            if epoch % 5 == 0 or epoch == num_epochs - 1:
                print(
                    f"  Epoch {epoch:3d} (d={distance:2d}): "
                    f"mse={loss.item():.6f} "
                    f"disp={disp:.1f}px "
                    f"mass={total_mass:.1f}/{initial_mass.item():.1f} "
                    f"best={best_disp:.1f}",
                    flush=True,
                )

            epoch += 1

    print(f"\n=== Best displacement: {best_disp:.1f}px ===", flush=True)


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="Glider Speed Training (Flow Lenia, PyTorch GPU)")
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--num-channels", type=int, default=NUM_CHANNELS)
    parser.add_argument("--num-kernels", type=int, default=NUM_KERNELS)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--output", type=str, default="train/trained_kernels.bin")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--save-png", action="store_true",
                        help="Save kernel visualizations as PNGs")
    args = parser.parse_args()

    run_train_glider(args)


if __name__ == "__main__":
    main()
