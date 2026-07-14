#!/usr/bin/env python3
"""Generate PNG frames from flowlenia_org.py using fixed parameters."""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax.numpy as jnp
import numpy as np
from flowlenia_org import (
    Params, Config, State, KernelComputer, FlowLenia,
)
from PIL import Image

os.makedirs("pngs", exist_ok=True)

parser = argparse.ArgumentParser(description="Generate Flow Lenia PNG frames")
parser.add_argument("--grid-size", type=int, default=64, choices=[64, 128, 256, 512],
                    help="Grid size (default: 64)")
args = parser.parse_args()

GRID = args.grid_size
SX, SY, nb_k, C = GRID, GRID, 2, 2

params = Params(
    r=jnp.array([0.5, 0.8]),
    b=jnp.array([[0.5, 0.3, 0.0], [0.7, 0.2, 0.0]]),
    w=jnp.array([[0.1, 0.05, 0.01], [0.08, 0.06, 0.01]]),
    a=jnp.array([[0.0, 0.5, 0.0], [0.0, 0.4, 0.0]]),
    m=jnp.array([0.1, 0.15]),
    s=jnp.array([0.05, 0.08]),
    h=jnp.array([0.5, 0.8]),
    R=10.0,
)

c0 = [0, 1]
c1 = [[0], [1]]

config = Config(
    SX=SX, SY=SY, nb_k=nb_k, C=C,
    c0=c0, c1=c1,
    dt=0.2, dd=5, sigma=0.65,
    n=2, theta_A=1.0, border='wall',
)

kernel_computer = KernelComputer(SX, SY, nb_k)
compiled = kernel_computer(params)

# Initialize state with a Gaussian blob (variance scales with grid size)
A = np.zeros((SX, SY, C), dtype=np.float64)
cx, cy = SX/2.0, SY/2.0
variance = (SX * SX) / 64.0
for i in range(SX):
    for j in range(SY):
        dx, dy = i - cx, j - cy
        dist = np.sqrt(dx*dx + dy*dy)
        val = np.exp(-dist*dist / variance)
        for c in range(C):
            A[i, j, c] = val * (0.5 + 0.5 * c / C)

init_state = State(A=jnp.array(A))

# Save initial state
def save_frame(arr, step):
    """Save a frame as PNG. arr shape (SX, SY, C)."""
    # Sum channels and normalize to 0-255
    img = np.sum(arr, axis=-1)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo) * 255
    else:
        img = np.zeros_like(img)
    img = img.astype(np.uint8)
    Image.fromarray(img).convert('L').save(f'pngs/py_frame_{step:04d}.png')

save_frame(A, 0)

# Run rollout via FlowLenia
flow_lenia = FlowLenia(config)
final_state, states = flow_lenia.rollout_fn(compiled, init_state, 50)

# Save frames at regular intervals
for step in range(1, 51):
    arr = np.array(states.A[step-1])  # states.A has shape (steps, SX, SY, C)
    save_frame(arr, step)

print(f"Saved 51 Python frames (step 0 + 50 steps) to pngs/ ({GRID}x{GRID})")
print(f"  py_frame_0000.png through py_frame_0050.png")
