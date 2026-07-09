# Flow Lenia — GPU-Accelerated Mass-Conserving CA 🌊

A GPU-accelerated **Flow Lenia** simulation in Rust (wgpu/Metal) with a PyTorch training pipeline. Flow Lenia is a mass-conserving variant of Lenia that uses Sobel gradients and semi-Lagrangian advection instead of the classic growth-and-clamp dynamics.

## Architecture

```
┌───────────────────────┐     ┌────────────────────────┐
│  train/train_lenia.py │     │  src/main.rs           │
│  (PyTorch training)   │────▶│  (wgpu GPU simulation) │
│                       │     │                        │
│  Trains kernel FFT    │     │  Loads trained kernels │
│  weights via SGD to   │     │  via --load-kernels    │
│  move a glider seed   │     │                        │
│  toward a target      │     │  Real-time rendering   │
└───────────────────────┘     └────────────────────────┘
         │                            │
         └──────────┬─────────────────┘
                    ▼
         seed/glider.json
         (shared seed config)
```

## Quick Start

### Prerequisites

- Rust (stable toolchain)
- Python 3 with PyTorch (for training)
- A GPU with Metal support (macOS) or Vulkan (Linux/Windows)

### Run the Simulation

```bash
cargo run --release
```

This generates a 3-channel Gaussian seed and runs the Flow Lenia simulation on the GPU.

### Load Trained Kernels

```bash
# Train first:
python3 train/train_lenia.py --epochs 200

# Then run with trained kernels:
cargo run --release -- --load-kernels train/trained_kernels.bin
```

## Training

The Python training script (`train/train_lenia.py`) uses **FlowLeniaTorch** — a PyTorch implementation of Flow Lenia with differentiable FFT convolution, Sobel gradients, and semi-Lagrangian advection. It trains the kernel FFT weights to move a glider seed toward targets at increasing distances.

```bash
python3 train/train_lenia.py --epochs 200 --lr 0.1
```

Optional flags:
- `--save-png` — Save kernel visualizations as PNGs
- `--grid-size N` — Grid size (default: 512)
- `--num-steps N` — Simulation steps per epoch (default: 40)
- `--output PATH` — Output path for trained kernels (default: `train/trained_kernels.bin`)

## Seed Configuration

The initial glider seed is defined in `seed/glider.json`. Both Python and Rust read from this file:

```json
{
  "channels": [
    { "sigma": 0.25, "offset_x": 0.0,   "offset_y": 0.0 },
    { "sigma": 0.22, "offset_x": 0.04,  "offset_y": 0.0 },
    { "sigma": 0.28, "offset_x": 0.0,   "offset_y": 0.04 }
  ]
}
```

Each channel is a 2D Gaussian with configurable sigma (size) and offset (asymmetry). Edit this file to change the glider's shape and size without modifying code.

## Controls

| Key | Action |
|---|---|
| **R** | Reset glider seed |
| **Q / Escape** | Quit |

## CLI Flags

| Flag | Description |
|---|---|
| `--load-kernels <file>` | Load trained kernel FFT weights from file |

## How Flow Lenia Works

Each simulation step runs as a series of GPU compute shaders:

1. **FFT Convolution** — Each kernel is convolved with its source channel via FFT
2. **Growth** — A growth function `G(x) = 2*exp(-x²/(2σ²)) - 1` is applied (μ=0, σ=5)
3. **Channel Aggregate** — Kernel outputs are summed per channel using a C0/C1 mapping
4. **Sobel Gradient** — Spatial gradients of the growth field and total mass are computed
5. **Flow Field** — Gradients are combined into a flow field with an alpha-blended mixing term
6. **Semi-Lagrangian Advection** — Mass is advected along the flow field using bilinear interpolation

This conserves mass — mass is moved around rather than created or destroyed.

## Project Structure

```
cella-nova-rs/
├── src/
│   ├── main.rs              # Simulation entry point + rendering
│   ├── lib.rs               # Crate root
│   ├── gpu_flow_lenia.rs    # GPU compute pipeline (WGSL shaders)
│   └── wfft.rs              # GPU FFT implementation
├── train/
│   └── train_lenia.py       # PyTorch training script
├── seed/
│   └── glider.json          # Seed configuration
├── images/                   # Documentation images
└── Cargo.toml
```

## Credits

- **Flow Lenia** — Mass-conserving variant of [Lenia](https://arxiv.org/abs/1812.05433) by Bert Chan
- **License:** See LICENSE file
