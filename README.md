# MaceLenia — GPU-Accelerated DiffusionLenia CA 🧬

A GPU-accelerated **DiffusionLenia** (mass-conserving multi-channel Lenia) simulation in Rust (wgpu/Metal) with a Python creature-generation pipeline. DiffusionLenia replaces the classic growth-and-clamp dynamics with a mass-conserving diffusion step — mass is redistributed rather than created or destroyed.

## Architecture

```
┌──────────────────────────────┐     ┌────────────────────────────┐
│  train/generate_kernel_json  │     │  ml-rs/main.rs             │
│  (Python creature generator) │────▶│  (wgpu GPU simulation)     │
│                              │     │                            │
│  Generates FFT kernels +     │     │  Loads seed/{name}.json    │
│  seed config for each        │     │  and kernels/{name}.bin    │
│  creature                    │     │                            │
└──────────────────────────────┘     │  Headless video generation │
                                     │  via ffmpeg pipe           │
                                     └────────────────────────────┘
```

## Quick Start

### Prerequisites

- Rust (stable toolchain)
- Python 3 with NumPy (for generating creatures)
- ffmpeg (for video encoding)
- A GPU with Metal support (macOS) or Vulkan (Linux/Windows)

### Generate a Creature

```bash
python3 train/generate_kernel_json.py --name my_creature --grid-size 1024
```

This creates `seed/my_creature.json` (config + seed) and `kernels/my_creature_1024.bin` (FFT kernels).

### Run the Simulation

```bash
# Generate a video for one creature
cargo run --release -- --creature my_creature --seconds 60 --fps 60

# Generate videos for ALL creatures in seed/
cargo run --release -- --seconds 30 --fps 30

# With audio synthesis
cargo run --release -- --creature my_creature --with-sound
```

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--creature <name>` | *(all)* | Creature name. Loads `seed/{name}.json` and `kernels/{name}_{size}.bin`. If omitted, generates videos for all creatures in `seed/`. |
| `--seconds <N>` | `60` | Video duration in seconds |
| `--fps <N>` | `60` | Video frame rate |
| `--output <dir>` | `videos` | Output directory for MP4 files |
| `--temp <N>` | `1.0` | Simulation temperature (diffusion affinity multiplier) |
| `--with-sound` | `false` | Generate reactive audio and mux it into the video |

## Additional Binaries

| Binary | Description |
|---|---|
| `ml-rs` | Main video generation (default) |
| `ml-rs-profile` | Per-phase GPU timing breakdown |
| `ml-rs-save-pngs` | Save simulation frames as PNGs |
| `ml-rs-example` | CPU reference implementation |

```bash
# Profile GPU performance
cargo run --release --bin ml-rs-profile -- --creature my_creature

# Save PNG frames
cargo run --release --bin ml-rs-save-pngs -- --creature my_creature
```

## Seed Configuration

Each creature is defined by a JSON file in `seed/`. Example:

```json
{
  "seed_size": 1024,
  "num_channels": 3,
  "num_kernels": 9,
  "seed_channels": [
    [/* 1024×1024 f64 values for channel 0 */],
    [/* channel 1 */],
    [/* channel 2 */]
  ],
  "c0": [0, 1, 2, 0, 1, 2, 0, 1, 2],
  "c1": [0, 0, 0, 1, 1, 1, 2, 2, 2],
  "growth_mu": [0.05, 0.07, 0.10, ...],
  "growth_sigma": [0.03, 0.04, 0.05, ...],
  "growth_weights": [0.5, 0.25, 0.25, ...]
}
```

- **`c0[k]`** — input channel for kernel `k`
- **`c1[k]`** — output channel for kernel `k`
- **`growth_mu[k]`** — mean of the bump growth function for kernel `k`
- **`growth_sigma[k]`** — standard deviation of the bump growth function
- **`growth_weights[k]`** — weight in the per-channel weighted sum

Use `train/generate_kernel_json.py` to create new creatures.

## How DiffusionLenia Works

Each simulation step runs as a series of GPU compute shaders:

1. **FFT Convolution** — Each kernel is convolved with its source channel via FFT (row FFT → column FFT → complex multiply → fused IFFT)
2. **Growth** — Bump function `G(u) = 2·exp(-((u-μ)/σ)²/2) - 1` is applied per kernel, then weighted sum per output channel → affinity buffer
3. **Diffusion (Pass 1)** — `aff_exp = exp(temp · affinity)`, compute `Z = 3×3` neighborhood sum of `aff_exp`
4. **Diffusion (Pass 2)** — `new_state[p] = aff_exp[p] · Σ(neighbors state[n] / Z[n])`
5. **Buffer Copy** — New state copied back to channel buffer

This conserves mass — mass is redistributed across the grid rather than created or destroyed.

## Audio Synthesis

When `--with-sound` is enabled, the simulation generates reactive audio:

- Each of the 3 channels produces tonal sonar pings at distinct frequencies (100 Hz, 300 Hz, 750 Hz)
- Pings fire when spatial variance changes (mass redistribution events)
- Frequency sweeps, low-pass filter, and stereo delay create an alien/underwater soundscape
- Audio is muxed into the final video via ffmpeg

## Project Structure

```
cella-nova/
├── ml-rs/
│   ├── main.rs                  # Video generation entry point
│   ├── lib.rs                   # Crate root
│   ├── config.rs                # Config + kernel loader
│   ├── wfft.rs                  # Shared wgpu context
│   ├── audio.rs                 # Audio synthesis
│   ├── profile.rs               # GPU profiling binary
│   ├── save_pngs.rs             # PNG export binary
│   ├── example.rs               # CPU reference implementation
│   ├── orchestrator/
│   │   ├── mod.rs               # GpuMaceLenia orchestrator
│   │   ├── convolution.rs       # FFT convolution phase
│   │   ├── growth.rs            # Growth + weighted sum phase
│   │   ├── diffusion.rs         # Mass-conserving diffusion phase
│   │   └── render.rs            # GPU render phase
│   └── shaders/
│       ├── compute_1024.wgsl    # WGSL shaders (1024×1024)
│       └── compute_2048.wgsl    # WGSL shaders (2048×2048)
├── train/
│   ├── lenia_org.py             # PyTorch MCLenia base class
│   ├── diff_lenia_org.py        # PyTorch DiffusionLenia
│   ├── generate_kernel_json.py  # Creature generator
│   ├── compare_pngs.py          # Python vs Rust comparison
│   └── save_frames_png.py       # Python PNG export
├── seed/                        # 66 pre-generated creature configs
├── kernels/                     # 66 pre-generated FFT kernel files
├── videos/                      # Output directory
└── Cargo.toml
```

## Credits

- **Lenia** — [Bert Chan (2018)](https://arxiv.org/abs/1812.05433)
- **DiffusionLenia** — Mass-conserving variant using affinity-based diffusion
- **License:** See LICENSE file
