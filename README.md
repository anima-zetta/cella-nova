# Lenia-RS 🌊

A high-performance Rust implementation of Lenia, a continuous cellular automaton system capable of producing complex, life-like patterns and behaviors.

## Features

- 🚀 **Fast FFT-based convolution** for efficient large-scale simulations
- 🔀 **Multi-channel support** with `ExpandedLenia` for complex ecosystems
- 🎨 **Interactive visualization** with real-time rendering using ggez
- ⚡ **Parallel processing** with rayon for multi-threaded computation
- 🧬 **Rich library** of growth functions and kernel generators
- 📊 **Multiple examples** demonstrating predator-prey dynamics and emergent behavior

## Quick Start

### Prerequisites

- Rust (stable toolchain)
- A working graphics environment (for visualization)

### Installation

```bash
git clone https://github.com/yourusername/lenia-rs.git
cd lenia-rs
```

### Running the Simulation

**Basic 3-channel predator-prey ecosystem:**
```bash
cargo run --release
```

**Advanced 4-channel food chain simulation:**
```bash
cargo run --example complex_ecosystem --release
```

> **Note:** Always use `--release` for optimal performance!

## Interactive Controls

- **Space** - Pause/Resume simulation
- **0-3/4** - Switch between individual channel views and composite view
- **R** - Reset simulation with new random patterns
- **Q/Escape** - Quit application

## What You'll See

The simulations demonstrate fascinating emergent behaviors:

- 🔴 **Predators** (red/magenta) - Hunt prey, form colonies
- 🟢 **Prey** (green/cyan) - Exhibit flocking behavior, consume resources
- 🔵 **Environment/Plants** (blue/green) - Renewable resources, spatial diffusion
- 🌀 **Emergent patterns** - Spirals, waves, gliders, and complex interactions

## Library Usage

Add to your `Cargo.toml`:
```toml
[dependencies]
lenia_ca = "0.1"
ndarray = "0.15"
```

### Basic Example

```rust
use lenia_ca::lenias::StandardLenia;
use lenia_ca::{growth_functions, kernels, Lenia, Simulator};
use ndarray::Array2;

fn main() {
    let shape = 256;
    let mut sim = Simulator::<StandardLenia>::new(&[shape, shape]);
    
    // Initialize with a pattern
    let mut initial = Array2::<f64>::zeros([shape, shape]);
    // ... fill with your pattern ...
    sim.fill_channel(&initial.into_dyn(), 0);
    
    // Run simulation
    for _ in 0..1000 {
        sim.iterate();
        // Access state with sim.get_channel_as_ref(0)
    }
}
```

### Multi-Channel Ecosystem

```rust
use lenia_ca::lenias::ExpandedLenia;
use lenia_ca::{growth_functions, kernels, Lenia, Simulator};

fn main() {
    let mut sim = Simulator::<ExpandedLenia>::new(&[256, 256]);
    
    // Set up 2 channels and 2 convolution channels
    sim.set_channels(2);
    sim.set_convolution_channels(2);
    
    // Configure channel 0 (predator)
    sim.set_convolution_channel_source(0, 0);
    sim.set_kernel(kernels::gaussian_donut_2d(15, 0.15), 0);
    sim.set_growth_function(
        growth_functions::standard_lenia,
        vec![0.15, 0.017],
        0
    );
    
    // Configure channel 1 (prey)
    sim.set_convolution_channel_source(1, 1);
    sim.set_kernel(kernels::gaussian_donut_2d(13, 0.12), 1);
    sim.set_growth_function(
        growth_functions::standard_lenia,
        vec![0.12, 0.02],
        1
    );
    
    // Set interaction weights
    sim.set_weights(0, &[1.0, 0.3]);    // Predator: grows + eats prey
    sim.set_weights(1, &[-0.4, 0.9]);   // Prey: harmed by predator + self-growth
    
    sim.set_dt(0.1);
    
    // Run simulation
    loop {
        sim.iterate();
    }
}
```

## Architecture

### Two Lenia Variants

#### `StandardLenia`
- Single channel, single convolution channel
- 2D only
- Pre-configured for *Orbium unicaudatus* glider
- Best for simple experiments

#### `ExpandedLenia`
- Multiple channels and convolution channels
- N-dimensional support
- Complex cross-channel interactions
- Best for multi-species ecosystems

### Working Principle: StandardLenia

1. Perform FFT-based convolution between `channel` and `kernel`
2. Apply `growth_function` to each pixel
3. Multiply by time step `dt` and add to original values
4. Clamp results to [0, 1] range

![Standard Lenia Algorithm](images/standardlenia.png)

### Working Principle: ExpandedLenia

1. For each `convolution_channel`, convolve its source `channel` with its `kernel`
2. Apply each `convolution_channel`'s `growth_function`
3. For each `channel`, compute weighted sum of convolution results
4. Multiply by `dt` and integrate into channels
5. Clamp all values to [0, 1] range

![Expanded Lenia Algorithm](images/expandedlenia.png)

## Available Growth Functions

- `standard_lenia` - Gaussian bump (classic Lenia)
- `multimodal_normal` - Multiple Gaussian bumps
- `polynomial` - Polynomial bump
- `smooth_life_sigmoid_smoothed` - SmoothLife-style transitions
- `conway_game_of_life` - Conway's Game of Life rules
- `pass` - No transformation (kernel dynamics only)

See `src/growth_functions.rs` for full documentation.

## Available Kernels

- `gaussian_donut_2d/nd` - Ring-shaped interaction zones
- `multi_gaussian_donut_2d/nd` - Multiple concentric rings
- `polynomial_nd` - Polynomial-based patterns
- `smoothlife` - SmoothLife kernels
- `conway_game_of_life` - Moore neighborhood

See `src/kernels.rs` for full documentation.

## API Reference

### Key Methods

- `set_channels(n)` - Set number of channels (ExpandedLenia only)
- `set_convolution_channels(n)` - Set number of convolution channels
- `set_convolution_channel_source(conv, src)` - Link convolution to source channel
- `set_kernel(kernel, conv)` - Assign kernel to convolution channel
- `set_growth_function(fn, params, conv)` - Set growth function
- `set_weights(channel, weights)` - Set interaction weights
- `set_dt(dt)` - Set integration time step
- `iterate()` - Advance simulation by one step
- `fill_channel(data, channel)` - Initialize channel with data
- `get_channel_as_ref(channel)` - Access channel data

## Examples

See the `examples/` directory for complete implementations:

- **`complex_ecosystem.rs`** - 4-channel apex-predator-prey-plant system with population tracking

Run examples with:
```bash
cargo run --example complex_ecosystem --release
```

See `examples/README.md` for detailed documentation.

## Performance Tips

1. **Always compile with `--release`** - 10-100x speedup
2. **Use reasonable grid sizes** - 128-512 works well
3. **Limit channels** - More channels = more computation
4. **Profile your code** - Use `cargo flamegraph` to find bottlenecks

## Troubleshooting

**Nothing appears on screen:**
- Check initial patterns are non-zero
- Try different channel views (press 0-3)
- Increase `dt` for faster evolution

**Patterns die out quickly:**
- Reduce negative interaction weights
- Increase growth function `sigma` parameter
- Ensure growth `mu` matches expected convolution output

**Simulation explodes (values blow up):**
- Reduce `dt` (try 0.01-0.05)
- Reduce growth function peaks
- Balance positive/negative feedback

**Compilation errors:**
- Run `rustup update`
- Ensure stable Rust toolchain is active
- Check dependencies are compatible

## Project Structure

```
lenia-rs/
├── src/
│   ├── main.rs              # Main 3-channel application
│   ├── lib.rs               # Core library and Simulator
│   ├── lenias.rs            # StandardLenia & ExpandedLenia
│   ├── fft.rs               # FFT implementation
│   ├── growth_functions.rs  # Growth function library
│   └── kernels.rs           # Kernel generators
├── examples/
│   ├── complex_ecosystem.rs # 4-channel ecosystem
│   └── README.md            # Examples documentation
└── images/                  # Documentation images
```

## Credits

**Original Lenia by Bert Chan**
- Paper: [Lenia - Biology of Artificial Life](https://arxiv.org/abs/1812.05433)
- Website: [chakazul.github.io/lenia.html](https://chakazul.github.io/lenia.html)

**This Implementation:**
- Author: Zoran Lazović
- Language: Rust
- License: See LICENSE file

## Further Reading

- [Original Lenia Paper](https://arxiv.org/abs/1812.05433)
- [Lenia Web Demo](https://chakazul.github.io/lenia.html)
- [SmoothLife](https://arxiv.org/abs/1111.1567) - Predecessor to Lenia
- [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)

## Contributing

Contributions welcome! Areas for improvement:

- More growth functions and kernels
- 3D visualization
- Parameter evolution/optimization
- Additional examples
- Performance optimizations
- Better documentation

## License

See LICENSE file for details.

---

**Enjoy exploring the fascinating world of continuous cellular automata! 🌊✨**