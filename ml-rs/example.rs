// Grid size (must be a power of two).
const N: usize = 1024;
// Channels
const C: usize = 3;
// Kernels
const K: usize = C * C;
// Temperature
const TEMP: f32 = 1.0;

/// Add two complex numbers.
#[inline]
fn cadd(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

/// Subtract two complex numbers.
#[inline]
fn csub(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

/// Multiply two complex numbers.
#[inline]
fn cmul(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

/// Create a complex number from polar coordinates.
#[inline]
fn cfrom_polar(r: f32, theta: f32) -> [f32; 2] {
    [r * theta.cos(), r * theta.sin()]
}

// ===========================================================================
// 1D Radix-2 FFT  (Cooley–Tukey, decimation-in-time)
// ===========================================================================

/// In-place forward FFT on a slice of complex numbers.
///
/// Length must be a power of two.  Output is in normal (not bit-reversed)
/// order.  Uses the negative-exponent convention (forward FFT).
fn fft_1d(data: &mut [[f32; 2]]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two(), "FFT length must be a power of two");

    // --- Bit-reversal permutation ---
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            data.swap(i, j);
        }
    }

    // --- Cooley–Tukey radix-2 butterflies ---
    let mut len = 1;
    while len < n {
        let half = len;
        len *= 2;
        let angle = -std::f32::consts::PI / half as f32;
        for i in (0..n).step_by(len) {
            for j in 0..half {
                let w = cfrom_polar(1.0, angle * j as f32);
                let u = data[i + j];
                let v = cmul(data[i + j + half], w);
                data[i + j] = cadd(u, v);
                data[i + j + half] = csub(u, v);
            }
        }
    }
}

/// In-place inverse FFT on a slice of complex numbers.
///
/// Length must be a power of two.  Uses the conjugate-and-scale trick.
fn ifft_1d(data: &mut [[f32; 2]]) {
    // Conjugate → forward FFT → conjugate and scale
    for x in data.iter_mut() {
        x[1] = -x[1];
    }
    fft_1d(data);
    let n = data.len() as f32;
    for x in data.iter_mut() {
        x[1] = -x[1];
        x[0] /= n;
        x[1] /= n;
    }
}

// ===========================================================================
// 2D FFT  (separable: rows then columns)
// ===========================================================================

/// In-place 2D forward FFT on an `N×N` grid.
fn fft_2d(grid: &mut [[[f32; 2]; N]; N]) {
    for row in grid.iter_mut() {
        fft_1d(row);
    }
    let mut col_buf = [[0.0f32; 2]; N];
    for col in 0..N {
        for row in 0..N {
            col_buf[row] = grid[row][col];
        }
        fft_1d(&mut col_buf[..N]);
        for row in 0..N {
            grid[row][col] = col_buf[row];
        }
    }
}

/// In-place 2D inverse FFT on an `N×N` grid.
fn ifft_2d(grid: &mut [[[f32; 2]; N]; N]) {
    for row in grid.iter_mut() {
        ifft_1d(row);
    }
    let mut col_buf = [[0.0f32; 2]; N];
    for col in 0..N {
        for row in 0..N {
            col_buf[row] = grid[row][col];
        }
        ifft_1d(&mut col_buf[..N]);
        for row in 0..N {
            grid[row][col] = col_buf[row];
        }
    }
}

// ===========================================================================
// FFT shift helpers
// ===========================================================================

/// Inverse FFT shift: swap quadrants so the center moves to the corners.
///
/// For even `N` (always true here since N is a power of two), this is
/// equivalent to rolling by `N/2` in both dimensions.
fn ifftshift_2d(grid: &mut [[[f32; 2]; N]; N]) {
    let h = N / 2;
    for i in 0..h {
        for j in 0..h {
            let tmp = grid[i][j];
            grid[i][j] = grid[i + h][j + h];
            grid[i + h][j + h] = tmp;
        }
    }
    for i in 0..h {
        for j in 0..h {
            let tmp = grid[i][j + h];
            grid[i][j + h] = grid[i + h][j];
            grid[i + h][j] = tmp;
        }
    }
}

// ===========================================================================
// Simulation step
// ===========================================================================

/// Run one full MaceLenia simulation step in-place.
///
/// `channels[c][i][j]` is the density at channel `c`, row `i`, col `j`.
/// `kernels_fft[k][i][j]` is the pre-FFT'd kernel `k` (already FFT-shifted).
/// `c0[k]`, `c1[k]` map kernel `k` from input channel to output channel.
/// `mu[k]`, `sigma[k]`, `weight[k]` are the growth parameters for kernel `k`.
fn step_ml(
    channels: &mut [[[f32; N]; N]; C],
    kernels_fft: &[[[[f32; 2]; N]; N]; K],
    c0: &[usize; K],
    c1: &[usize; K],
    mu: &[f32; K],
    sigma: &[f32; K],
    weight: &[f32; K],
) {
    // Temporary buffer: convolution results for each kernel
    let mut conv_result = [[[[0.0f32; 2]; N]; N]; K];

    // =============================================================
    // Phase 1: FFT convolution for each (channel, kernel) pair
    // =============================================================
    for k in 0..K {
        let in_ch = c0[k];

        let mut grid_cpx = [[[0.0f32; 2]; N]; N];
        for i in 0..N {
            for j in 0..N {
                grid_cpx[i][j] = [channels[in_ch][i][j], 0.0];
            }
        }

        fft_2d(&mut grid_cpx);

        for i in 0..N {
            for j in 0..N {
                grid_cpx[i][j] = cmul(grid_cpx[i][j], kernels_fft[k][i][j]);
            }
        }

        ifft_2d(&mut grid_cpx);
        conv_result[k] = grid_cpx;
    }

    // =============================================================
    // Phase 2: Growth + weighted sum → affinity
    // =============================================================
    let mut affinity = [[0.0f32; N]; N];
    for out_ch in 0..C {
        growth_phase(&conv_result, c1, mu, sigma, weight, out_ch, &mut affinity);

        // Copy channel into temp buffer (input == output, so borrow checker
        // needs separate storage for the read source).
        let mut ch_tmp = [[0.0f32; N]; N];
        for i in 0..N {
            for j in 0..N {
                ch_tmp[i][j] = channels[out_ch][i][j];
            }
        }

        // =============================================================
        // Phase 3: Diffusion (mass-conserving 3×3 redistribution)
        // =============================================================
        diffusion_phase(&affinity, &ch_tmp, &mut channels[out_ch]);
    }
}

/// Growth phase: apply bump function, weighted sum, and affinity.
///
/// For a single output channel, accumulates the weighted growth over all
/// kernels that map to it, then computes `A = exp(temp · sum)`.
fn growth_phase(
    conv_result: &[[[[f32; 2]; N]; N]; K],
    c1: &[usize; K],
    mu: &[f32; K],
    sigma: &[f32; K],
    weight: &[f32; K],
    out_ch: usize,
    affinity: &mut [[f32; N]; N],
) {
    let mut wsum = [[0.0f32; N]; N];

    for k in 0..K {
        if c1[k] != out_ch {
            continue;
        }

        let m = mu[k];
        let s = sigma[k];
        let w = weight[k];
        let inv_s2 = 1.0 / (s * s);

        for i in 0..N {
            for j in 0..N {
                let u = conv_result[k][i][j][0];
                let d = u - m;
                // Bump growth: G(u) = 2·exp(-((u-μ)/σ)²/2) - 1
                let g = 2.0 * (-0.5 * d * d * inv_s2).exp() - 1.0;
                wsum[i][j] += g * w;
            }
        }
    }

    // Affinity: A = exp(temp · wsum)
    for i in 0..N {
        for j in 0..N {
            affinity[i][j] = (TEMP * wsum[i][j]).exp();
        }
    }
}

/// Diffusion phase: mass-conserving 3×3 redistribution.
///
/// Computes `Z = Σ₃ₓ₃(A)` then
/// `new[i][j] = A[i][j] · Σ₃ₓ₃(ch / Z)`.
fn diffusion_phase(affinity: &[[f32; N]; N], channel: &[[f32; N]; N], out: &mut [[f32; N]; N]) {
    // Step 1: Z = 3×3 sum of affinity (circular boundary)
    let mut z_sum = [[0.0f32; N]; N];
    for i in 0..N {
        for j in 0..N {
            let mut s = 0.0;
            for di in 0..3 {
                for dj in 0..3 {
                    let ni = (i + di + N - 1) % N;
                    let nj = (j + dj + N - 1) % N;
                    s += affinity[ni][nj];
                }
            }
            z_sum[i][j] = s;
        }
    }

    // Step 2: new[i][j] = A[i][j] · Σ₃ₓ₃( ch / Z )
    for i in 0..N {
        for j in 0..N {
            let mut s = 0.0;
            for di in 0..3 {
                for dj in 0..3 {
                    let ni = (i + di + N - 1) % N;
                    let nj = (j + dj + N - 1) % N;
                    s += channel[ni][nj] / z_sum[ni][nj];
                }
            }
            out[i][j] = affinity[i][j] * s;
        }
    }
}

// ===========================================================================
// Kernel generation  (ring-based Gaussian, matches generate_kernel_json.py)
// ===========================================================================

/// Sigmoid function used for kernel envelope.
fn sigmoid(x: f32) -> f32 {
    0.5 * (x * 0.5).tanh() + 0.5
}

/// Generate ring-based Gaussian spatial kernels and return their FFT.
///
/// Why the FFT of a kernel?
/// -------------------------
/// Convolving two NxN grids directly costs O(N^2 * K^2) — for each output
/// pixel you sum over the whole kernel. The Convolution Theorem lets us do
/// it in O(N^2 log N) instead:
///
///   1. FFT the channel (O(N^2 log N))
///   2. Pointwise multiply with the FFT of the kernel (O(N^2))
///   3. IFFT the result (O(N^2 log N))
///
/// The FFT of a kernel is just the kernel transformed into the frequency
/// domain. We pre-compute it once and reuse it every timestep, so the
/// expensive FFT only happens at setup time, not during simulation.
///
/// Why `ifftshift` before the FFT?
/// --------------------------------
/// A standard FFT expects the "origin" (the kernel centre) at the top-left
/// corner (index [0, 0]). Our kernel is built with its centre at the middle
/// of the grid. `ifftshift` swaps quadrants so the centre moves to [0, 0],
/// making the FFT represent a true convolution (not a shifted one).
///
/// Matches the Python `generate_kernels_fft` in `generate_kernel_json.py`.
fn generate_kernels_fft() -> [[[[f32; 2]; N]; N]; K] {
    let mid = (N / 2) as f32;
    let global_r = 10.0f32;

    // Radii and widths control the shape of each ring kernel.
    //   radius — how far from centre the ring sits (fraction of global_r).
    //   width  — how thick the ring is.
    // Spreading them across [0, 1] gives a diverse set of feature detectors
    // at different scales.
    let radii: [f32; K] = [
        0.25, 0.3167, 0.3833, 0.45, 0.5167, 0.5833, 0.65, 0.7167, 0.85,
    ];
    let widths: [f32; K] = [
        0.03, 0.0411, 0.0522, 0.0633, 0.0744, 0.0856, 0.0967, 0.1078, 0.13,
    ];

    let mut kernels_fft = [[[[0.0f32; 2]; N]; N]; K];

    for k in 0..K {
        let radius = radii[k];
        let width = widths[k];

        // --- Step 1: build the spatial kernel ---
        // Each kernel is a ring: a Gaussian bump centred at a specific radius,
        // multiplied by a sigmoid envelope that smooths the inner edge.
        let mut spatial = [[0.0f32; N]; N];
        for i in 0..N {
            for j in 0..N {
                // Distance from centre of the grid.
                let di = i as f32 - mid;
                let dj = j as f32 - mid;
                let dist = (di * di + dj * dj).sqrt();

                // Scale distance by global_r and the per-kernel radius so
                // different kernels peak at different spatial scales.
                let d_scaled = dist / (global_r * radius);

                // Sigmoid envelope: smooth step from 0 -> 1 around d_scaled = 1.
                // This cuts off the ring beyond its radius.
                let sig = sigmoid(-(d_scaled - 1.0) * 10.0);

                // Gaussian ring centred at d_scaled = 0.5.
                let diff = d_scaled - 0.5;
                let ker_val = (-(diff * diff) / (2.0 * width * width)).exp();

                spatial[i][j] = sig * ker_val;
            }
        }

        // --- Step 2: normalise so the kernel integrates to 1 ---
        // This keeps the convolution output in the same magnitude range
        // regardless of kernel size.
        let total: f32 = spatial.iter().flat_map(|r| r.iter()).sum();
        if total > 0.0 {
            for i in 0..N {
                for j in 0..N {
                    spatial[i][j] /= total;
                }
            }
        }

        // --- Step 3: convert to complex and FFT-shift ---
        // Move the kernel centre from the middle of the grid to the top-left
        // corner so the subsequent FFT produces a true convolution.
        let mut grid = [[[0.0f32; 2]; N]; N];
        for i in 0..N {
            for j in 0..N {
                grid[i][j] = [spatial[i][j], 0.0];
            }
        }
        ifftshift_2d(&mut grid);

        // --- Step 4: FFT into frequency domain ---
        // The result is stored as the "FFT of the kernel". At runtime,
        // convolving a channel with this kernel is just:
        //   FFT(channel) -> pointwise multiply -> IFFT
        fft_2d(&mut grid);
        kernels_fft[k] = grid;
    }

    kernels_fft
}

// ===========================================================================
// Seed generation  (Gaussian blobs with offsets)
// ===========================================================================

/// Generate the initial state: three Gaussian blobs with different positions
/// and sizes per channel.
fn generate_seed() -> [[[f32; N]; N]; C] {
    let mut channels = [[[0.0f32; N]; N]; C];

    let blobs: [(f32, f32, f32); C] = [(0.25, 0.0, 0.0), (0.22, 0.04, 0.0), (0.28, 0.0, 0.04)];

    let half = N as f32 / 2.0;

    for c in 0..C {
        let (sigma, ox, oy) = blobs[c];

        // Precompute 1/(2·σ²) so the inner loop only needs a multiply.
        let inv_sigma2 = 1.0 / (2.0 * sigma * sigma);

        for i in 0..N {
            for j in 0..N {
                // Normalise pixel coordinates to [-1, 1].
                let x = (j as f32 - half) / half;
                let y = (i as f32 - half) / half;

                // Offset the centre of the Gaussian per channel so each
                // blob sits at a different position.  This asymmetry drives
                // directional pattern formation in the first few steps.
                let dx = x - ox;
                let dy = y - oy;

                // 2D Gaussian:  exp(-(dx² + dy²) / (2·σ²))
                channels[c][i][j] = (-(dx * dx + dy * dy) * inv_sigma2).exp();
            }
        }
    }

    channels
}

// ===========================================================================
// Growth parameters
// ===========================================================================

/// Generate growth parameters (mu, sigma, weights) and channel mapping.
fn generate_growth_params() -> ([f32; K], [f32; K], [f32; K], [usize; K], [usize; K]) {
    let mut c0 = [0usize; K];
    let mut c1 = [0usize; K];
    let mut mu = [0.0f32; K];
    let mut sigma = [0.0f32; K];
    let mut weights = [0.0f32; K];

    for k in 0..K {
        let in_ch = k % C;
        let out_ch = k / C;
        c0[k] = in_ch;
        c1[k] = out_ch;

        let perm = (in_ch * C + out_ch) as f32;
        mu[k] = 0.05 + 0.20 * perm / K as f32;
        sigma[k] = 0.03 + 0.12 * perm / K as f32;

        if in_ch == out_ch {
            weights[k] = 0.5;
        } else {
            weights[k] = 0.25;
        }
    }

    (mu, sigma, weights, c0, c1)
}

// ===========================================================================
// PPM output  (for visualisation — the only I/O dependency)
// ===========================================================================

/// Save the current state as a PPM image (P6 binary format).
///
/// Channels are mapped to RGB: c0 → R, c1 → G, c2 → B.
fn save_ppm(channels: &[[[f32; N]; N]; C], path: &str) {
    let mut pixels = Vec::with_capacity(N * N * 3);
    for i in 0..N {
        for j in 0..N {
            let r = (channels[0][i][j] * 1.5).clamp(0.0, 1.0).sqrt();
            let g = (channels[1][i][j] * 1.5).clamp(0.0, 1.0).sqrt();
            let b = (channels[2][i][j] * 1.5).clamp(0.0, 1.0).sqrt();
            pixels.push((r * 255.0) as u8);
            pixels.push((g * 255.0) as u8);
            pixels.push((b * 255.0) as u8);
        }
    }

    if let Ok(mut f) = std::fs::File::create(path) {
        use std::io::Write;
        let header = format!("P6\n{} {}\n255\n", N, N);
        let _ = f.write_all(header.as_bytes());
        let _ = f.write_all(&pixels);
    }
}

// ===========================================================================
// Main
// ===========================================================================

fn main() {
    println!("=== MaceLenia (pure Rust, no GPU, no structs) ===");
    println!("Grid: {}×{}, Channels: {}, Kernels: {}", N, N, C, K);
    println!("Temp: {}", TEMP);
    println!();

    // Generate all parameters as plain arrays
    let mut channels = generate_seed();
    let kernels_fft = generate_kernels_fft();
    let (mu, sigma, weights, c0, c1) = generate_growth_params();

    let initial_mass: f32 = channels
        .iter()
        .flat_map(|c| c.iter().flat_map(|r| r.iter()))
        .sum();
    println!("Initial total mass: {:.6}", initial_mass);
    println!();

    // Run simulation
    let num_steps = 100;
    for step in 0..=num_steps {
        if step % 10 == 0 {
            let mut masses = [0.0f32; C];
            for c in 0..C {
                for i in 0..N {
                    for j in 0..N {
                        masses[c] += channels[c][i][j];
                    }
                }
            }
            let total: f32 = masses.iter().sum();
            print!(
                "Step {:3}:  masses = [{:.4}, {:.4}, {:.4}]  total = {:.6}",
                step, masses[0], masses[1], masses[2], total,
            );
            if step > 0 {
                let drift = (total - initial_mass).abs();
                print!("  drift = {:.2e}", drift);
            }
            println!();
        }

        if step < num_steps {
            step_ml(&mut channels, &kernels_fft, &c0, &c1, &mu, &sigma, &weights);
        }
    }

    // Verify mass conservation
    let final_mass: f32 = channels
        .iter()
        .flat_map(|c| c.iter().flat_map(|r| r.iter()))
        .sum();
    let drift = (final_mass - initial_mass).abs();
    println!();
    println!("Initial mass: {:.10}", initial_mass);
    println!("Final mass:   {:.10}", final_mass);
    println!("Drift:        {:.2e}", drift);
    if drift < 1e-4 {
        println!("✅ Mass conserved (drift < 1e-4)");
    } else {
        println!("⚠️  Mass drift detected: {:.2e}", drift);
    }

    // Save output image
    save_ppm(&channels, "ml_rs_example.ppm");
    println!();
    println!("Saved final state to ml_rs_example.ppm");
}
