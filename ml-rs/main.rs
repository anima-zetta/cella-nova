// -*- coding: utf-8 -*-
// MaceLenia GPU simulation with video output.
#![allow(non_snake_case, dead_code)]

mod orchestrator;
mod wfft;

use clap::Parser;
use orchestrator::GpuMaceLenia;
use std::path::PathBuf;
use std::sync::Arc;
use wfft::WgpuContext;

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "ml-rs", about = "MaceLenia GPU simulation")]
struct Cli {
    /// Grid size (must be power of two, e.g. 64, 128, 256, 512, 1024)
    #[arg(long, default_value = "256")]
    grid_size: usize,

    /// Simulation time step
    #[arg(long, default_value_t = 0.2)]
    dt: f32,

    /// Number of channels
    #[arg(long, default_value_t = 3)]
    channels: usize,

    /// Number of kernels (typically channels^2)
    #[arg(long, default_value_t = 9)]
    kernels: usize,

    /// Path to trained kernel FFT file (optional)
    #[arg(long)]
    load_kernels: Option<String>,

    /// Video duration in seconds
    #[arg(long, default_value_t = 30)]
    seconds: u32,

    /// Video frame rate
    #[arg(long, default_value_t = 30)]
    fps: u32,

    /// Output directory for video
    #[arg(long, default_value = "videos")]
    output: String,
}

// ---------------------------------------------------------------------------
// Kernel generation (matches Python's generate_kernels_fft)
// ---------------------------------------------------------------------------

fn sigmoid(x: f32) -> f32 {
    0.5 * ((x / 2.0).tanh() + 1.0)
}

fn generate_kernels_fft(size: usize, num_kernels: usize) -> Vec<Vec<num_complex::Complex32>> {
    let mid = size as i32 / 2;
    let global_r = 10.0f32;

    let mut kernels = Vec::with_capacity(num_kernels);
    for k in 0..num_kernels {
        // Build spatial kernel with a simple Gaussian ring
        let mut spatial = vec![0.0f32; size * size];
        let radius = 0.5 + 0.3 * (k as f32 / num_kernels as f32);
        let width = 0.05 + 0.03 * (k as f32 / num_kernels as f32);

        for i in 0..size {
            for j in 0..size {
                let di = i as i32 - mid;
                let dj = j as i32 - mid;
                let dist = ((di * di + dj * dj) as f32).sqrt();
                let d_scaled = dist / (global_r * radius);
                let sig = sigmoid(-(d_scaled - 1.0) * 10.0);
                let diff = d_scaled - 0.5;
                let ker_val = (-(diff * diff) / (2.0 * width * width)).exp();
                spatial[i * size + j] = sig * ker_val;
            }
        }

        // Normalize
        let total: f32 = spatial.iter().sum();
        if total > 0.0 {
            for v in spatial.iter_mut() {
                *v /= total;
            }
        }

        // FFT shift
        let mut shifted = vec![0.0f32; size * size];
        let half = size / 2;
        for i in 0..size {
            for j in 0..size {
                let ni = (i + half) % size;
                let nj = (j + half) % size;
                shifted[ni * size + nj] = spatial[i * size + j];
            }
        }

        // 2D FFT
        use rustfft::{num_complex::Complex32, FftPlanner};
        let mut planner = FftPlanner::<f32>::new();
        let mut data: Vec<Complex32> = shifted.iter().map(|&v| Complex32::new(v, 0.0)).collect();

        let fft_row = planner.plan_fft_forward(size);
        for i in 0..size {
            let mut row: Vec<Complex32> = (0..size).map(|j| data[i * size + j]).collect();
            fft_row.process(&mut row);
            for j in 0..size {
                data[i * size + j] = row[j];
            }
        }

        let fft_col = planner.plan_fft_forward(size);
        for j in 0..size {
            let mut col: Vec<Complex32> = (0..size).map(|i| data[i * size + j]).collect();
            fft_col.process(&mut col);
            for i in 0..size {
                data[i * size + j] = col[i];
            }
        }

        kernels.push(data);
    }
    kernels
}

// ---------------------------------------------------------------------------
// Seed generation (Gaussian blob at center)
// ---------------------------------------------------------------------------

fn generate_seed(size: usize, num_channels: usize) -> Vec<Vec<f64>> {
    let mut channels = Vec::with_capacity(num_channels);
    let variance = (size * size) as f64 / 64.0;
    for c in 0..num_channels {
        let mut ch = vec![0.0f64; size * size];
        let cx = size as f64 / 2.0;
        let cy = size as f64 / 2.0;
        for i in 0..size {
            for j in 0..size {
                let dx = i as f64 - cx;
                let dy = j as f64 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let val = (-dist * dist / variance).exp();
                ch[i * size + j] = val * (0.5 + 0.5 * c as f64 / num_channels as f64);
            }
        }
        channels.push(ch);
    }
    channels
}

// ---------------------------------------------------------------------------
// Kernel loading (pre-FFT'd at GRID_SIZE)
// ---------------------------------------------------------------------------

fn load_kernels_fft(
    path: &str,
    num_kernels: usize,
    size: usize,
) -> Vec<Vec<num_complex::Complex32>> {
    let data = std::fs::read(path).expect("Failed to read kernel FFT file");
    let file_size = data.len();
    let kernel_bytes = file_size / num_kernels;
    let kernel_elems = kernel_bytes / 8;
    let src_size = (kernel_elems as f64).sqrt() as usize;

    let mut kernels = Vec::with_capacity(num_kernels);
    let kernel_len = size * size;

    for k in 0..num_kernels {
        let mut kernel = vec![num_complex::Complex32::new(0.0, 0.0); kernel_len];
        let base = k * kernel_bytes;

        if src_size == size {
            for i in 0..kernel_len {
                let offset = base + i * 8;
                let re = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                let im = f32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap());
                kernel[i] = num_complex::Complex32::new(re, im);
            }
        } else {
            let mut src = vec![num_complex::Complex32::new(0.0, 0.0); src_size * src_size];
            for i in 0..src_size * src_size {
                let offset = base + i * 8;
                let re = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                let im = f32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap());
                src[i] = num_complex::Complex32::new(re, im);
            }

            let half_src = src_size / 2;
            let mut centered = vec![num_complex::Complex32::new(0.0, 0.0); src_size * src_size];
            for i in 0..src_size {
                for j in 0..src_size {
                    let ni = (i + half_src) % src_size;
                    let nj = (j + half_src) % src_size;
                    centered[ni * src_size + nj] = src[i * src_size + j];
                }
            }

            let pad = (size - src_size) / 2;
            for i in 0..src_size {
                for j in 0..src_size {
                    kernel[(pad + i) * size + (pad + j)] = centered[i * src_size + j];
                }
            }

            let half_dst = size / 2;
            let mut result = vec![num_complex::Complex32::new(0.0, 0.0); kernel_len];
            for i in 0..size {
                for j in 0..size {
                    let ni = (i + half_dst) % size;
                    let nj = (j + half_dst) % size;
                    result[ni * size + nj] = kernel[i * size + j];
                }
            }
            kernel = result;
        }
        kernels.push(kernel);
    }
    kernels
}

// ---------------------------------------------------------------------------
// Common setup
// ---------------------------------------------------------------------------

fn setup_simulation(cli: &Cli) -> (Arc<WgpuContext>, GpuMaceLenia, usize) {
    let grid_size = cli.grid_size;
    let num_channels = cli.channels;
    let num_kernels = cli.kernels;

    // Build C0/C1 mapping: cyclic channel relationship.
    // Each kernel maps an input channel to an output channel.
    let c0: Vec<u32> = (0..num_kernels as u32)
        .map(|k| k % num_channels as u32)
        .collect();
    let c1: Vec<u32> = (0..num_kernels as u32)
        .map(|k| (k / num_channels as u32) % num_channels as u32)
        .collect();

    // Growth params: mu, sigma, weights for each kernel (permuted to match Python state[:,:,None] indexing)
    let perm = |k: usize| (k % num_channels) * num_channels + (k / num_channels);
    let mu: Vec<f32> = (0..num_kernels)
        .map(|k| 0.1 + 0.05 * (perm(k) as f32 / num_kernels as f32))
        .collect();
    let sigma: Vec<f32> = (0..num_kernels)
        .map(|k| 0.05 + 0.03 * (perm(k) as f32 / num_kernels as f32))
        .collect();
    let weights: Vec<f32> = (0..num_kernels)
        .map(|_| 1.0 / num_channels as f32)
        .collect();

    println!("{} channels, {} kernels", num_channels, num_kernels);

    // wgpu setup (headless)
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("No suitable GPU adapter found!");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("MaceLenia Device"),
            features: wgpu::Features::empty(),
            limits: wgpu::Limits {
                max_buffer_size: 2 << 30,
                max_storage_buffer_binding_size: 2 << 30,
                ..Default::default()
            },
        },
        None,
    ))
    .expect("Failed to request GPU device!");

    let context = Arc::new(WgpuContext::from_device(device, queue));

    // MaceLenia setup
    let shape = grid_size.next_power_of_two();

    let game = GpuMaceLenia::new(
        Arc::clone(&context),
        &[shape, shape],
        num_channels,
        num_kernels,
        &c0,
        &c1,
        &mu,
        &sigma,
        &weights,
        cli.dt,
    );

    // Generate and upload seed
    let seed = generate_seed(shape, num_channels);
    for (c, data) in seed.iter().enumerate() {
        game.upload_channel(data, c);
    }

    // Generate and upload kernels (permuted to match Python state[:,:,None] indexing)
    if let Some(ref kernel_path) = cli.load_kernels {
        let kernels_fft = load_kernels_fft(kernel_path, num_kernels, shape);
        for k in 0..num_kernels {
            let perm_idx = (k % num_channels) * num_channels + (k / num_channels);
            game.set_kernel(&kernels_fft[perm_idx], k);
        }
    } else {
        let kernels_fft = generate_kernels_fft(shape, num_kernels);
        for k in 0..num_kernels {
            let perm_idx = (k % num_channels) * num_channels + (k / num_channels);
            game.set_kernel(&kernels_fft[perm_idx], k);
        }
    }

    println!("MaceLenia: GPU-Accelerated Multi-channel CA");
    println!(
        "{} channels, {} kernels, {}x{} grid",
        num_channels, num_kernels, grid_size, grid_size,
    );

    (context, game, grid_size)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    run_video(cli);
}

// ---------------------------------------------------------------------------
// Video mode: headless frame generation + ffmpeg encoding
// ---------------------------------------------------------------------------

fn run_video(cli: Cli) {
    let (_context, game, grid_size) = setup_simulation(&cli);

    let total_frames = (cli.seconds as u64) * (cli.fps as u64);
    let output_path = format!("{}/ml_output.mp4", cli.output);
    let output_dir = PathBuf::from(&cli.output);

    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    println!(
        "Generating {} frames at {} FPS ({} seconds)",
        total_frames, cli.fps, cli.seconds
    );
    println!("Output: {}", output_path);

    let shape = grid_size;

    // Spawn ffmpeg and pipe raw RGB frames to its stdin
    let mut ffmpeg = std::process::Command::new("ffmpeg")
        .args(&[
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            &format!("{}x{}", shape, shape),
            "-framerate",
            &cli.fps.to_string(),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            &output_path,
        ])
        .stdin(std::process::Stdio::piped())
        .spawn()
        .expect("Failed to spawn ffmpeg. Is ffmpeg installed?");

    let mut stdin = ffmpeg.stdin.take().expect("Failed to open ffmpeg stdin");

    for step in 0..total_frames {
        game.iterate();

        if step % cli.fps as u64 == 0 {
            let elapsed_secs = step / cli.fps as u64;
            print!(
                "\rFrame {}/{} ({}s)...",
                step + 1,
                total_frames,
                elapsed_secs
            );
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }

        let data = game.download_all_channels();

        // Map channels to RGB:
        //   c0 -> R, c1 -> G, c2 -> B
        //   color = clamp(channel * 1.5, 0, 1), then gamma = sqrt(color)
        let mut pixels = vec![0u8; shape * shape * 3];
        for i in 0..shape {
            for j in 0..shape {
                let idx = i * shape + j;
                let c0 = if cli.channels > 0 {
                    data[0 * shape * shape + idx]
                } else {
                    0.0
                };
                let c1 = if cli.channels > 1 {
                    data[1 * shape * shape + idx]
                } else {
                    0.0
                };
                let c2 = if cli.channels > 2 {
                    data[2 * shape * shape + idx]
                } else {
                    0.0
                };
                let r = (c0 * 1.5).clamp(0.0, 1.0).sqrt();
                let g = (c1 * 1.5).clamp(0.0, 1.0).sqrt();
                let b = (c2 * 1.5).clamp(0.0, 1.0).sqrt();
                let p = idx * 3;
                pixels[p] = (r * 255.0) as u8;
                pixels[p + 1] = (g * 255.0) as u8;
                pixels[p + 2] = (b * 255.0) as u8;
            }
        }

        // Write raw RGB frame to ffmpeg stdin
        use std::io::Write;
        stdin
            .write_all(&pixels)
            .expect("Failed to write frame to ffmpeg");
    }

    println!("\nClosing ffmpeg...");
    drop(stdin);
    let status = ffmpeg.wait().expect("Failed to wait for ffmpeg");

    if status.success() {
        println!("Video saved to: {}", output_path);
    } else {
        eprintln!("ffmpeg encoding failed (exit code: {:?})", status.code());
    }
}
