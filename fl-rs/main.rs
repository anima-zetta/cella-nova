mod orchestrator;
mod wfft;

use clap::Parser;
use orchestrator::GpuFlowLenia;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use wfft::WgpuContext;

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "fl-rs", about = "Flow Lenia GPU simulation")]
struct Cli {
    /// Grid size (must be power of two, e.g. 64, 128, 256, 512, 1024)
    #[arg(long, default_value = "512")]
    grid_size: usize,

    /// Simulation time step
    #[arg(long, default_value_t = 0.2)]
    dt: f32,

    /// Reintegration stencil radius
    #[arg(long, default_value_t = 5)]
    dd: i32,

    /// Reintegration sigma
    #[arg(long, default_value_t = 0.65)]
    sigma: f32,

    /// Path to trained kernel FFT file (optional)
    #[arg(long)]
    load_kernels: Option<String>,

    /// Video duration in seconds
    #[arg(long, default_value_t = 60)]
    seconds: u32,

    /// Video frame rate
    #[arg(long, default_value_t = 60)]
    fps: u32,

    /// Output directory for video
    #[arg(long, default_value = "videos")]
    output: String,
}

// ---------------------------------------------------------------------------
// Creature config (subset of the JSON)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct GrowthParams {
    m: Vec<f32>,
    s: Vec<f32>,
    h: Vec<f32>,
}

#[derive(Deserialize)]
struct BumpParams {
    num_kernels: usize,
}

#[derive(Deserialize)]
struct CreatureConfig {
    seed_size: usize,
    num_channels: usize,
    seed_channels: Vec<Vec<f64>>,
    bump_params: BumpParams,
    growth_params: GrowthParams,
}

// ---------------------------------------------------------------------------
// Load creature config from seed/
// ---------------------------------------------------------------------------

fn load_creature_config(name: &str) -> CreatureConfig {
    let path = format!("seed/{}.json", name);
    let config_str =
        std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("Failed to read {}", path));
    serde_json::from_str(&config_str).unwrap_or_else(|_| panic!("Failed to parse {}", path))
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
            // Same size: read directly
            for i in 0..kernel_len {
                let offset = base + i * 8;
                let re = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                let im = f32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap());
                kernel[i] = num_complex::Complex32::new(re, im);
            }
        } else {
            // Read source FFT kernel (stored as fft2(ifftshift(spatial)))
            let mut src = vec![num_complex::Complex32::new(0.0, 0.0); src_size * src_size];
            for i in 0..src_size * src_size {
                let offset = base + i * 8;
                let re = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                let im = f32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap());
                src[i] = num_complex::Complex32::new(re, im);
            }

            // fftshift: move DC from corners to center
            let half_src = src_size / 2;
            let mut centered = vec![num_complex::Complex32::new(0.0, 0.0); src_size * src_size];
            for i in 0..src_size {
                for j in 0..src_size {
                    let ni = (i + half_src) % src_size;
                    let nj = (j + half_src) % src_size;
                    centered[ni * src_size + nj] = src[i * src_size + j];
                }
            }

            // Place in center of larger FFT array (zero-pad high frequencies)
            let pad = (size - src_size) / 2;
            for i in 0..src_size {
                for j in 0..src_size {
                    kernel[(pad + i) * size + (pad + j)] = centered[i * src_size + j];
                }
            }

            // ifftshift: move DC from center back to corners
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

fn setup_simulation(cli: &Cli) -> (Arc<WgpuContext>, GpuFlowLenia, usize) {
    let grid_size = cli.grid_size;

    // Load the single creature config
    let config = load_creature_config("big_creature");
    let num_channels = config.num_channels;
    let num_kernels = config.bump_params.num_kernels;

    // Build flat arrays
    let all_kernel_m: Vec<f32> = config.growth_params.m.clone();
    let all_kernel_s: Vec<f32> = config.growth_params.s.clone();
    let all_kernel_h: Vec<f32> = config.growth_params.h.clone();

    // Build C0/C1 mapping: cyclic channel relationship.
    let c0: Vec<u32> = (0..num_kernels as u32)
        .map(|k| k % num_channels as u32)
        .collect();
    let c1: Vec<Vec<u32>> = (0..num_channels)
        .map(|c| {
            let src = ((c as u32) + 2) % 3;
            (0..num_kernels as u32).filter(|&k| k % 3 == src).collect()
        })
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
            label: Some("Flow Lenia Device"),
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

    // Flow Lenia setup
    let shape = grid_size.next_power_of_two();

    let game = GpuFlowLenia::new(
        Arc::clone(&context),
        &[shape, shape],
        num_channels,
        num_kernels,
        &c0,
        &c1,
        &all_kernel_m,
        &all_kernel_s,
        &all_kernel_h,
        cli.dt,
        cli.dd,
        cli.sigma,
    );

    // Upload seed (already at the right grid size)
    for (c, data) in config.seed_channels.iter().enumerate() {
        game.upload_channel(data, c);
    }

    // Load and upload kernels (file matches grid size)
    let kernel_path = format!("kernels/big_creature_{}.bin", grid_size);
    let kernels_fft = load_kernels_fft(&kernel_path, num_kernels, shape);
    for (k, kfft) in kernels_fft.iter().enumerate() {
        game.set_kernel(kfft, k);
    }

    println!("Flow Lenia: GPU-Accelerated Mass-Conserving CA");
    println!(
        "{} channels, {} kernels, {}x{} grid",
        num_channels, num_kernels, grid_size, grid_size,
    );
    println!("Reintegration tracking: dd={}, sigma={}", cli.dd, cli.sigma);

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
    let output_path = format!("{}/output.mp4", cli.output);
    let output_dir = PathBuf::from(&cli.output);

    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    println!(
        "Generating {} frames at {} FPS ({} seconds)",
        total_frames, cli.fps, cli.seconds
    );
    println!("Output: {}", output_path);

    let shape = grid_size;

    // Spawn ffmpeg and pipe raw grayscale frames to its stdin
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

        // Map channels to RGB (matching the render shader):
        //   c0 -> R, c1 -> G, c2 -> B
        //   color = clamp(channel * 1.5, 0, 1), then gamma = sqrt(color)
        // No per-frame normalization -- fixed mapping prevents flickering.
        let mut pixels = vec![0u8; shape * shape * 3];
        for i in 0..shape {
            for j in 0..shape {
                let idx = i * shape + j;
                let c0 = data[0 * shape * shape + idx];
                let c1 = data[1 * shape * shape + idx];
                let c2 = data[2 * shape * shape + idx];
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
