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
    /// Grid size (must be power of two, e.g. 64, 128, 256, 512)
    #[arg(long, default_value = "512")]
    grid_size: usize,

    /// Cell size for creature placement grid (default: grid_size / 8)
    #[arg(long, default_value_t = 0)]
    cell_size: usize,

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
// Load all random creatures from seed/
// ---------------------------------------------------------------------------

fn discover_creatures() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir("seed")
        .expect("Failed to read seed/ directory")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let file_name = path.file_stem()?.to_str()?.to_string();
            if file_name.starts_with("random_") && path.extension()? == "json" {
                Some(file_name)
            } else {
                None
            }
        })
        .collect();
    names.sort();
    names
}

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
    let expected = num_kernels * size * size * 8;
    assert_eq!(data.len(), expected, "Kernel FFT file size mismatch");

    let mut kernels = Vec::with_capacity(num_kernels);
    let kernel_len = size * size;
    for k in 0..num_kernels {
        let mut kernel = Vec::with_capacity(kernel_len);
        let base = k * kernel_len * 8;
        for i in 0..kernel_len {
            let offset = base + i * 8;
            let re = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            let im = f32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap());
            kernel.push(num_complex::Complex32::new(re, im));
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

    // Discover creatures
    let creature_names = discover_creatures();
    let num_creatures = creature_names.len();
    assert!(
        num_creatures > 0,
        "No random_* creatures found in seed/ directory"
    );
    println!("Found {} creatures", num_creatures);

    // Load all creature configs
    let configs: Vec<CreatureConfig> = creature_names
        .iter()
        .map(|name| load_creature_config(name))
        .collect();

    // Cell size defaults to seed size (so one creature fits in one cell)
    let seed_size = configs[0].seed_size;
    let cell_size = if cli.cell_size == 0 {
        seed_size
    } else {
        cli.cell_size
    };
    let cells_per_row = grid_size / cell_size;

    // Compute totals
    let num_channels: usize = configs[0].num_channels;
    let num_kernels: usize = configs.iter().map(|c| c.bump_params.num_kernels).sum();

    // Build flat arrays
    let mut all_kernel_m: Vec<f32> = Vec::with_capacity(num_kernels);
    let mut all_kernel_s: Vec<f32> = Vec::with_capacity(num_kernels);
    let mut all_kernel_h: Vec<f32> = Vec::with_capacity(num_kernels);
    for config in &configs {
        all_kernel_m.extend(&config.growth_params.m);
        all_kernel_s.extend(&config.growth_params.s);
        all_kernel_h.extend(&config.growth_params.h);
    }

    // Build C0/C1 mapping: simple round-robin across channels
    let c0: Vec<u32> = (0..num_kernels as u32)
        .map(|k| k % num_channels as u32)
        .collect();
    let c1: Vec<Vec<u32>> = (0..num_channels)
        .map(|c| {
            (0..num_kernels as u32)
                .filter(|&k| k % num_channels as u32 == c as u32)
                .collect()
        })
        .collect();

    println!(
        "{} channels, {} kernels, {} creatures",
        num_channels, num_kernels, num_creatures
    );

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
            limits: wgpu::Limits::default(),
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

    // Upload seeds
    let mut channel_buffers: Vec<Vec<f64>> = (0..num_channels)
        .map(|_| vec![0.0f64; grid_size * grid_size])
        .collect();
    for (ci, config) in configs.iter().enumerate() {
        let row = ci / cells_per_row;
        let col = ci % cells_per_row;
        let offset_x = col * cell_size;
        let offset_y = row * cell_size;

        for c in 0..config.num_channels {
            let src = &config.seed_channels[c];
            for iy in 0..seed_size {
                for ix in 0..seed_size {
                    let dst_idx = (offset_y + iy) * grid_size + (offset_x + ix);
                    let src_idx = iy * seed_size + ix;
                    channel_buffers[c][dst_idx] = src[src_idx];
                }
            }
        }
    }
    for (c, buf) in channel_buffers.iter().enumerate() {
        game.upload_channel(buf, c);
    }

    // Load and upload kernels
    let mut kernel_offset = 0;
    for (ci, config) in configs.iter().enumerate() {
        let name = &creature_names[ci];
        let nk = config.bump_params.num_kernels;
        let kernel_path = format!("kernels/{}_512.bin", name);
        let kernels_fft = load_kernels_fft(&kernel_path, nk, shape);
        for (k, kfft) in kernels_fft.iter().enumerate() {
            game.set_kernel(kfft, kernel_offset + k);
        }
        kernel_offset += nk;
    }

    println!("Flow Lenia: GPU-Accelerated Mass-Conserving CA");
    println!(
        "{} channels, {} kernels, {}x{} grid, {} creatures in {}x{} grid",
        num_channels,
        num_kernels,
        grid_size,
        grid_size,
        num_creatures,
        cells_per_row,
        cells_per_row
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
