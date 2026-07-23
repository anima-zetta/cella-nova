// -*- coding: utf-8 -*-
// MaceLenia GPU simulation with video output.
// If --creature is specified, generates a video for that creature.
// If --creature is omitted, generates videos for ALL creatures in seed/.
// Loads seed/[creature].json and kernels/[creature]_[grid_size].bin.
#![allow(non_snake_case, dead_code)]

mod audio;
mod config;
mod orchestrator;
mod wfft;

use clap::Parser;
use config::{load_config, load_kernels};
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
    /// Creature name. Loads config from seed/{creature}.json
    /// and kernels from kernels/{creature}_{grid_size}.bin.
    /// If omitted, generates videos for ALL creatures in seed/.
    #[arg(long)]
    creature: Option<String>,

    /// Video duration in seconds
    #[arg(long, default_value_t = 60)]
    seconds: u32,

    /// Video frame rate
    #[arg(long, default_value_t = 60)]
    fps: u32,

    /// Output directory for video
    #[arg(long, default_value = "videos")]
    output: String,

    /// Simulation temperature (for diffusion affinity)
    #[arg(long, default_value_t = 1.0)]
    temp: f32,

    /// Generate audio that reacts to grid patterns and mux it into the video
    #[arg(long, default_value_t = false)]
    with_sound: bool,
}

// ---------------------------------------------------------------------------
// WGPU context creation (shared across creatures)
// ---------------------------------------------------------------------------

fn create_context() -> Arc<WgpuContext> {
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

    Arc::new(WgpuContext::from_device(device, queue))
}

// ---------------------------------------------------------------------------
// Setup simulation for a single creature
// ---------------------------------------------------------------------------

fn setup_simulation(
    context: &Arc<WgpuContext>,
    creature: &str,
    temp: f32,
) -> (GpuMaceLenia, usize) {
    let config_path = format!("seed/{}.json", creature);

    // Load everything from config
    let cfg = load_config(&config_path);
    let grid_size = cfg.seed_size;
    let num_channels = cfg.num_channels;
    let num_kernels = cfg.num_kernels;

    println!(
        "  Using creature '{}': {}x{} grid, {} ch, {} kernels",
        creature, grid_size, grid_size, num_channels, num_kernels
    );

    let shape = grid_size.next_power_of_two();

    let game = GpuMaceLenia::new(
        Arc::clone(context),
        &[shape, shape],
        num_channels,
        num_kernels,
        &cfg.c0,
        &cfg.c1,
        &cfg.growth_mu,
        &cfg.growth_sigma,
        &cfg.growth_weights,
        temp,
    );

    // Upload seed from config
    for (c, data) in cfg.seed_channels.iter().enumerate() {
        game.upload_channel(data, c);
    }

    // Upload kernels (in natural order — c0 cycles 0,1,2, c1 increments every C)
    let kernel_path = format!("kernels/{}_{}.bin", creature, shape);
    let kernels_fft = load_kernels(&kernel_path, num_kernels, shape);
    for k in 0..num_kernels {
        game.set_kernel(&kernels_fft[k], k);
    }

    println!("MaceLenia: GPU-Accelerated Multi-channel CA");
    println!(
        "{} channels, {} kernels, {}x{} grid",
        num_channels, num_kernels, grid_size, grid_size,
    );

    (game, grid_size)
}

// ---------------------------------------------------------------------------
// Discover creatures from seed/ directory
// ---------------------------------------------------------------------------

fn discover_creatures() -> Vec<String> {
    let mut creatures = Vec::new();
    let seed_dir = PathBuf::from("seed");
    if !seed_dir.is_dir() {
        eprintln!("Warning: seed/ directory not found");
        return creatures;
    }
    for entry in std::fs::read_dir(&seed_dir).expect("Failed to read seed/ directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "json") {
            if let Some(stem) = path.file_stem() {
                creatures.push(stem.to_string_lossy().to_string());
            }
        }
    }
    creatures.sort();
    creatures
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    let creatures: Vec<String> = if let Some(name) = &cli.creature {
        vec![name.clone()]
    } else {
        let all = discover_creatures();
        if all.is_empty() {
            eprintln!(
                "No creatures found in seed/. Use --creature or run generate_kernel_json.py first."
            );
            std::process::exit(1);
        }
        println!(
            "Found {} creatures in seed/. Generating videos for all...",
            all.len()
        );
        all
    };

    let context = create_context();

    for creature in &creatures {
        println!("\n==============================================");
        println!("  Creature: {}", creature);
        println!("==============================================");

        let (game, grid_size) = setup_simulation(&context, creature, cli.temp);

        let total_frames = (cli.seconds as u64) * (cli.fps as u64);
        let output_path = format!("{}/{}.mp4", cli.output, creature);
        let output_dir = PathBuf::from(&cli.output);

        // Skip if video already exists
        if std::path::Path::new(&output_path).exists() {
            println!("  Video already exists, skipping.");
            continue;
        }

        std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

        println!(
            "Generating {} frames at {} FPS ({} seconds)",
            total_frames, cli.fps, cli.seconds
        );
        println!("Output: {}", output_path);

        let shape = grid_size;

        // --- Audio setup (when --with-sound) ---
        let sample_rate: u32 = 44100;
        let mut audio_synth: Option<audio::AudioSynth> = None;
        let mut audio_samples: Vec<i16> = Vec::new();
        let mut samples_per_frame: usize = 0;
        let mut remainder_frames: usize = 0;

        if cli.with_sound {
            audio_synth = Some(audio::AudioSynth::new(sample_rate));
            let total_samples = (cli.seconds as usize) * (sample_rate as usize);
            let total_frames_usize = total_frames as usize;
            samples_per_frame = total_samples / total_frames_usize;
            remainder_frames = total_samples % total_frames_usize;
            println!(
                "Audio enabled: {} Hz, {} samples/frame (+{} remainder)",
                sample_rate, samples_per_frame, remainder_frames
            );
        }

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
                "-crf",
                "18",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                &output_path,
            ])
            .stdin(std::process::Stdio::piped())
            .spawn()
            .expect("Failed to spawn ffmpeg. Is ffmpeg installed?");

        let mut stdin = ffmpeg.stdin.take().expect("Failed to open ffmpeg stdin");

        for step in 0..total_frames {
            game.iterate_and_render();

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

            // Download pre-rendered RGB24 pixels directly from GPU
            let pixels = game.download_render();

            // --- Audio generation (when --with-sound) ---
            if let Some(ref mut synth) = audio_synth {
                let features = synth.extract_features(&pixels, shape);
                let mut n = samples_per_frame;
                // Distribute remainder samples across the first N frames
                if (step as usize) < remainder_frames {
                    n += 1;
                }
                let frame_audio = synth.generate_frame(&features, n);
                audio_samples.extend_from_slice(&frame_audio);
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

        // --- Mux audio into video (when --with-sound) ---
        if cli.with_sound && !audio_samples.is_empty() {
            let wav_path = format!("{}/{}.wav", cli.output, creature);
            let muxed_path = format!("{}/{}_muxed.mp4", cli.output, creature);

            println!("Writing audio to {}...", wav_path);
            if let Err(e) = audio::write_wav(&wav_path, &audio_samples, sample_rate, 2) {
                eprintln!("Failed to write WAV: {}", e);
            } else {
                println!(
                    "Audio: {:.1}s of stereo PCM",
                    audio_samples.len() as f64 / (sample_rate as f64 * 2.0)
                );

                // Mux video + audio with ffmpeg
                println!("Muxing audio into video...");
                let mux_status = std::process::Command::new("ffmpeg")
                    .args(&[
                        "-y",
                        "-i",
                        &output_path,
                        "-i",
                        &wav_path,
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-shortest",
                        &muxed_path,
                    ])
                    .status()
                    .expect("Failed to spawn ffmpeg for muxing");

                if mux_status.success() {
                    // Replace original with muxed version
                    std::fs::rename(&muxed_path, &output_path)
                        .expect("Failed to replace video with muxed version");
                    // Clean up WAV
                    let _ = std::fs::remove_file(&wav_path);
                    println!("Audio muxed into: {}", output_path);
                } else {
                    eprintln!(
                        "ffmpeg muxing failed (exit code: {:?}). WAV kept at: {}",
                        mux_status.code(),
                        wav_path
                    );
                }
            }
        }
    }
}
