// -*- coding: utf-8 -*-
// Profile each phase of the MaceLenia GPU pipeline.
#![allow(non_snake_case, dead_code)]

mod orchestrator;
mod wfft;

use orchestrator::GpuMaceLenia;
use std::sync::Arc;
use std::time::Instant;
use wfft::WgpuContext;

// ---------------------------------------------------------------------------
// Kernel generation
// ---------------------------------------------------------------------------

fn sigmoid(x: f32) -> f32 {
    0.5 * ((x / 2.0).tanh() + 1.0)
}

fn generate_kernels_fft(size: usize, num_kernels: usize) -> Vec<Vec<num_complex::Complex32>> {
    let mid = size as i32 / 2;
    let global_r = 10.0f32;

    let mut kernels = Vec::with_capacity(num_kernels);
    for k in 0..num_kernels {
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

        let total: f32 = spatial.iter().sum();
        if total > 0.0 {
            for v in spatial.iter_mut() {
                *v /= total;
            }
        }

        let mut shifted = vec![0.0f32; size * size];
        let half = size / 2;
        for i in 0..size {
            for j in 0..size {
                let ni = (i + half) % size;
                let nj = (j + half) % size;
                shifted[ni * size + nj] = spatial[i * size + j];
            }
        }

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
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let grid_size: usize = if let Some(pos) = args.iter().position(|a| a == "--grid-size") {
        args.get(pos + 1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(256)
    } else {
        256
    };
    assert!(
        [64, 128, 256, 512, 1024].contains(&grid_size),
        "Grid size must be 64, 128, 256, 512, or 1024"
    );

    let shape = grid_size.next_power_of_two();
    let num_channels: usize = 3;
    let num_kernels: usize = 9;
    let num_warmup: usize = 20;
    let num_profile: usize = 100;

    let c0: Vec<u32> = (0..num_kernels as u32)
        .map(|k| k % num_channels as u32)
        .collect();
    let c1: Vec<u32> = (0..num_kernels as u32)
        .map(|k| (k / num_channels as u32) % num_channels as u32)
        .collect();

    let dt: f32 = 0.2;

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

    // --- wgpu setup (headless) ---
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

    // --- MaceLenia setup ---
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
        dt,
    );

    // Generate and upload kernels (permuted to match Python state[:,:,None] indexing)
    let kernels_fft = generate_kernels_fft(shape, num_kernels);
    for k in 0..num_kernels {
        let perm_idx = (k % num_channels) * num_channels + (k / num_channels);
        game.set_kernel(&kernels_fft[perm_idx], k);
    }

    // Generate and upload seed
    let seed = generate_seed(shape, num_channels);
    for (c, data) in seed.iter().enumerate() {
        game.upload_channel(data, c);
    }

    println!("\n=== Profiling {}x{} grid ===", grid_size, grid_size);
    println!("Warming up {} iterations...", num_warmup);
    for _ in 0..num_warmup {
        game.iterate();
    }
    context.device.poll(wgpu::Maintain::Wait);

    // --- Profile full iteration ---
    println!("Profiling {} iterations...", num_profile);
    let start = Instant::now();
    for _ in 0..num_profile {
        game.iterate();
    }
    context.device.poll(wgpu::Maintain::Wait);
    let total_elapsed = start.elapsed();
    let avg_ms = total_elapsed.as_secs_f64() * 1000.0 / num_profile as f64;
    let fps = num_profile as f64 / total_elapsed.as_secs_f64();
    println!("\n  Full iteration: {:8.3} ms avg ({:.1} FPS)", avg_ms, fps);

    // --- Profile per-phase breakdown ---
    println!(
        "\n  Per-phase breakdown ({} warmup + {} measured):",
        num_warmup, num_profile
    );
    println!("  {:30} {:>10} {:>10}", "Phase", "ms", "% of total");
    println!("  {:-<30} {:->10} {:->10}", "", "", "");

    // Warmup
    for _ in 0..num_warmup {
        game.timed_iterate();
    }

    // Profile
    let mut phase_totals = [0.0f64; 3];
    for _ in 0..num_profile {
        let times = game.timed_iterate();
        for i in 0..3 {
            phase_totals[i] += times[i];
        }
    }

    let phase_names = [
        "Convolution (FFT+cmul+IFFT)",
        "Growth + weighted sum + Euler",
        "Buffer copy",
    ];
    let total_phase: f64 = phase_totals.iter().sum();
    for i in 0..3 {
        let avg = phase_totals[i] / num_profile as f64;
        let pct = avg / (total_phase / num_profile as f64) * 100.0;
        println!("  {:30} {:8.3} ms  {:6.1}%", phase_names[i], avg, pct);
    }
    println!("  {:-<30} {:->10} {:->10}", "", "", "");
    println!(
        "  {:30} {:8.3} ms  {:6.1}%",
        "Sum (serialized)",
        total_phase / num_profile as f64,
        100.0
    );
    println!();
    println!("  Note: serialized timing adds submit+sync overhead.");
    println!("  Actual parallel time: {:.3} ms ({:.1} FPS)", avg_ms, fps);
}
