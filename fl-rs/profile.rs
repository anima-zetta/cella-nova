// -*- coding: utf-8 -*-
// Profile each phase of the Flow Lenia GPU pipeline.
#![allow(non_snake_case, dead_code)]

mod orchestrator;
mod wfft;

use orchestrator::GpuFlowLenia;
use std::sync::Arc;
use std::time::Instant;
use wfft::WgpuContext;

// ---------------------------------------------------------------------------
// Kernel generation (matches Python's generate_kernels_fft)
// ---------------------------------------------------------------------------

fn sigmoid(x: f32) -> f32 {
    0.5 * ((x / 2.0).tanh() + 1.0)
}

fn generate_kernels_fft(size: usize) -> Vec<Vec<num_complex::Complex32>> {
    let mid = size as i32 / 2;
    let num_kernels = 3;
    let global_r = 10.0f32;
    let radii: [f32; 3] = [0.5, 0.8, 0.65];
    let a: [[f32; 3]; 3] = [[0.0, 0.5, 0.0], [0.0, 0.4, 0.0], [0.0, 0.45, 0.0]];
    let w: [[f32; 3]; 3] = [[0.1, 0.05, 0.01], [0.08, 0.06, 0.01], [0.09, 0.055, 0.01]];
    let b: [[f32; 3]; 3] = [[0.5, 0.3, 0.0], [0.7, 0.2, 0.0], [0.6, 0.25, 0.0]];

    let mut kernels = Vec::with_capacity(num_kernels);
    for k in 0..num_kernels {
        let mut spatial = vec![0.0f32; size * size];
        for i in 0..size {
            for j in 0..size {
                let di = i as i32 - mid;
                let dj = j as i32 - mid;
                let dist = ((di * di + dj * dj) as f32).sqrt();
                let d_scaled = dist / ((global_r + 15.0) * radii[k]);
                let sig = sigmoid(-(d_scaled - 1.0) * 10.0);
                let mut ker_val = 0.0f32;
                for p in 0..3 {
                    let diff = d_scaled - a[k][p];
                    ker_val += b[k][p] * (-(diff * diff) / w[k][p]).exp();
                }
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
            .unwrap_or(512)
    } else {
        512
    };
    assert!(
        [64, 128, 256, 512, 1024].contains(&grid_size),
        "Grid size must be 64, 128, 256, 512, or 1024"
    );

    let shape = grid_size.next_power_of_two();
    let num_channels: usize = 3;
    let num_kernels: usize = 3;
    let num_warmup: usize = 20;
    let num_profile: usize = 100;

    let c0: Vec<u32> = vec![0, 1, 2];
    let c1: Vec<Vec<u32>> = vec![vec![2], vec![0], vec![1]];

    let dt: f32 = 0.2;
    let dd: i32 = 5;
    let sigma: f32 = 0.65;

    let kernel_m: Vec<f32> = vec![0.1, 0.15, 0.12];
    let kernel_s: Vec<f32> = vec![0.05, 0.08, 0.065];
    let kernel_h: Vec<f32> = vec![0.5, 0.8, 0.65];

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

    // --- Flow Lenia setup ---
    let game = GpuFlowLenia::new(
        Arc::clone(&context),
        &[shape, shape],
        num_channels,
        num_kernels,
        &c0,
        &c1,
        &kernel_m,
        &kernel_s,
        &kernel_h,
        dt,
        dd,
        sigma,
    );

    // Generate and upload kernels
    let kernels_fft = generate_kernels_fft(shape);
    for (k, kfft) in kernels_fft.iter().enumerate() {
        game.set_kernel(kfft, k);
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
    let mut phase_totals = [0.0f64; 5];
    for _ in 0..num_profile {
        let times = game.timed_iterate();
        for i in 0..5 {
            phase_totals[i] += times[i];
        }
    }

    let phase_names = [
        "Convolution (FFT+growth)",
        "Aggregation",
        "Gradient flow",
        "Advection",
        "Buffer copy",
    ];
    let total_phase: f64 = phase_totals.iter().sum();
    for i in 0..5 {
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
