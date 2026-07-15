// -*- coding: utf-8 -*-
// Generate PNG frames using the fl-rs GPU implementation.
#![allow(non_snake_case, dead_code)]

mod orchestrator;
mod wfft;

use orchestrator::GpuFlowLenia;
use std::sync::Arc;
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

    // Parameters matching reference/save_pngs.rs and train/save_frames_png.py
    let global_r = 10.0f32;
    let radii: [f32; 3] = [0.5, 0.8, 0.65];
    let a: [[f32; 3]; 3] = [[0.0, 0.5, 0.0], [0.0, 0.4, 0.0], [0.0, 0.45, 0.0]];
    let w: [[f32; 3]; 3] = [[0.1, 0.05, 0.01], [0.08, 0.06, 0.01], [0.09, 0.055, 0.01]];
    let b: [[f32; 3]; 3] = [[0.5, 0.3, 0.0], [0.7, 0.2, 0.0], [0.6, 0.25, 0.0]];

    let mut kernels = Vec::with_capacity(num_kernels);
    for k in 0..num_kernels {
        // Build spatial kernel
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
        // 2D FFT (using rustfft)
        use rustfft::{num_complex::Complex32, FftPlanner};
        let mut planner = FftPlanner::<f32>::new();
        let mut data: Vec<Complex32> = shifted.iter().map(|&v| Complex32::new(v, 0.0)).collect();
        // FFT rows
        let fft_row = planner.plan_fft_forward(size);
        for i in 0..size {
            let mut row: Vec<Complex32> = (0..size).map(|j| data[i * size + j]).collect();
            fft_row.process(&mut row);
            for j in 0..size {
                data[i * size + j] = row[j];
            }
        }
        // FFT cols
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
// PNG saving
// ---------------------------------------------------------------------------

fn save_png(data: &[f32], size: usize, path: &str) {
    // Sum channels, normalize to 0-255
    let num_channels = data.len() / (size * size);
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    let mut pixels = vec![0u8; size * size];
    for i in 0..size {
        for j in 0..size {
            let mut sum = 0.0f32;
            for c in 0..num_channels {
                sum += data[c * size * size + i * size + j];
            }
            if sum < min_val {
                min_val = sum;
            }
            if sum > max_val {
                max_val = sum;
            }
        }
    }
    let range = max_val - min_val;
    for i in 0..size {
        for j in 0..size {
            let mut sum = 0.0f32;
            for c in 0..num_channels {
                sum += data[c * size * size + i * size + j];
            }
            let normalized = if range > 0.0 {
                (sum - min_val) / range
            } else {
                0.0
            };
            pixels[i * size + j] = (normalized * 255.0) as u8;
        }
    }
    image::save_buffer(
        path,
        &pixels,
        size as u32,
        size as u32,
        image::ColorType::L8,
    )
    .unwrap();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let grid_size: usize = if let Some(pos) = args.iter().position(|a| a == "--grid-size") {
        args.get(pos + 1).and_then(|s| s.parse().ok()).unwrap_or(64)
    } else {
        64
    };
    assert!(
        [64, 128, 256, 512, 1024].contains(&grid_size),
        "Grid size must be 64, 128, 256, 512, or 1024"
    );

    let shape = grid_size.next_power_of_two();
    let num_channels: usize = 3;
    let num_kernels: usize = 3;
    let num_steps: usize = 50;

    let c0: Vec<u32> = vec![0, 1, 2];
    let c1: Vec<Vec<u32>> = vec![vec![0], vec![1], vec![2]];

    let dt: f32 = 0.2;
    let dd: i32 = 5;
    let sigma: f32 = 0.65;

    // Growth params matching reference
    let kernel_m: Vec<f32> = vec![0.1, 0.15, 0.12];
    let kernel_s: Vec<f32> = vec![0.05, 0.08, 0.065];
    let kernel_h: Vec<f32> = vec![0.5, 0.8, 0.65];

    // --- wgpu setup (headless, no window) ---
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

    // Save initial state
    std::fs::create_dir_all("pngs").unwrap();
    let initial_data = game.download_all_channels();
    save_png(
        &initial_data,
        shape,
        &format!("pngs/fl_rs_frame_{:04}.png", 0),
    );
    println!("Saved frame 0");

    // Run steps and save PNGs
    for step in 1..=num_steps {
        game.run_steps(1);
        let data = game.download_all_channels();
        let path = format!("pngs/fl_rs_frame_{:04}.png", step);
        save_png(&data, shape, &path);
        println!("Saved frame {}", step);
    }

    println!("\nDone! Saved {} frames to pngs/", num_steps + 1);
}
