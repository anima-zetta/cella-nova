// -*- coding: utf-8 -*-
// Generate PNG frames using the ml-rs GPU implementation.
#![allow(non_snake_case, dead_code)]

mod orchestrator;
mod wfft;

use orchestrator::GpuMaceLenia;
use std::sync::Arc;
use wfft::WgpuContext;

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
    let num_kernels: usize = 9;
    let num_steps: usize = 50;

    // Build C0/C1 mapping: cyclic channel relationship.
    let c0: Vec<u32> = (0..num_kernels as u32)
        .map(|k| k % num_channels as u32)
        .collect();
    let c1: Vec<u32> = (0..num_kernels as u32)
        .map(|k| (k / num_channels as u32) % num_channels as u32)
        .collect();

    let dt: f32 = 0.2;

    // Growth params matching reference
    let mu: Vec<f32> = (0..num_kernels)
        .map(|k| 0.1 + 0.05 * (k as f32 / num_kernels as f32))
        .collect();
    let sigma: Vec<f32> = (0..num_kernels)
        .map(|k| 0.05 + 0.03 * (k as f32 / num_kernels as f32))
        .collect();
    let weights: Vec<f32> = (0..num_kernels)
        .map(|_| 1.0 / num_channels as f32)
        .collect();

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

    // Generate and upload kernels
    let kernels_fft = generate_kernels_fft(shape, num_kernels);
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
        &format!("pngs/ml_rs_frame_{:04}.png", 0),
    );
    println!("Saved frame 0");

    // Run steps and save PNGs
    for step in 1..=num_steps {
        game.run_steps(1);
        let data = game.download_all_channels();
        let path = format!("pngs/ml_rs_frame_{:04}.png", step);
        save_png(&data, shape, &path);
        println!("Saved frame {}", step);
    }

    println!("\nDone! Saved {} frames to pngs/", num_steps + 1);
}
