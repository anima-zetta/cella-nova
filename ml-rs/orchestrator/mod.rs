//! GPU-resident DiffusionLenia (mass-conserving Multi-channel Lenia) simulation.
//!
//! Implements the DiffusionLenia algorithm from `train/diff_lenia_org.py`:
//! - Multi-channel, multi-kernel architecture with FFT-based convolution
//! - Ring-based kernels (pre-FFT'd on CPU)
//! - Bump growth function: G(u) = 2*exp(-((u-mu)/sigma)^2/2) - 1
//! - Weighted sum over input channels per output channel
//! - Affinity: exp(temp * weighted_sum)
//! - Diffusion step: mass-conserving redistribution via 3x3 local normalization
//!
//! All computation runs on the GPU with zero CPU readback between frames.

mod convolution;
mod diffusion;
mod growth;

use crate::wfft::WgpuContext;
use convolution::ConvolutionPhase;
use diffusion::DiffusionPhase;
use growth::GrowthPhase;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// WGSL shaders
// ---------------------------------------------------------------------------

/// Select the compute shader source based on grid size.
fn select_compute_shader(grid_size: usize) -> &'static str {
    match grid_size {
        64 => include_str!("../shaders/compute_64.wgsl"),
        128 => include_str!("../shaders/compute_128.wgsl"),
        256 => include_str!("../shaders/compute_256.wgsl"),
        512 => include_str!("../shaders/compute_512.wgsl"),
        1024 => include_str!("../shaders/compute_1024.wgsl"),
        2048 => include_str!("../shaders/compute_2048.wgsl"),
        _ => panic!(
            "Unsupported grid size: {}. Supported sizes: 64, 128, 256, 512, 1024, 2048",
            grid_size
        ),
    }
}

// ===========================================================================
// GpuMaceLenia — top-level simulation orchestrator
// ===========================================================================

pub struct GpuMaceLenia {
    context: Arc<WgpuContext>,
    shape: Vec<usize>,
    total_elements: usize,
    num_channels: usize,
    num_kernels: usize,

    // Channel mapping
    c0: Vec<u32>, // input channel for each kernel
    c1: Vec<u32>, // output channel for each kernel

    // --- Packed GPU buffers ---
    /// All channels packed: [C * H * W] f32
    channel_buffer: wgpu::Buffer,
    /// Output of diffusion step: [C * H * W] f32
    new_channel_buffer: wgpu::Buffer,
    /// Affinity buffer (weighted sum before diffusion): [C * H * W] f32
    affinity_buffer: wgpu::Buffer,
    /// Temp buffer for Z values (3x3 sum of aff_exp): [C * H * W] f32
    z_buffer: wgpu::Buffer,
    /// All kernels FFT'd: [K * H * W] vec2<f32>
    kernel_buffer: wgpu::Buffer,

    // --- Phase sub-structs ---
    convolution: ConvolutionPhase,
    growth: GrowthPhase,
    diffusion: DiffusionPhase,
}

impl GpuMaceLenia {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: Arc<WgpuContext>,
        shape: &[usize],
        num_channels: usize,
        num_kernels: usize,
        c0: &[u32],
        c1: &[u32],
        mu: &[f32],
        sigma: &[f32],
        weights: &[f32],
        temp: f32,
    ) -> Self {
        let device = &context.device;
        let queue = &context.queue;

        assert_eq!(shape.len(), 2, "DiffusionLenia requires 2D grids");
        let total_elements: usize = shape.iter().product();

        // --- Create all shared GPU buffers ---
        let make_storage = |label: &str, size: u64| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let ch_size = (total_elements * num_channels * 4) as u64;
        let channel_buffer = make_storage("ml::channel", ch_size);
        let new_channel_buffer = make_storage("ml::new_channel", ch_size);
        let affinity_buffer = make_storage("ml::affinity", ch_size);
        let z_buffer = make_storage("ml::z_buffer", ch_size);

        let k_size = (total_elements * num_kernels * 8) as u64;
        let kernel_buffer = make_storage("ml::kernel", k_size);

        let conv_size = (total_elements * num_kernels * 8) as u64;
        let conv_buffer = make_storage("ml::conv", conv_size);

        // --- Initialize phase sub-structs ---
        let compute_shader = select_compute_shader(shape[0]);

        let convolution = ConvolutionPhase::new(
            device,
            queue,
            compute_shader,
            shape,
            total_elements,
            num_kernels,
            num_channels,
            c0,
            conv_buffer,
            &channel_buffer,
            &kernel_buffer,
        );

        let growth = GrowthPhase::new(
            device,
            queue,
            compute_shader,
            shape,
            num_kernels,
            num_channels,
            c1,
            mu,
            sigma,
            weights,
            &convolution.conv_buffer,
            &affinity_buffer,
        );

        let diffusion = DiffusionPhase::new(
            device,
            queue,
            compute_shader,
            shape,
            num_channels,
            temp,
            &affinity_buffer,
            &z_buffer,
            &channel_buffer,
            &new_channel_buffer,
        );

        GpuMaceLenia {
            context,
            shape: shape.to_vec(),
            total_elements,
            num_channels,
            num_kernels,
            c0: c0.to_vec(),
            c1: c1.to_vec(),
            channel_buffer,
            new_channel_buffer,
            affinity_buffer,
            z_buffer,
            kernel_buffer,
            convolution,
            growth,
            diffusion,
        }
    }

    // --- Accessors ---

    pub fn channel_buffer(&self) -> &wgpu::Buffer {
        &self.channel_buffer
    }

    /// Upload a pre-FFT'd kernel at the given index.
    pub fn set_kernel(&self, kernel_fft: &[num_complex::Complex32], kernel_idx: usize) {
        assert_eq!(kernel_fft.len(), self.total_elements);
        let offset = (kernel_idx * self.total_elements * 8) as u64;
        let data: Vec<[f32; 2]> = kernel_fft.iter().map(|c| [c.re, c.im]).collect();
        self.context
            .queue
            .write_buffer(&self.kernel_buffer, offset, bytemuck::cast_slice(&data));
    }

    /// Upload channel data at the given index.
    pub fn upload_channel(&self, data: &[f64], channel: usize) {
        assert_eq!(data.len(), self.total_elements);
        let offset = (channel * self.total_elements * 4) as u64;
        let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        self.context.queue.write_buffer(
            &self.channel_buffer,
            offset,
            bytemuck::cast_slice(&f32_data),
        );
    }

    /// Run N iterations (forward pass).
    pub fn run_steps(&self, n: usize) {
        for _ in 0..n {
            self.iterate();
        }
    }

    /// Download all channel data concatenated.
    pub fn download_all_channels(&self) -> Vec<f32> {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let total_floats = self.total_elements * self.num_channels;
        let total_bytes = (total_floats * 4) as u64;

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ml::readback_all"),
            size: total_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ml::download_all"),
        });
        encoder.copy_buffer_to_buffer(&self.channel_buffer, 0, &readback, 0, total_bytes);
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let view = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        readback.unmap();
        result
    }

    /// Download the conv buffer (for debugging).
    /// Returns the real part of each complex element.
    pub fn download_conv_buffer(&self) -> Vec<f32> {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let total_complex = self.total_elements * self.num_kernels;
        let total_bytes = (total_complex * 8) as u64;

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ml::readback_conv"),
            size: total_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ml::download_conv"),
        });
        encoder.copy_buffer_to_buffer(&self.convolution.conv_buffer, 0, &readback, 0, total_bytes);
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let view = slice.get_mapped_range();
        // Extract real parts only
        let raw: &[f32] = bytemuck::cast_slice(&view);
        let mut result = Vec::with_capacity(total_complex);
        for i in 0..total_complex {
            result.push(raw[i * 2]);
        }
        drop(view);
        readback.unmap();
        result
    }

    // =======================================================================
    // Main iteration
    // =======================================================================

    /// Performs a single DiffusionLenia iteration entirely on the GPU.
    ///
    /// Algorithm:
    /// 1. For each input channel: FFT → complex multiply with each kernel → IFFT
    /// 2. For each pixel: growth function → weighted sum → affinity buffer
    /// 3. Diffusion: aff_exp = exp(temp * affinity), Z = 3x3 sum, redistribute mass
    /// 4. Copy new_channel → channel
    pub fn iterate(&self) {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let total = self.total_elements as u32;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ml::iterate"),
        });

        // Phase 1: FFT convolution for all channels and kernels
        self.convolution.run(
            device,
            &mut encoder,
            &self.channel_buffer,
            &self.kernel_buffer,
            &self.c0,
            self.num_kernels,
            self.num_channels,
            self.total_elements,
            &self.shape,
        );

        // Phase 2: Growth + weighted sum → affinity buffer
        self.growth.run(&mut encoder, total);

        // Phase 3: Diffusion (pass 1: aff_exp + Z, pass 2: redistribute)
        self.diffusion.run(&mut encoder, total);

        // Phase 4: Copy new_channel → channel
        encoder.copy_buffer_to_buffer(
            &self.new_channel_buffer,
            0,
            &self.channel_buffer,
            0,
            (self.total_elements * self.num_channels * 4) as u64,
        );

        queue.submit(Some(encoder.finish()));
    }

    /// Performs a single iteration using an existing command encoder.
    /// Used by the interactive renderer to combine compute + render in one submit.
    pub fn iterate_with_encoder(&self, encoder: &mut wgpu::CommandEncoder) {
        let device = &self.context.device;
        let total = self.total_elements as u32;

        // Phase 1: FFT convolution
        self.convolution.run(
            device,
            encoder,
            &self.channel_buffer,
            &self.kernel_buffer,
            &self.c0,
            self.num_kernels,
            self.num_channels,
            self.total_elements,
            &self.shape,
        );

        // Phase 2: Growth + weighted sum → affinity buffer
        self.growth.run(encoder, total);

        // Phase 3: Diffusion
        self.diffusion.run(encoder, total);

        // Phase 4: Copy new_channel → channel
        encoder.copy_buffer_to_buffer(
            &self.new_channel_buffer,
            0,
            &self.channel_buffer,
            0,
            (self.total_elements * self.num_channels * 4) as u64,
        );
    }

    /// Performs a single iteration with per-phase GPU timing.
    pub fn timed_iterate(&self) -> [f64; 4] {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let total = self.total_elements as u32;
        let mut times = [0.0f64; 4];

        // Phase 1: Convolution
        let start = std::time::Instant::now();
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ml::timed_conv"),
            });
            self.convolution.run(
                device,
                &mut encoder,
                &self.channel_buffer,
                &self.kernel_buffer,
                &self.c0,
                self.num_kernels,
                self.num_channels,
                self.total_elements,
                &self.shape,
            );
            queue.submit(Some(encoder.finish()));
        }
        device.poll(wgpu::Maintain::Wait);
        times[0] = start.elapsed().as_secs_f64() * 1000.0;

        // Phase 2: Growth
        let start = std::time::Instant::now();
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ml::timed_growth"),
            });
            self.growth.run(&mut encoder, total);
            queue.submit(Some(encoder.finish()));
        }
        device.poll(wgpu::Maintain::Wait);
        times[1] = start.elapsed().as_secs_f64() * 1000.0;

        // Phase 3: Diffusion
        let start = std::time::Instant::now();
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ml::timed_diffusion"),
            });
            self.diffusion.run(&mut encoder, total);
            queue.submit(Some(encoder.finish()));
        }
        device.poll(wgpu::Maintain::Wait);
        times[2] = start.elapsed().as_secs_f64() * 1000.0;

        // Phase 4: Buffer copy
        let start = std::time::Instant::now();
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ml::timed_copy"),
            });
            encoder.copy_buffer_to_buffer(
                &self.new_channel_buffer,
                0,
                &self.channel_buffer,
                0,
                (self.total_elements * self.num_channels * 4) as u64,
            );
            queue.submit(Some(encoder.finish()));
        }
        device.poll(wgpu::Maintain::Wait);
        times[3] = start.elapsed().as_secs_f64() * 1000.0;

        times
    }
}
