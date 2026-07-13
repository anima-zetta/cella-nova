//! GPU-resident Flow Lenia simulation.
//!
//! Extends standard Lenia with:
//! - Multi-kernel, multi-channel architecture
//! - Flow field computation via Sobel gradients
//! - Reintegration tracking (semi-Lagrangian advection)
//!
//! Reference: "Flow Lenia: Mass conservation for the simulation of
//! continuous cellular automata" (https://arxiv.org/abs/2212.07906)
//!
//! All computation runs on the GPU with zero CPU readback between frames.

mod convolution;
mod phase2;

use crate::wfft::WgpuContext;
use convolution::ConvolutionPhase;
use phase2::{AdvectionPhase, AggregationPhase, GradientFlowPhase, ParamAdvectionPhase};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// WGSL shaders
// ---------------------------------------------------------------------------

const COMPUTE_SHADER: &str = include_str!("../shaders/compute.wgsl");

/// Stride between per-kernel growth param slots in the GPU buffer.
/// Must be at least `min_storage_buffer_offset_alignment` (256 on most devices).
const GP_STRIDE: u64 = 256;

// ===========================================================================
// GpuFlowLenia — top-level simulation orchestrator
// ===========================================================================

pub struct GpuFlowLenia {
    context: Arc<WgpuContext>,
    shape: Vec<usize>,
    total_elements: usize,
    dt: f32,
    num_channels: usize,
    num_kernels: usize,
    dd: i32,
    sigma: f32,

    // Channel mapping
    c0: Vec<u32>,

    // Per-kernel growth params
    kernel_m: Vec<f32>,
    kernel_s: Vec<f32>,
    kernel_h: Vec<f32>,

    // --- Packed GPU buffers (shared across phases) ---
    /// All channels packed: [X*Y*C] f32
    channel_buffer: wgpu::Buffer,
    /// Output of reintegration: [X*Y*C] f32
    new_channel_buffer: wgpu::Buffer,
    /// All kernels packed: [X*Y*k] vec2<f32>
    kernel_buffer: wgpu::Buffer,
    /// Parameter field: [X*Y*K] f32
    param_buffer: wgpu::Buffer,
    /// Output of parameter advection: [X*Y*K] f32
    new_param_buffer: wgpu::Buffer,

    // Readback
    readback_buffer: wgpu::Buffer,

    // --- Phase sub-structs ---
    convolution: ConvolutionPhase,
    aggregation: AggregationPhase,
    gradient_flow: GradientFlowPhase,
    advection: AdvectionPhase,
    param_advection: ParamAdvectionPhase,
}

impl GpuFlowLenia {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: Arc<WgpuContext>,
        shape: &[usize],
        num_channels: usize,
        num_kernels: usize,
        c0: &[u32],
        c1: &[Vec<u32>],
        kernel_m: &[f32],
        kernel_s: &[f32],
        kernel_h: &[f32],
        dt: f32,
        dd: i32,
        sigma: f32,
    ) -> Self {
        let device = &context.device;
        let queue = &context.queue;

        assert_eq!(shape.len(), 2, "Flow Lenia requires 2D grids");
        let total_elements: usize = shape.iter().product();
        let buf_size = (total_elements * 4) as u64;
        let conv_buf_size = (total_elements * 8) as u64;

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
        let channel_buffer = make_storage("fl::channel", ch_size);
        let new_channel_buffer = make_storage("fl::new_channel", ch_size);
        let conv_buffer = make_storage("fl::conv", conv_buf_size);
        let conv_saved_buffer = make_storage("fl::conv_saved", conv_buf_size);

        let k_size = (total_elements * num_kernels * 8) as u64;
        let kernel_buffer = make_storage("fl::kernel", k_size);

        let u_size = (total_elements * num_kernels * 4) as u64;
        let u_buffer = make_storage("fl::u", u_size);
        let conv_x_buffer = make_storage("fl::conv_x", u_size);

        let uc_size = (total_elements * num_channels * 4) as u64;
        let u_channel_buffer = make_storage("fl::u_channel", uc_size);
        let nabla_u_x_buffer = make_storage("fl::nabla_u_x", uc_size);
        let nabla_u_y_buffer = make_storage("fl::nabla_u_y", uc_size);

        let nabla_a_x_buffer = make_storage("fl::nabla_a_x", buf_size);
        let nabla_a_y_buffer = make_storage("fl::nabla_a_y", buf_size);
        let sum_a_buffer = make_storage("fl::sum_a", buf_size);

        let flow_x_buffer = make_storage("fl::flow_x", uc_size);
        let flow_y_buffer = make_storage("fl::flow_y", uc_size);

        // Parameter field buffers: [X*Y*K] f32
        let param_size = (total_elements * num_kernels * 4) as u64;
        let param_buffer = make_storage("fl::param", param_size);
        let new_param_buffer = make_storage("fl::new_param", param_size);

        // Initialize param buffer to 1.0 (no effect on growth initially)
        let ones: Vec<f32> = vec![1.0f32; total_elements * num_kernels];
        queue.write_buffer(&param_buffer, 0, bytemuck::cast_slice(&ones));

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fl::readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Initialize phase sub-structs ---
        let convolution = ConvolutionPhase::new(
            device,
            queue,
            shape,
            total_elements,
            num_kernels,
            kernel_m,
            kernel_s,
            kernel_h,
            conv_buffer, // move ownership
            conv_saved_buffer,
            &u_buffer,
            conv_x_buffer,
            &param_buffer,
        );

        let aggregation = AggregationPhase::new(
            device,
            queue,
            shape,
            num_kernels,
            num_channels,
            c1,
            &u_buffer,
            &u_channel_buffer,
        );

        let gradient_flow = GradientFlowPhase::new(
            device,
            queue,
            shape,
            num_channels,
            &channel_buffer,
            &u_channel_buffer,
            &flow_x_buffer,
            &flow_y_buffer,
            nabla_u_x_buffer, // move ownership
            nabla_u_y_buffer,
            nabla_a_x_buffer,
            nabla_a_y_buffer,
            sum_a_buffer,
        );

        let advection = AdvectionPhase::new(
            device,
            queue,
            shape,
            num_channels,
            num_kernels,
            dd,
            sigma,
            dt,
            &channel_buffer,
            &flow_x_buffer,
            &flow_y_buffer,
            &new_channel_buffer,
            &param_buffer,
            &new_param_buffer,
        );

        let param_advection = ParamAdvectionPhase::new(
            device,
            &channel_buffer,
            &flow_x_buffer,
            &flow_y_buffer,
            &param_buffer,
            &new_param_buffer,
            &advection.params_buffer,
        );

        GpuFlowLenia {
            context,
            shape: shape.to_vec(),
            total_elements,
            dt,
            num_channels,
            num_kernels,
            dd,
            sigma,
            c0: c0.to_vec(),
            kernel_m: kernel_m.to_vec(),
            kernel_s: kernel_s.to_vec(),
            kernel_h: kernel_h.to_vec(),
            channel_buffer,
            new_channel_buffer,
            kernel_buffer,
            param_buffer,
            new_param_buffer,
            readback_buffer,
            convolution,
            aggregation,
            gradient_flow,
            advection,
            param_advection,
        }
    }

    // --- Accessors ---

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn channel_buffer(&self) -> &wgpu::Buffer {
        &self.channel_buffer
    }

    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    pub fn num_kernels(&self) -> usize {
        self.num_kernels
    }

    pub fn set_dt(&mut self, dt: f32) {
        self.dt = dt;
        let ma = self.dd as f32 - self.sigma;
        let mut data = Vec::with_capacity(32);
        data.extend_from_slice(&(self.shape[0] as u32).to_le_bytes());
        data.extend_from_slice(&(self.shape[1] as u32).to_le_bytes());
        data.extend_from_slice(&(self.dd as i32).to_le_bytes());
        data.extend_from_slice(&self.sigma.to_le_bytes());
        data.extend_from_slice(&dt.to_le_bytes());
        data.extend_from_slice(&(self.num_channels as u32).to_le_bytes());
        data.extend_from_slice(&(self.num_kernels as u32).to_le_bytes());
        data.extend_from_slice(&ma.to_le_bytes());
        self.context
            .queue
            .write_buffer(&self.advection.params_buffer, 0, &data);
    }

    /// Uploads a pre-FFT'd kernel at the given index.
    pub fn set_kernel(&self, kernel_fft: &[num_complex::Complex32], kernel_idx: usize) {
        assert_eq!(kernel_fft.len(), self.total_elements);
        let offset = (kernel_idx * self.total_elements * 8) as u64;
        let data: Vec<[f32; 2]> = kernel_fft.iter().map(|c| [c.re, c.im]).collect();
        self.context
            .queue
            .write_buffer(&self.kernel_buffer, offset, bytemuck::cast_slice(&data));
    }

    /// Uploads channel data at the given index.
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

    /// Update growth parameters for a single kernel.
    pub fn set_growth_param(&mut self, k: usize, m: f32, s: f32, h: f32) {
        self.kernel_m[k] = m;
        self.kernel_s[k] = s;
        self.kernel_h[k] = h;
        let gp: [f32; 3] = [m, s, h];
        self.context.queue.write_buffer(
            self.convolution.growth_params_buffer(),
            (k as u64) * GP_STRIDE,
            bytemuck::cast_slice(&gp),
        );
    }

    /// Update all growth parameters at once.
    pub fn set_all_growth_params(&mut self, m: &[f32], s: &[f32], h: &[f32]) {
        self.kernel_m.copy_from_slice(m);
        self.kernel_s.copy_from_slice(s);
        self.kernel_h.copy_from_slice(h);
        for k in 0..self.kernel_m.len() {
            let gp: [f32; 3] = [m[k], s[k], h[k]];
            self.context.queue.write_buffer(
                self.convolution.growth_params_buffer(),
                (k as u64) * GP_STRIDE,
                bytemuck::cast_slice(&gp),
            );
        }
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
            label: Some("fl::readback_all"),
            size: total_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fl::download_all"),
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

    /// Download kernel FFT weights for a given kernel.
    pub fn download_kernel(&self, k: usize) -> Vec<num_complex::Complex32> {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let total = self.total_elements;
        let total_bytes = (total * 8) as u64;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fl::readback_k"),
            size: total_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fl::download_k"),
        });
        let offset = (k * total * 8) as u64;
        encoder.copy_buffer_to_buffer(&self.kernel_buffer, offset, &readback, 0, total_bytes);
        queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let view = slice.get_mapped_range();
        let raw: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        readback.unmap();
        raw.chunks(2)
            .map(|c| num_complex::Complex32::new(c[0], c[1]))
            .collect()
    }

    /// Re-initialize all channels from flat f64 data.
    pub fn reinit_channels(&self, data: &[Vec<f64>]) {
        for (c, ch_data) in data.iter().enumerate() {
            self.upload_channel(ch_data, c);
        }
    }

    /// Get total elements (width * height).
    pub fn total_elements(&self) -> usize {
        self.total_elements
    }

    /// Get kernel_m slice.
    pub fn kernel_m_slice(&self) -> &[f32] {
        &self.kernel_m
    }

    /// Get kernel_s slice.
    pub fn kernel_s_slice(&self) -> &[f32] {
        &self.kernel_s
    }

    /// Get kernel_h slice.
    pub fn kernel_h_slice(&self) -> &[f32] {
        &self.kernel_h
    }

    pub fn download_channel(&self, channel: usize) -> Vec<f32> {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let buf_size = (self.total_elements * 4) as u64;
        let offset = (channel * self.total_elements * 4) as u64;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fl::download"),
        });
        encoder.copy_buffer_to_buffer(
            &self.channel_buffer,
            offset,
            &self.readback_buffer,
            0,
            buf_size,
        );
        queue.submit(Some(encoder.finish()));

        let slice = self.readback_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let view = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        self.readback_buffer.unmap();
        result
    }

    // =======================================================================
    // Main iteration
    // =======================================================================

    /// Performs a single Flow Lenia iteration entirely on the GPU.
    pub fn iterate(&self) {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let total = self.total_elements as u32;
        let wg = (total + 255) / 256;
        let wg_c = ((total * self.num_channels as u32) + 255) / 256;
        let wg_k = ((total * self.num_kernels as u32) + 255) / 256;

        // ================================================================
        // Phase 1: Per-kernel convolution + growth
        //
        // Group kernels by source channel to share forward FFTs:
        //   FFT each unique channel once, save frequency-domain result,
        //   then restore + multiply + IFFT for each kernel in the group.
        // ================================================================

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fl::phase1"),
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

        // ================================================================
        // Phase 2: Aggregation, gradients, flow, advection
        // ================================================================

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fl::phase2"),
        });

        // Phase 2a: Channel aggregation (kernel growths → per-channel sums)
        self.aggregation.run(&mut encoder, wg_c);

        // Phase 2b: Sobel gradients + flow field
        self.gradient_flow.run(&mut encoder, wg, wg_c);

        // Phase 2c: Semi-Lagrangian advection (density)
        self.advection.run(&mut encoder, wg_c);

        // Phase 2d: Semi-Lagrangian advection (parameter field)
        self.param_advection.run(&mut encoder, wg_k);

        queue.submit(Some(encoder.finish()));

        // ================================================================
        // Phase 3: Swap new → current for both density and params
        // ================================================================

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fl::swap"),
        });
        encoder.copy_buffer_to_buffer(
            &self.new_channel_buffer,
            0,
            &self.channel_buffer,
            0,
            (self.total_elements * self.num_channels * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.new_param_buffer,
            0,
            &self.param_buffer,
            0,
            (self.total_elements * self.num_kernels * 4) as u64,
        );
        queue.submit(Some(encoder.finish()));
    }
}
