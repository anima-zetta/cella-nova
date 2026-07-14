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

/// Select the compute shader source based on grid size.
fn select_compute_shader(grid_size: usize) -> &'static str {
    match grid_size {
        64 => include_str!("../shaders/compute_64.wgsl"),
        128 => include_str!("../shaders/compute_128.wgsl"),
        256 => include_str!("../shaders/compute_256.wgsl"),
        512 => include_str!("../shaders/compute_512.wgsl"),
        _ => panic!(
            "Unsupported grid size: {}. Supported sizes: 64, 128, 256, 512",
            grid_size
        ),
    }
}

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
    num_channels: usize,
    num_kernels: usize,

    // Channel mapping
    c0: Vec<u32>,

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

    // --- Phase sub-structs ---
    convolution: ConvolutionPhase,
    aggregation: AggregationPhase,
    gradient_flow: GradientFlowPhase,
    advection: AdvectionPhase,
    param_advection: ParamAdvectionPhase,

    // Total mass buffer (sum of all channels), shared between gradient_flow and param_advection
    #[allow(dead_code)]
    sum_a_buffer: wgpu::Buffer,
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
            kernel_m,
            kernel_s,
            kernel_h,
            c0,
            conv_buffer, // move ownership
            conv_saved_buffer,
            &channel_buffer,
            &kernel_buffer,
            &u_buffer,
            &param_buffer,
        );

        let aggregation = AggregationPhase::new(
            device,
            queue,
            compute_shader,
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
            compute_shader,
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
            &sum_a_buffer,
        );

        let advection = AdvectionPhase::new(
            device,
            queue,
            compute_shader,
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
            compute_shader,
            &channel_buffer,
            &flow_x_buffer,
            &flow_y_buffer,
            &sum_a_buffer,
            &param_buffer,
            &new_param_buffer,
            &advection.params_buffer,
        );

        GpuFlowLenia {
            context,
            shape: shape.to_vec(),
            total_elements,
            num_channels,
            num_kernels,
            c0: c0.to_vec(),
            channel_buffer,
            new_channel_buffer,
            kernel_buffer,
            param_buffer,
            new_param_buffer,
            sum_a_buffer,
            convolution,
            aggregation,
            gradient_flow,
            advection,
            param_advection,
        }
    }

    // --- Accessors ---

    pub fn channel_buffer(&self) -> &wgpu::Buffer {
        &self.channel_buffer
    }

    /// Run N iterations (forward pass).
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

    #[allow(dead_code)]
    /// Run N iterations (forward pass).
    pub fn run_steps(&self, n: usize) {
        for _ in 0..n {
            self.iterate();
        }
    }

    #[allow(dead_code)]
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

    /// Download all channel data concatenated.
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
        // Single combined encoder: all phases in one submit.
        // The GPU handles buffer dependencies between dispatches internally.
        // ================================================================

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fl::iterate"),
        });

        // Phase 1: Per-kernel convolution + growth
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

        // Phase 2a: Channel aggregation (kernel growths → per-channel sums)
        self.aggregation.run(&mut encoder, wg_c);

        // Phase 2b: Sobel gradients + flow field
        self.gradient_flow.run(&mut encoder, wg, wg_c);

        // Phase 2c: Semi-Lagrangian advection (density)
        self.advection.run(&mut encoder, wg_c);

        // Phase 3: Swap new → current for density
        encoder.copy_buffer_to_buffer(
            &self.new_channel_buffer,
            0,
            &self.channel_buffer,
            0,
            (self.total_elements * self.num_channels * 4) as u64,
        );

        queue.submit(Some(encoder.finish()));
    }
}
