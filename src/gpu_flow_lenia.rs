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

use crate::wfft::{WgpuContext, WgpuFFT1D};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// WGSL shaders
// ---------------------------------------------------------------------------

const COMPUTE_SHADER: &str = include_str!("shaders/compute.wgsl");

/// Stride between per-kernel growth param slots in the GPU buffer.
/// Must be at least `min_storage_buffer_offset_alignment` (256 on most devices).
const GP_STRIDE: u64 = 256;

// ---------------------------------------------------------------------------
// GpuFlowLenia
// ---------------------------------------------------------------------------

pub struct GpuFlowLenia {
    context: Arc<WgpuContext>,
    shape: Vec<usize>,
    total_elements: usize,
    dt: f32,
    num_channels: usize,
    num_kernels: usize,
    dd: i32,
    sigma: f32,
    basal_metabolic_rate: f32,
    kinetic_cost: f32,

    // Channel mapping
    c0: Vec<u32>,

    // Per-kernel growth params
    kernel_m: Vec<f32>,
    kernel_s: Vec<f32>,
    kernel_h: Vec<f32>,

    // --- Packed GPU buffers ---
    /// All channels packed: [X*Y*C] f32
    channel_buffer: wgpu::Buffer,
    /// Output of reintegration: [X*Y*C] f32
    new_channel_buffer: wgpu::Buffer,
    /// Working buffer for FFT: [X*Y] vec2<f32>
    conv_buffer: wgpu::Buffer,
    /// Saved frequency-domain data for sharing forward FFTs across kernels.
    conv_saved_buffer: wgpu::Buffer,
    /// All kernels packed: [X*Y*k] vec2<f32>
    kernel_buffer: wgpu::Buffer,

    // FFT
    forward_fft_1d: Vec<WgpuFFT1D>,
    inverse_fft_1d: Vec<WgpuFFT1D>,

    // Pipelines
    copy_to_conv_pipeline: wgpu::ComputePipeline,
    complex_mul_pipeline: wgpu::ComputePipeline,
    normalize_growth_pipeline: wgpu::ComputePipeline,
    channel_aggregate_pipeline: wgpu::ComputePipeline,
    sobel_pipeline: wgpu::ComputePipeline,
    sum_channels_pipeline: wgpu::ComputePipeline,
    flow_field_pipeline: wgpu::ComputePipeline,
    reintegration_pipeline: wgpu::ComputePipeline,

    // Bind group layouts (only those needed for per-kernel bind groups in iterate)
    copy_to_conv_bgl: wgpu::BindGroupLayout,
    complex_mul_bgl: wgpu::BindGroupLayout,

    // Cached bind groups (static bindings)
    channel_aggregate_bg: wgpu::BindGroup,
    sobel_u_bg: wgpu::BindGroup,
    sobel_a_bg: wgpu::BindGroup,
    sum_channels_bg: wgpu::BindGroup,
    flow_field_bg: wgpu::BindGroup,
    reintegration_bg: wgpu::BindGroup,

    // Per-kernel cached bind groups
    normalize_growth_bgs: Vec<wgpu::BindGroup>,

    growth_params_buffer: wgpu::Buffer,
    reintegration_params_buffer: wgpu::Buffer,

    // Readback
    readback_buffer: wgpu::Buffer,
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
        basal_metabolic_rate: f32,
        kinetic_cost: f32,
    ) -> Self {
        let device = &context.device;
        let queue = &context.queue;

        assert_eq!(shape.len(), 2, "Flow Lenia requires 2D grids");
        let total_elements: usize = shape.iter().product();
        let buf_size = (total_elements * 4) as u64;
        let conv_buf_size = (total_elements * 8) as u64;

        // Flatten c1
        let mut c1_flat = Vec::new();
        let mut c1_offsets = vec![0u32]; // start with 0
        for c in 0..num_channels {
            c1_flat.extend(c1[c].iter().cloned());
            c1_offsets.push(c1_flat.len() as u32);
        }

        // FFT instances
        let mut forward_fft_1d = Vec::with_capacity(shape.len());
        let mut inverse_fft_1d = Vec::with_capacity(shape.len());
        for &dim in shape {
            forward_fft_1d.push(WgpuFFT1D::new(Arc::clone(&context), dim, false));
            inverse_fft_1d.push(WgpuFFT1D::new(Arc::clone(&context), dim, true));
        }

        // --- Buffer helpers ---
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
        let make_uniform = |label: &str, size: u64| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        // --- Create buffers ---
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

        // Growth params buffer: per-kernel slots (storage, 256-byte aligned)
        let growth_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fl::growth_params"),
            size: (num_kernels as u64) * GP_STRIDE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Pre-write all kernel growth params
        for k in 0..num_kernels {
            let gp: [f32; 3] = [kernel_m[k], kernel_s[k], kernel_h[k]];
            queue.write_buffer(
                &growth_params_buffer,
                (k as u64) * GP_STRIDE,
                bytemuck::cast_slice(&gp),
            );
        }
        let normalize_params_buffer = make_uniform("fl::normalize_params", 4);
        let channel_aggregate_params_buffer = make_uniform("fl::ca_params", 12);
        let sobel_params_u_buffer = make_uniform("fl::sobel_u_params", 12);
        let sobel_params_a_buffer = make_uniform("fl::sobel_a_params", 12);
        let sum_channels_params_buffer = make_uniform("fl::sc_params", 8);
        let flow_field_params_buffer = make_uniform("fl::ff_params", 12);
        let reintegration_params_buffer = make_uniform("fl::ri_params", 36);

        // Mapping buffers
        let c1_flat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fl::c1_flat"),
            size: (c1_flat.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&c1_flat_buffer, 0, bytemuck::cast_slice(&c1_flat));

        let c1_offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fl::c1_offsets"),
            size: (c1_offsets.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&c1_offsets_buffer, 0, bytemuck::cast_slice(&c1_offsets));

        // Readback
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fl::readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Shader module ---
        let sm = |label: &str| -> wgpu::ShaderModule {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(COMPUTE_SHADER.into()),
            })
        };
        let compute_sm = sm("fl::compute");

        // --- Bind group layout helpers ---
        let sro = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let srw = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let unif = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let copy_to_conv_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fl::copy_to_conv bgl"),
            entries: &[sro(4), srw(5)],
        });
        let complex_mul_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fl::complex_mul bgl"),
            entries: &[srw(6), sro(7)],
        });
        let normalize_growth_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fl::normalize_growth bgl"),
                entries: &[srw(8), srw(9), srw(10), unif(11), sro(12)],
            });
        let channel_aggregate_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fl::ca bgl"),
                entries: &[sro(13), srw(14), sro(15), sro(16), unif(17)],
            });
        let sobel_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fl::sobel bgl"),
            entries: &[sro(21), srw(22), srw(23), unif(24)],
        });
        let sum_channels_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fl::sc bgl"),
            entries: &[sro(18), srw(19), unif(20)],
        });
        let flow_field_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fl::ff bgl"),
            entries: &[
                sro(25),
                sro(26),
                sro(27),
                sro(28),
                sro(29),
                srw(30),
                srw(31),
                unif(32),
            ],
        });
        let reintegration_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fl::ri bgl"),
            entries: &[sro(33), sro(34), sro(35), srw(36), unif(37)],
        });

        // --- Pipeline layouts & pipelines ---
        let pl = |label: &str, bgl: &wgpu::BindGroupLayout| -> wgpu::PipelineLayout {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            })
        };
        let cp = |label: &str,
                  layout: &wgpu::PipelineLayout,
                  module: &wgpu::ShaderModule,
                  entry: &str|
         -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module,
                entry_point: entry,
            })
        };

        let copy_to_conv_pipeline = cp(
            "fl::copy_to_conv",
            &pl("fl::copy_to_conv pl", &copy_to_conv_bgl),
            &compute_sm,
            "copy_to_conv_main",
        );
        let complex_mul_pipeline = cp(
            "fl::complex_mul",
            &pl("fl::complex_mul pl", &complex_mul_bgl),
            &compute_sm,
            "complex_mul_main",
        );
        let normalize_growth_pipeline = cp(
            "fl::normalize_growth",
            &pl("fl::normalize_growth pl", &normalize_growth_bgl),
            &compute_sm,
            "normalize_growth_main",
        );
        let channel_aggregate_pipeline = cp(
            "fl::ca",
            &pl("fl::ca pl", &channel_aggregate_bgl),
            &compute_sm,
            "channel_aggregate_main",
        );
        let sobel_pipeline = cp(
            "fl::sobel",
            &pl("fl::sobel pl", &sobel_bgl),
            &compute_sm,
            "sobel_main",
        );
        let sum_channels_pipeline = cp(
            "fl::sc",
            &pl("fl::sc pl", &sum_channels_bgl),
            &compute_sm,
            "sum_channels_main",
        );
        let flow_field_pipeline = cp(
            "fl::ff",
            &pl("fl::ff pl", &flow_field_bgl),
            &compute_sm,
            "flow_field_main",
        );
        let reintegration_pipeline = cp(
            "fl::ri",
            &pl("fl::ri pl", &reintegration_bgl),
            &compute_sm,
            "reintegration_main",
        );

        // --- Write uniform params ---
        let padded_n = forward_fft_1d[0].padded_len() as f32;
        let norm_factor = 1.0 / padded_n.powi(shape.len() as i32);
        queue.write_buffer(
            &normalize_params_buffer,
            0,
            bytemuck::cast_slice(&[norm_factor]),
        );

        let ca: [u32; 3] = [shape[0] as u32, num_kernels as u32, num_channels as u32];
        queue.write_buffer(
            &channel_aggregate_params_buffer,
            0,
            bytemuck::cast_slice(&ca),
        );

        let su: [u32; 3] = [shape[0] as u32, shape[1] as u32, num_channels as u32];
        queue.write_buffer(&sobel_params_u_buffer, 0, bytemuck::cast_slice(&su));

        let sa: [u32; 3] = [shape[0] as u32, shape[1] as u32, 1];
        queue.write_buffer(&sobel_params_a_buffer, 0, bytemuck::cast_slice(&sa));

        let sc: [u32; 2] = [shape[0] as u32, num_channels as u32];
        queue.write_buffer(&sum_channels_params_buffer, 0, bytemuck::cast_slice(&sc));

        // flow_field_params: width(u32), num_channels(u32), num_channels_f32(f32)
        let mut data = Vec::with_capacity(12);
        data.extend_from_slice(&(shape[0] as u32).to_le_bytes());
        data.extend_from_slice(&(num_channels as u32).to_le_bytes());
        data.extend_from_slice(&(num_channels as f32).to_le_bytes());
        queue.write_buffer(&flow_field_params_buffer, 0, &data);

        // reintegration_params: width(u32), height(u32), dd(i32), sigma(f32), dt(f32), num_channels(u32), ma(f32), basal_rate(f32), kinetic_cost(f32)
        let ma = dd as f32 - sigma;
        let mut ri_data = Vec::with_capacity(36);
        ri_data.extend_from_slice(&(shape[0] as u32).to_le_bytes());
        ri_data.extend_from_slice(&(shape[1] as u32).to_le_bytes());
        ri_data.extend_from_slice(&(dd as i32).to_le_bytes());
        ri_data.extend_from_slice(&sigma.to_le_bytes());
        ri_data.extend_from_slice(&dt.to_le_bytes());
        ri_data.extend_from_slice(&(num_channels as u32).to_le_bytes());
        ri_data.extend_from_slice(&ma.to_le_bytes());
        ri_data.extend_from_slice(&basal_metabolic_rate.to_le_bytes());
        ri_data.extend_from_slice(&kinetic_cost.to_le_bytes());
        queue.write_buffer(&reintegration_params_buffer, 0, &ri_data);

        // --- Write uniform params ---
        // --- Cached bind groups ---
        // Per-kernel normalize+growth bind groups
        let mut normalize_growth_bgs = Vec::with_capacity(num_kernels);
        let gp_size = std::num::NonZeroU64::new(12);
        for k in 0..num_kernels {
            let u_offset = (k * total_elements * 4) as u64;
            let u_size = std::num::NonZeroU64::new((total_elements * 4) as u64);

            normalize_growth_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("fl::normalize_growth bg k={k}")),
                layout: &normalize_growth_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: conv_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &u_buffer,
                            offset: u_offset,
                            size: u_size,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &conv_x_buffer,
                            offset: u_offset,
                            size: u_size,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: normalize_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &growth_params_buffer,
                            offset: (k as u64) * GP_STRIDE,
                            size: gp_size,
                        }),
                    },
                ],
            }));
        }

        let channel_aggregate_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::ca bg"),
            layout: &channel_aggregate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: u_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: u_channel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: c1_flat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: c1_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: channel_aggregate_params_buffer.as_entire_binding(),
                },
            ],
        });

        let sobel_u_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::sobel_u bg"),
            layout: &sobel_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 21,
                    resource: u_channel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 22,
                    resource: nabla_u_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 23,
                    resource: nabla_u_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 24,
                    resource: sobel_params_u_buffer.as_entire_binding(),
                },
            ],
        });

        let sobel_a_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::sobel_a bg"),
            layout: &sobel_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 21,
                    resource: sum_a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 22,
                    resource: nabla_a_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 23,
                    resource: nabla_a_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 24,
                    resource: sobel_params_a_buffer.as_entire_binding(),
                },
            ],
        });

        let sum_channels_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::sc bg"),
            layout: &sum_channels_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: channel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: sum_a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 20,
                    resource: sum_channels_params_buffer.as_entire_binding(),
                },
            ],
        });

        let flow_field_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::ff bg"),
            layout: &flow_field_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 25,
                    resource: channel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 26,
                    resource: nabla_u_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 27,
                    resource: nabla_u_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 28,
                    resource: nabla_a_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 29,
                    resource: nabla_a_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 30,
                    resource: flow_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 31,
                    resource: flow_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 32,
                    resource: flow_field_params_buffer.as_entire_binding(),
                },
            ],
        });

        let reintegration_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::ri bg"),
            layout: &reintegration_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 33,
                    resource: channel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 34,
                    resource: flow_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 35,
                    resource: flow_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 36,
                    resource: new_channel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 37,
                    resource: reintegration_params_buffer.as_entire_binding(),
                },
            ],
        });

        GpuFlowLenia {
            context,
            shape: shape.to_vec(),
            total_elements,
            dt,
            num_channels,
            num_kernels,
            dd,
            sigma,
            basal_metabolic_rate,
            kinetic_cost,
            c0: c0.to_vec(),
            kernel_m: kernel_m.to_vec(),
            kernel_s: kernel_s.to_vec(),
            kernel_h: kernel_h.to_vec(),
            channel_buffer,
            new_channel_buffer,
            conv_buffer,
            conv_saved_buffer,
            kernel_buffer,
            forward_fft_1d,
            inverse_fft_1d,
            copy_to_conv_pipeline,
            complex_mul_pipeline,
            normalize_growth_pipeline,
            channel_aggregate_pipeline,
            sobel_pipeline,
            sum_channels_pipeline,
            flow_field_pipeline,
            reintegration_pipeline,
            copy_to_conv_bgl,
            complex_mul_bgl,
            channel_aggregate_bg,
            sobel_u_bg,
            sobel_a_bg,
            sum_channels_bg,
            flow_field_bg,
            reintegration_bg,
            growth_params_buffer,
            normalize_growth_bgs,
            reintegration_params_buffer,
            readback_buffer,
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
        let mut data = Vec::with_capacity(36);
        data.extend_from_slice(&(self.shape[0] as u32).to_le_bytes());
        data.extend_from_slice(&(self.shape[1] as u32).to_le_bytes());
        data.extend_from_slice(&(self.dd as i32).to_le_bytes());
        data.extend_from_slice(&self.sigma.to_le_bytes());
        data.extend_from_slice(&dt.to_le_bytes());
        data.extend_from_slice(&(self.num_channels as u32).to_le_bytes());
        data.extend_from_slice(&ma.to_le_bytes());
        data.extend_from_slice(&self.basal_metabolic_rate.to_le_bytes());
        data.extend_from_slice(&self.kinetic_cost.to_le_bytes());
        self.context
            .queue
            .write_buffer(&self.reintegration_params_buffer, 0, &data);
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
            &self.growth_params_buffer,
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
                &self.growth_params_buffer,
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

        // Need a bigger readback buffer
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

    // --- Main iteration ---

    /// Helper: begin a compute pass, set pipeline + bind group, dispatch.
    fn dispatch_compute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        wg_count: u32,
    ) {
        let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label) });
        p.set_pipeline(pipeline);
        p.set_bind_group(0, bind_group, &[]);
        p.dispatch_workgroups(wg_count, 1, 1);
    }

    /// Performs a single Flow Lenia iteration entirely on the GPU.
    pub fn iterate(&self) {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let total = self.total_elements as u32;
        let wg = (total + 255) / 256;
        let wg_c = ((total * self.num_channels as u32) + 255) / 256;

        let axis0 = self.shape[0] as u32;
        let axis1 = self.shape[1] as u32;
        let lanes0 = self.total_elements / axis0 as usize;
        let lanes1 = self.total_elements / axis1 as usize;

        // Cache FFT bind groups
        let fft_bgs: Vec<_> = self
            .forward_fft_1d
            .iter()
            .chain(self.inverse_fft_1d.iter())
            .map(|fft| fft.ensure_bind_groups(&self.conv_buffer))
            .collect();
        let (fw_bgs, inv_bgs) = fft_bgs.split_at(self.forward_fft_1d.len());

        // ================================================================
        // Phase 1: Per-kernel convolution + growth
        //
        // Group kernels by source channel to share forward FFTs:
        //   FFT each unique channel once, save frequency-domain result,
        //   then restore + multiply + IFFT for each kernel in the group.
        // ================================================================

        // Build source-channel → kernel-indices mapping
        let mut ch_to_kernels: Vec<Vec<usize>> = Vec::new();
        ch_to_kernels.resize_with(self.num_channels, Vec::new);
        for k in 0..self.num_kernels {
            ch_to_kernels[self.c0[k] as usize].push(k);
        }

        let conv_buf_size = (self.total_elements * 8) as u64;

        for kernels in &ch_to_kernels {
            if kernels.is_empty() {
                continue;
            }

            // Create per-kernel bind groups for this group
            let mut copy_bgs = Vec::with_capacity(kernels.len());
            let mut cmul_bgs = Vec::with_capacity(kernels.len());
            for &k in kernels {
                let src_c = self.c0[k] as usize;
                let ch_offset = (src_c * self.total_elements * 4) as u64;
                let k_offset = (k * self.total_elements * 8) as u64;
                let ch_size = std::num::NonZeroU64::new((self.total_elements * 4) as u64);
                let k_size = std::num::NonZeroU64::new((self.total_elements * 8) as u64);

                copy_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("fl::copy_bg k={k}")),
                    layout: &self.copy_to_conv_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &self.channel_buffer,
                                offset: ch_offset,
                                size: ch_size,
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.conv_buffer.as_entire_binding(),
                        },
                    ],
                }));

                cmul_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("fl::cmul_bg k={k}")),
                    layout: &self.complex_mul_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: self.conv_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &self.kernel_buffer,
                                offset: k_offset,
                                size: k_size,
                            }),
                        },
                    ],
                }));
            }

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fl::phase1_group"),
            });

            // Copy channel → conv (first kernel's source channel)
            self.dispatch_compute(
                &mut encoder,
                "copy",
                &self.copy_to_conv_pipeline,
                &copy_bgs[0],
                wg,
            );

            // Forward FFT axis 0
            let (br, fft) = fw_bgs[0];
            self.forward_fft_1d[0].record_transform(&mut encoder, lanes0, axis0, 1, br, fft);

            // Forward FFT axis 1
            let (br, fft) = fw_bgs[1];
            self.forward_fft_1d[1].record_transform(&mut encoder, lanes1, 1, axis0, br, fft);

            // Save frequency-domain data for reuse
            encoder.copy_buffer_to_buffer(
                &self.conv_buffer,
                0,
                &self.conv_saved_buffer,
                0,
                conv_buf_size,
            );

            for (i, k) in kernels.iter().enumerate() {
                // Restore frequency-domain data (skip for first kernel, it's already there)
                if i == 0 {
                    continue;
                };
                encoder.copy_buffer_to_buffer(
                    &self.conv_saved_buffer,
                    0,
                    &self.conv_buffer,
                    0,
                    conv_buf_size,
                );

                // Complex multiply
                self.dispatch_compute(
                    &mut encoder,
                    "cmul",
                    &self.complex_mul_pipeline,
                    &cmul_bgs[i],
                    wg,
                );

                // Inverse FFT axis 1
                let (br, fft) = inv_bgs[1];
                self.inverse_fft_1d[1].record_transform(&mut encoder, lanes1, 1, axis0, br, fft);

                // Inverse FFT axis 0
                let (br, fft) = inv_bgs[0];
                self.inverse_fft_1d[0].record_transform(&mut encoder, lanes0, axis0, 1, br, fft);

                // Normalize + Growth (fused)
                self.dispatch_compute(
                    &mut encoder,
                    "norm_grow",
                    &self.normalize_growth_pipeline,
                    &self.normalize_growth_bgs[*k],
                    wg,
                );
            }

            queue.submit(Some(encoder.finish()));
        }

        // ================================================================
        // Phase 2: Channel aggregation, gradients, flow, reintegration
        // ================================================================
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fl::phase2"),
        });

        // Channel aggregate: u_buffer → u_channel_buffer
        self.dispatch_compute(
            &mut encoder,
            "ca",
            &self.channel_aggregate_pipeline,
            &self.channel_aggregate_bg,
            wg_c,
        );

        // Sum channels: channel_buffer → sum_a_buffer
        self.dispatch_compute(
            &mut encoder,
            "sc",
            &self.sum_channels_pipeline,
            &self.sum_channels_bg,
            wg,
        );

        // Sobel U: u_channel_buffer → nabla_u_x, nabla_u_y
        self.dispatch_compute(
            &mut encoder,
            "sobel_u",
            &self.sobel_pipeline,
            &self.sobel_u_bg,
            wg_c,
        );

        // Sobel A: sum_a_buffer → nabla_a_x, nabla_a_y
        self.dispatch_compute(
            &mut encoder,
            "sobel_a",
            &self.sobel_pipeline,
            &self.sobel_a_bg,
            wg,
        );

        // Flow field
        self.dispatch_compute(
            &mut encoder,
            "flow",
            &self.flow_field_pipeline,
            &self.flow_field_bg,
            wg_c,
        );

        // Reintegration tracking
        self.dispatch_compute(
            &mut encoder,
            "ri",
            &self.reintegration_pipeline,
            &self.reintegration_bg,
            wg_c,
        );

        queue.submit(Some(encoder.finish()));

        // ================================================================
        // Phase 3: Swap new_channel → channel
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
        queue.submit(Some(encoder.finish()));
    }
}
