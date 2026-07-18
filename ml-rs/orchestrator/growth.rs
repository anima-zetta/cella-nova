//! Growth phase for DiffusionLenia.
//!
//! Applies the bump growth function to each convolution result,
//! performs weighted sum over input channels per output channel,
//! and writes the result (affinity) to the affinity buffer.
//! The Euler step is replaced by the diffusion phase.

// ===========================================================================
// GrowthPhase
// ===========================================================================

pub struct GrowthPhase {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,

    // Parameter buffers
    growth_params_buffer: wgpu::Buffer,
    weights_buffer: wgpu::Buffer,
    c1_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
}

impl GrowthPhase {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        compute_shader: &str,
        shape: &[usize],
        num_kernels: usize,
        num_channels: usize,
        c1: &[u32],
        mu: &[f32],
        sigma: &[f32],
        weights: &[f32],
        dt: f32,
        conv_buffer: &wgpu::Buffer,
        affinity_buffer: &wgpu::Buffer,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ml::growth"),
            source: wgpu::ShaderSource::Wgsl(compute_shader.into()),
        });

        let width = shape[0] as u32;

        // --- Growth params buffer: [K] vec2<f32> (mu, sigma) ---
        let mut gp_data: Vec<[f32; 2]> = Vec::with_capacity(num_kernels);
        for k in 0..num_kernels {
            gp_data.push([mu[k], sigma[k]]);
        }
        let growth_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ml::growth_params"),
            size: (num_kernels * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&growth_params_buffer, 0, bytemuck::cast_slice(&gp_data));

        // --- Weights buffer: [K] f32 ---
        let weights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ml::weights"),
            size: (num_kernels * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&weights_buffer, 0, bytemuck::cast_slice(weights));

        // --- C1 mapping buffer: [K] u32 ---
        let c1_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ml::c1"),
            size: (num_kernels * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&c1_buffer, 0, bytemuck::cast_slice(c1));

        // --- Params uniform buffer ---
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
        struct MclParamsRaw {
            width: u32,
            num_kernels: u32,
            num_channels: u32,
            dt: f32,
            norm_factor: f32,
        }

        let norm_factor = 1.0 / (width as f32 * width as f32);

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ml::mcl_params"),
            size: std::mem::size_of::<MclParamsRaw>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &params_buffer,
            0,
            bytemuck::bytes_of(&MclParamsRaw {
                width,
                num_kernels: num_kernels as u32,
                num_channels: num_channels as u32,
                dt,
                norm_factor,
            }),
        );

        // --- Bind group layout ---
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ml::growth_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ml::growth"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ml::growth"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "mcl_growth_main",
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ml::growth_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: conv_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: affinity_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &growth_params_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &weights_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &c1_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        Self {
            pipeline,
            bind_group,
            growth_params_buffer,
            weights_buffer,
            c1_buffer,
            params_buffer,
        }
    }

    /// Run the growth phase.
    /// Dispatches one workgroup per pixel.
    pub fn run(&self, encoder: &mut wgpu::CommandEncoder, total_elements: u32) {
        let wg = (total_elements + 255) / 256;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ml::growth"),
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(wg, 1, 1);
    }
}
