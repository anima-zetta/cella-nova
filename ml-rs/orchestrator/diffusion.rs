//! Diffusion phase for DiffusionLenia.
//!
//! Implements the mass-conserving diffusion step:
//! 1. Pass 1: Compute aff_exp = exp(temp * affinity) and Z = 3x3 sum of aff_exp
//! 2. Pass 2: new_state[p] = aff_exp[p] * sum over 3x3 of (state[n] / Z[n])

// ===========================================================================
// DiffusionPhase
// ===========================================================================

pub struct DiffusionPhase {
    // Pass 1: compute aff_exp and Z
    pass1_pipeline: wgpu::ComputePipeline,
    pass1_bg: wgpu::BindGroup,
    // Pass 2: compute new state
    pass2_pipeline: wgpu::ComputePipeline,
    pass2_bg: wgpu::BindGroup,

    // Parameter buffer
    params_buffer: wgpu::Buffer,
}

impl DiffusionPhase {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        compute_shader: &str,
        shape: &[usize],
        num_channels: usize,
        temp: f32,
        affinity_buffer: &wgpu::Buffer,
        z_buffer: &wgpu::Buffer,
        channel_buffer: &wgpu::Buffer,
        new_channel_buffer: &wgpu::Buffer,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ml::diffusion"),
            source: wgpu::ShaderSource::Wgsl(compute_shader.into()),
        });

        let width = shape[0] as u32;

        // --- Params uniform buffer ---
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
        struct DiffusionParamsRaw {
            width: u32,
            num_channels: u32,
            temp: f32,
        }

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ml::diffusion_params"),
            size: std::mem::size_of::<DiffusionParamsRaw>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &params_buffer,
            0,
            bytemuck::bytes_of(&DiffusionParamsRaw {
                width,
                num_channels: num_channels as u32,
                temp,
            }),
        );

        // --- Pass 1 bind group layout (bindings 15-17) ---
        let pass1_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ml::diffusion_pass1_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 17,
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

        let pass1_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ml::diffusion_pass1"),
                bind_group_layouts: &[&pass1_bgl],
                push_constant_ranges: &[],
            });

        let pass1_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ml::diffusion_pass1"),
            layout: Some(&pass1_pipeline_layout),
            module: &shader,
            entry_point: "diffusion_pass1_main",
        });

        // --- Pass 1 bind group ---
        let pass1_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ml::diffusion_pass1_bg"),
            layout: &pass1_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: affinity_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: z_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // --- Pass 2 bind group layout (bindings 15-19) ---
        let pass2_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ml::diffusion_pass2_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 17,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 18,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 19,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pass2_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ml::diffusion_pass2"),
                bind_group_layouts: &[&pass2_bgl],
                push_constant_ranges: &[],
            });

        let pass2_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ml::diffusion_pass2"),
            layout: Some(&pass2_pipeline_layout),
            module: &shader,
            entry_point: "diffusion_pass2_main",
        });

        // --- Pass 2 bind group ---
        let pass2_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ml::diffusion_pass2_bg"),
            layout: &pass2_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: affinity_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: z_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: channel_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: new_channel_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        Self {
            pass1_pipeline,
            pass1_bg,
            pass2_pipeline,
            pass2_bg,
            params_buffer,
        }
    }

    /// Run the diffusion phase (pass 1 + pass 2).
    pub fn run(&self, encoder: &mut wgpu::CommandEncoder, total_elements: u32) {
        let wg = (total_elements + 255) / 256;

        // Pass 1: compute aff_exp and Z
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ml::diffusion_pass1"),
            });
            cpass.set_pipeline(&self.pass1_pipeline);
            cpass.set_bind_group(0, &self.pass1_bg, &[]);
            cpass.dispatch_workgroups(wg, 1, 1);
        }

        // Pass 2: compute new state
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ml::diffusion_pass2"),
            });
            cpass.set_pipeline(&self.pass2_pipeline);
            cpass.set_bind_group(0, &self.pass2_bg, &[]);
            cpass.dispatch_workgroups(wg, 1, 1);
        }
    }
}
