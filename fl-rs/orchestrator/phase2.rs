// ===========================================================================
// Phase 2: Aggregation, gradients, flow, advection
//
// Sub-phases:
//   2a. Channel aggregation  — sum kernel growths per channel (C0/C1 mapping)
//   2b. Sobel gradients + flow field
//        - sum_channels: collapse all channels into total mass field `a`
//        - sobel_u:      ∇u (gradient of per-channel growth field)
//        - sobel_a:      ∇a (gradient of total mass field)
//        - flow_field:   α-blend ∇u and -∇a into flow vectors
//   2c. Semi-Lagrangian advection — advect mass along flow field
// ===========================================================================

// ===========================================================================
// Phase 2a: Channel aggregation
//
// Sum the growth outputs from multiple kernels into each channel using the
// C0/C1 mapping (sparse connectivity matrix):
//   u_channels[c][pixel] = Σ u_all[k][pixel]  for k in kernels_mapped_to_c
// ===========================================================================

pub struct AggregationPhase {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl AggregationPhase {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        compute_shader: &str,
        shape: &[usize],
        num_kernels: usize,
        num_channels: usize,
        c1: &[Vec<u32>],
        // Shared buffers
        u_buffer: &wgpu::Buffer,
        u_channel_buffer: &wgpu::Buffer,
    ) -> Self {
        // Flatten c1
        let mut c1_flat = Vec::new();
        let mut c1_offsets = vec![0u32];
        for c in 0..num_channels {
            c1_flat.extend(c1[c].iter().cloned());
            c1_offsets.push(c1_flat.len() as u32);
        }

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

        // Uniform params
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fl::ca_params"),
            size: 12,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ca: [u32; 3] = [shape[0] as u32, num_kernels as u32, num_channels as u32];
        queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&ca));

        // Bind group layout
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

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fl::ca bgl"),
            entries: &[sro(13), srw(14), sro(15), sro(16), unif(17)],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fl::ca pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let compute_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fl::compute"),
            source: wgpu::ShaderSource::Wgsl(compute_shader.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fl::ca"),
            layout: Some(&pipeline_layout),
            module: &compute_sm,
            entry_point: "channel_aggregate_main",
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::ca bg"),
            layout: &bgl,
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
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            bind_group,
        }
    }

    pub fn run(&self, encoder: &mut wgpu::CommandEncoder, wg_count: u32) {
        let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ca") });
        p.set_pipeline(&self.pipeline);
        p.set_bind_group(0, &self.bind_group, &[]);
        p.dispatch_workgroups(wg_count, 1, 1);
    }
}

// ===========================================================================
// Phase 2b: Sobel gradients + flow field
//
// Sub-steps:
//   1. sum_channels  — collapse all channels into total mass field `a`
//   2. sobel_u       — ∇u (gradient of per-channel growth field)
//   3. sobel_a       — ∇a (gradient of total mass field)
//   4. flow_field    — α-blend ∇u and -∇a into flow vectors
// ===========================================================================

pub struct GradientFlowPhase {
    // Sobel
    sobel_pipeline: wgpu::ComputePipeline,
    sobel_u_bg: wgpu::BindGroup,
    sobel_a_bg: wgpu::BindGroup,

    // Sum channels
    sum_channels_pipeline: wgpu::ComputePipeline,
    sum_channels_bg: wgpu::BindGroup,

    // Flow field
    flow_field_pipeline: wgpu::ComputePipeline,
    flow_field_bg: wgpu::BindGroup,
}

impl GradientFlowPhase {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        compute_shader: &str,
        shape: &[usize],
        num_channels: usize,
        // Shared buffers (references: owned by GpuFlowLenia or another phase)
        channel_buffer: &wgpu::Buffer,
        u_channel_buffer: &wgpu::Buffer,
        flow_x_buffer: &wgpu::Buffer,
        flow_y_buffer: &wgpu::Buffer,
        // Internal buffers (take ownership: only used within this phase)
        nabla_u_x_buffer: wgpu::Buffer,
        nabla_u_y_buffer: wgpu::Buffer,
        nabla_a_x_buffer: wgpu::Buffer,
        nabla_a_y_buffer: wgpu::Buffer,
        sum_a_buffer: &wgpu::Buffer,
    ) -> Self {
        let make_uniform = |label: &str, size: u64| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let sobel_params_u_buffer = make_uniform("fl::sobel_u_params", 12);
        let sobel_params_a_buffer = make_uniform("fl::sobel_a_params", 12);
        let sum_channels_params_buffer = make_uniform("fl::sc_params", 8);
        let flow_field_params_buffer = make_uniform("fl::ff_params", 12);

        // Write uniform params
        let su: [u32; 3] = [shape[0] as u32, shape[1] as u32, num_channels as u32];
        queue.write_buffer(&sobel_params_u_buffer, 0, bytemuck::cast_slice(&su));

        let sa: [u32; 3] = [shape[0] as u32, shape[1] as u32, 1];
        queue.write_buffer(&sobel_params_a_buffer, 0, bytemuck::cast_slice(&sa));

        let sc: [u32; 2] = [shape[0] as u32, num_channels as u32];
        queue.write_buffer(&sum_channels_params_buffer, 0, bytemuck::cast_slice(&sc));

        let mut data = Vec::with_capacity(12);
        data.extend_from_slice(&(shape[0] as u32).to_le_bytes());
        data.extend_from_slice(&(num_channels as u32).to_le_bytes());
        data.extend_from_slice(&(1.0f32).to_le_bytes());
        queue.write_buffer(&flow_field_params_buffer, 0, &data);

        // Bind group layout helpers
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

        // Bind group layouts
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

        // Pipeline layouts & pipelines
        let pl = |label: &str, bgl: &wgpu::BindGroupLayout| -> wgpu::PipelineLayout {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            })
        };
        let compute_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fl::compute"),
            source: wgpu::ShaderSource::Wgsl(compute_shader.into()),
        });
        let cp =
            |label: &str, layout: &wgpu::PipelineLayout, entry: &str| -> wgpu::ComputePipeline {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    module: &compute_sm,
                    entry_point: entry,
                })
            };

        let sobel_pipeline = cp("fl::sobel", &pl("fl::sobel pl", &sobel_bgl), "sobel_main");
        let sum_channels_pipeline = cp(
            "fl::sc",
            &pl("fl::sc pl", &sum_channels_bgl),
            "sum_channels_main",
        );
        let flow_field_pipeline = cp(
            "fl::ff",
            &pl("fl::ff pl", &flow_field_bgl),
            "flow_field_main",
        );

        // Cached bind groups
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

        Self {
            sobel_pipeline,
            sobel_u_bg,
            sobel_a_bg,
            sum_channels_pipeline,
            sum_channels_bg,
            flow_field_pipeline,
            flow_field_bg,
        }
    }

    pub fn run(&self, encoder: &mut wgpu::CommandEncoder, wg: u32, wg_c: u32) {
        // Step 1: Sum all channels into total mass field `a`
        {
            let mut p =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sc") });
            p.set_pipeline(&self.sum_channels_pipeline);
            p.set_bind_group(0, &self.sum_channels_bg, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }

        // Step 2: Sobel gradient of per-channel growth field (∇u)
        {
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sobel_u"),
            });
            p.set_pipeline(&self.sobel_pipeline);
            p.set_bind_group(0, &self.sobel_u_bg, &[]);
            p.dispatch_workgroups(wg_c, 1, 1);
        }

        // Step 3: Sobel gradient of total mass field (∇a)
        {
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sobel_a"),
            });
            p.set_pipeline(&self.sobel_pipeline);
            p.set_bind_group(0, &self.sobel_a_bg, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }

        // Step 4: Compute flow field from gradients (α-blend ∇u and -∇a)
        {
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flow"),
            });
            p.set_pipeline(&self.flow_field_pipeline);
            p.set_bind_group(0, &self.flow_field_bg, &[]);
            p.dispatch_workgroups(wg_c, 1, 1);
        }
    }
}

// ===========================================================================
// Phase 2c: Semi-Lagrangian advection
//
// Advect mass along the flow field using a semi-Lagrangian scheme with a
// Gaussian spreading kernel. Also applies basal decay and kinetic cost.
// ===========================================================================

pub struct AdvectionPhase {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    pub params_buffer: wgpu::Buffer,
}

impl AdvectionPhase {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        compute_shader: &str,
        shape: &[usize],
        num_channels: usize,
        num_kernels: usize,
        dd: i32,
        sigma: f32,
        dt: f32,
        // Shared buffers
        channel_buffer: &wgpu::Buffer,
        flow_x_buffer: &wgpu::Buffer,
        flow_y_buffer: &wgpu::Buffer,
        new_channel_buffer: &wgpu::Buffer,
        param_buffer: &wgpu::Buffer,
        new_param_buffer: &wgpu::Buffer,
    ) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fl::ri_params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ma = dd as f32 - sigma;
        let mut ri_data = Vec::with_capacity(32);
        ri_data.extend_from_slice(&(shape[0] as u32).to_le_bytes());
        ri_data.extend_from_slice(&(shape[1] as u32).to_le_bytes());
        ri_data.extend_from_slice(&(dd as i32).to_le_bytes());
        ri_data.extend_from_slice(&sigma.to_le_bytes());
        ri_data.extend_from_slice(&dt.to_le_bytes());
        ri_data.extend_from_slice(&(num_channels as u32).to_le_bytes());
        ri_data.extend_from_slice(&(num_kernels as u32).to_le_bytes());
        ri_data.extend_from_slice(&ma.to_le_bytes());
        queue.write_buffer(&params_buffer, 0, &ri_data);

        // Bind group layout
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

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fl::ri bgl"),
            entries: &[
                sro(33),
                sro(34),
                sro(35),
                srw(36),
                unif(37),
                sro(38),
                srw(39),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fl::ri pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let compute_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fl::compute"),
            source: wgpu::ShaderSource::Wgsl(compute_shader.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fl::ri"),
            layout: Some(&pipeline_layout),
            module: &compute_sm,
            entry_point: "reintegration_main",
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::ri bg"),
            layout: &bgl,
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
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 38,
                    resource: param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 39,
                    resource: new_param_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            bind_group,
            params_buffer,
        }
    }

    pub fn run(&self, encoder: &mut wgpu::CommandEncoder, wg_count: u32) {
        let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ri") });
        p.set_pipeline(&self.pipeline);
        p.set_bind_group(0, &self.bind_group, &[]);
        p.dispatch_workgroups(wg_count, 1, 1);
    }
}

// ===========================================================================
// ParamAdvectionPhase: semi-Lagrangian advection of the parameter field
// Uses the same flow field as density, weighted by total mass.
// ===========================================================================

pub struct ParamAdvectionPhase {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl ParamAdvectionPhase {
    pub fn new(
        device: &wgpu::Device,
        compute_shader: &str,
        // Shared buffers
        channel_buffer: &wgpu::Buffer,
        flow_x_buffer: &wgpu::Buffer,
        flow_y_buffer: &wgpu::Buffer,
        sum_a_buffer: &wgpu::Buffer,
        param_buffer: &wgpu::Buffer,
        new_param_buffer: &wgpu::Buffer,
        ri_params_buffer: &wgpu::Buffer,
    ) -> Self {
        // Bind group layout
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

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fl::ri_param bgl"),
            entries: &[
                sro(33),
                sro(34),
                sro(35),
                sro(40),
                unif(37),
                sro(38),
                srw(39),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fl::ri_param pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let compute_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fl::compute"),
            source: wgpu::ShaderSource::Wgsl(compute_shader.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fl::ri_param"),
            layout: Some(&pipeline_layout),
            module: &compute_sm,
            entry_point: "reintegration_params_main",
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::ri_param bg"),
            layout: &bgl,
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
                    binding: 40,
                    resource: sum_a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 37,
                    resource: ri_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 38,
                    resource: param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 39,
                    resource: new_param_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            bind_group,
        }
    }

    pub fn run(&self, encoder: &mut wgpu::CommandEncoder, wg_count: u32) {
        let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ri_param"),
        });
        p.set_pipeline(&self.pipeline);
        p.set_bind_group(0, &self.bind_group, &[]);
        p.dispatch_workgroups(wg_count, 1, 1);
    }
}
