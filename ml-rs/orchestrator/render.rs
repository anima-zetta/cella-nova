//! Render phase for MaceLenia.
//!
//! Maps multi-channel simulation state to packed RGB on the GPU,
//! avoiding the expensive CPU readback + color-mapping loop.

// ===========================================================================
// RenderPhase
// ===========================================================================

pub struct RenderPhase {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
}

impl RenderPhase {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        compute_shader: &str,
        shape: &[usize],
        num_channels: usize,
        channel_buffer: &wgpu::Buffer,
        render_buffer: &wgpu::Buffer,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ml::render"),
            source: wgpu::ShaderSource::Wgsl(compute_shader.into()),
        });

        let width = shape[0] as u32;

        // --- Params uniform buffer ---
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
        struct RenderParamsRaw {
            width: u32,
            num_channels: u32,
        }

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ml::render_params"),
            size: std::mem::size_of::<RenderParamsRaw>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &params_buffer,
            0,
            bytemuck::bytes_of(&RenderParamsRaw {
                width,
                num_channels: num_channels as u32,
            }),
        );

        // --- Bind group layout (bindings 20-22) ---
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ml::render_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 20,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 21,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 22,
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
            label: Some("ml::render"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ml::render"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "render_main",
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ml::render_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 20,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: channel_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 21,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: render_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 22,
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
            params_buffer,
        }
    }

    /// Run the render phase. Dispatches one workgroup per 256 pixels.
    pub fn run(&self, encoder: &mut wgpu::CommandEncoder, total_elements: u32) {
        let wg = (total_elements + 255) / 256;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ml::render"),
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(wg, 1, 1);
    }
}
