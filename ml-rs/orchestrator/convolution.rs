//! FFT-based convolution phase for MaceLenia.
//!
//! Performs bulk FFT of all channels, then per-kernel complex multiply + IFFT.
//!
//! Algorithm:
//! 1. Copy ALL channels to conv buffer (real f32 → complex vec2)
//! 2. FFT ALL channels (row + col passes, dispatched for all channels at once)
//! 3. For each kernel: copy FFT'd input channel to kernel's slot
//! 4. Complex multiply ALL kernels with their FFT'd kernels
//! 5. IFFT ALL kernels (col + row passes, dispatched for all kernels at once)

use std::num::NonZeroU64;

// ===========================================================================
// Twiddle factor generation
// ===========================================================================

/// Generate twiddle factors for the Stockham FFT.
fn generate_twiddles(size: usize) -> Vec<[f32; 2]> {
    let num_stages = (size as f64).log2().ceil() as usize;
    let mut twiddles = Vec::new();
    for stage in 0..num_stages {
        let r = 1 << stage;
        for k in 0..r {
            let angle = -2.0 * std::f32::consts::PI * (k as f32) / (2 * r) as f32;
            twiddles.push([angle.cos(), angle.sin()]);
        }
    }
    twiddles
}

/// Generate bit-reversal permutation for the Stockham FFT.
fn generate_bitrev(size: usize) -> Vec<u32> {
    let bits = (size as f64).log2().ceil() as u32;
    let mut lut = vec![0u32; size];
    for i in 0..size as u32 {
        lut[i as usize] = i.reverse_bits() >> (32 - bits);
    }
    lut
}

// ===========================================================================
// ConvolutionPhase
// ===========================================================================

pub struct ConvolutionPhase {
    // FFT pipelines
    fft_row_pipeline: wgpu::ComputePipeline,
    fft_col_pipeline: wgpu::ComputePipeline,
    // IFFT pipelines (stages in reverse order)
    ifft_row_pipeline: wgpu::ComputePipeline,
    ifft_col_pipeline: wgpu::ComputePipeline,
    fft_bgl: wgpu::BindGroupLayout,

    // Copy to conv
    copy_to_conv_pipeline: wgpu::ComputePipeline,
    copy_to_conv_bgl: wgpu::BindGroupLayout,

    // Complex multiply
    complex_mul_pipeline: wgpu::ComputePipeline,
    complex_mul_bgl: wgpu::BindGroupLayout,

    // Fused cmul + IFFT column
    fused_cmul_ifft_pipeline: wgpu::ComputePipeline,
    fused_cmul_ifft_bgl: wgpu::BindGroupLayout,

    // Shared resources
    twiddle_buffer: wgpu::Buffer,
    bitrev_buffer: wgpu::Buffer,
    fft_params_buffer: wgpu::Buffer,
    inv_fft_params_buffer: wgpu::Buffer,

    // Conv buffer (shared with orchestrator)
    pub conv_buffer: wgpu::Buffer,
    /// Temporary buffer for same-buffer copies
    temp_buffer: wgpu::Buffer,
}

impl ConvolutionPhase {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        compute_shader: &str,
        shape: &[usize],
        total_elements: usize,
        _num_kernels: usize,
        _num_channels: usize,
        _c0: &[u32],
        conv_buffer: wgpu::Buffer,
        _channel_buffer: &wgpu::Buffer,
        _kernel_buffer: &wgpu::Buffer,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ml::compute"),
            source: wgpu::ShaderSource::Wgsl(compute_shader.into()),
        });

        let width = shape[0] as u32;

        // --- Twiddle factors ---
        let twiddles = generate_twiddles(shape[0]);
        let twiddle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ml::twiddles"),
            size: (twiddles.len() * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&twiddle_buffer, 0, bytemuck::cast_slice(&twiddles));

        // --- Bit-reversal LUT ---
        let bitrev = generate_bitrev(shape[0]);
        let bitrev_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ml::bitrev"),
            size: (bitrev.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&bitrev_buffer, 0, bytemuck::cast_slice(&bitrev));

        // --- FFT params buffers ---
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
        struct FftParamsRaw {
            width: u32,
            inverse: u32,
        }

        let fft_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ml::fft_params"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &fft_params_buffer,
            0,
            bytemuck::bytes_of(&FftParamsRaw { width, inverse: 0 }),
        );

        let inv_fft_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ml::inv_fft_params"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &inv_fft_params_buffer,
            0,
            bytemuck::bytes_of(&FftParamsRaw { width, inverse: 1 }),
        );

        // --- FFT bind group layout ---
        let fft_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ml::fft_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 41,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let fft_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ml::fft"),
            bind_group_layouts: &[&fft_bgl],
            push_constant_ranges: &[],
        });

        let fft_row_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ml::fft_row"),
            layout: Some(&fft_pipeline_layout),
            module: &shader,
            entry_point: "fft_row_main",
        });

        let fft_col_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ml::fft_col"),
            layout: Some(&fft_pipeline_layout),
            module: &shader,
            entry_point: "fft_col_main",
        });

        // --- IFFT pipelines (stages in reverse order) ---
        let ifft_row_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ml::ifft_row"),
            layout: Some(&fft_pipeline_layout),
            module: &shader,
            entry_point: "ifft_row_main",
        });

        let ifft_col_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ml::ifft_col"),
            layout: Some(&fft_pipeline_layout),
            module: &shader,
            entry_point: "ifft_col_main",
        });

        // --- Copy to conv pipeline ---
        let copy_to_conv_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ml::copy_to_conv_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        let copy_to_conv_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ml::copy_to_conv"),
                bind_group_layouts: &[&copy_to_conv_bgl],
                push_constant_ranges: &[],
            });

        let copy_to_conv_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ml::copy_to_conv"),
                layout: Some(&copy_to_conv_pipeline_layout),
                module: &shader,
                entry_point: "copy_to_conv_main",
            });

        // --- Complex multiply pipeline ---
        let complex_mul_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ml::cmul_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let complex_mul_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ml::cmul"),
                bind_group_layouts: &[&complex_mul_bgl],
                push_constant_ranges: &[],
            });

        let complex_mul_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ml::cmul"),
                layout: Some(&complex_mul_pipeline_layout),
                module: &shader,
                entry_point: "complex_mul_main",
            });

        // --- Fused cmul + IFFT column pipeline ---
        let fused_cmul_ifft_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ml::fused_cmul_ifft_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 41,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let fused_cmul_ifft_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ml::fused_cmul_ifft"),
                bind_group_layouts: &[&fused_cmul_ifft_bgl],
                push_constant_ranges: &[],
            });

        let fused_cmul_ifft_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ml::fused_cmul_ifft"),
                layout: Some(&fused_cmul_ifft_pipeline_layout),
                module: &shader,
                entry_point: "fused_cmul_ifft_col_main",
            });

        // --- Temporary buffer for same-buffer copies ---
        let ch_complex_size = (total_elements as u64) * 8; // vec2<f32> per element
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ml::temp"),
            size: ch_complex_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            fft_row_pipeline,
            fft_col_pipeline,
            ifft_row_pipeline,
            ifft_col_pipeline,
            fft_bgl,
            copy_to_conv_pipeline,
            copy_to_conv_bgl,
            complex_mul_pipeline,
            complex_mul_bgl,
            fused_cmul_ifft_pipeline,
            fused_cmul_ifft_bgl,
            twiddle_buffer,
            bitrev_buffer,
            fft_params_buffer,
            inv_fft_params_buffer,
            conv_buffer,
            temp_buffer,
        }
    }

    /// Create a bind group for the FFT pipeline with a specific buffer offset.
    fn make_fft_bg(
        &self,
        device: &wgpu::Device,
        data_buffer: &wgpu::Buffer,
        offset: u64,
        size: u64,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ml::fft_bg"),
            layout: &self.fft_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: data_buffer,
                        offset,
                        size: NonZeroU64::new(size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.twiddle_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: params_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 41,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.bitrev_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        })
    }

    /// Create a bind group for copy_to_conv with specific buffer offsets.
    fn make_copy_bg(
        &self,
        device: &wgpu::Device,
        channel_buffer: &wgpu::Buffer,
        ch_offset: u64,
        conv_buffer: &wgpu::Buffer,
        conv_offset: u64,
        copy_size: u64,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ml::copy_bg"),
            layout: &self.copy_to_conv_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: channel_buffer,
                        offset: ch_offset,
                        size: NonZeroU64::new(copy_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: conv_buffer,
                        offset: conv_offset,
                        size: NonZeroU64::new(copy_size * 2), // f32 -> vec2<f32> doubles size
                    }),
                },
            ],
        })
    }

    /// Create a bind group for complex multiply with specific buffer offsets.
    fn make_cmul_bg(
        &self,
        device: &wgpu::Device,
        conv_buffer: &wgpu::Buffer,
        conv_offset: u64,
        kernel_buffer: &wgpu::Buffer,
        kernel_offset: u64,
        size: u64,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ml::cmul_bg"),
            layout: &self.complex_mul_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: conv_buffer,
                        offset: conv_offset,
                        size: NonZeroU64::new(size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: kernel_buffer,
                        offset: kernel_offset,
                        size: NonZeroU64::new(size),
                    }),
                },
            ],
        })
    }

    /// Create a bind group for fused cmul + IFFT column with specific offsets.
    #[allow(dead_code)]
    fn make_fused_cmul_ifft_bg(
        &self,
        device: &wgpu::Device,
        conv_buffer: &wgpu::Buffer,
        conv_offset: u64,
        kernel_buffer: &wgpu::Buffer,
        kernel_offset: u64,
        size: u64,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ml::fused_cmul_ifft_bg"),
            layout: &self.fused_cmul_ifft_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: conv_buffer,
                        offset: conv_offset,
                        size: NonZeroU64::new(size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.twiddle_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.inv_fft_params_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: kernel_buffer,
                        offset: kernel_offset,
                        size: NonZeroU64::new(size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 41,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.bitrev_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        })
    }

    /// Run the convolution phase.
    ///
    /// Processes all channels and kernels in bulk:
    /// 1. Copy all channels to conv buffer (real → complex)
    /// 2. FFT all channels (row + col)
    /// 3. For each kernel: copy FFT'd input channel to kernel's slot
    /// 4. Complex multiply each kernel with its FFT'd kernel
    /// 5. IFFT all kernels (col + row)
    pub fn run(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        channel_buffer: &wgpu::Buffer,
        kernel_buffer: &wgpu::Buffer,
        c0: &[u32],
        num_kernels: usize,
        num_channels: usize,
        total_elements: usize,
        shape: &[usize],
    ) {
        let width = shape[0] as u32;
        let elem_size = 4u64; // f32
        let complex_size = 8u64; // vec2<f32>
        let ch_bytes = (total_elements as u64) * elem_size;
        let ch_complex_bytes = (total_elements as u64) * complex_size;

        // ================================================================
        // Step 1: Copy all channels to conv buffer (real f32 → complex vec2)
        // ================================================================
        for c in 0..num_channels {
            let ch_offset = (c as u64) * ch_bytes;
            let conv_offset = (c as u64) * ch_complex_bytes;

            let copy_bg = self.make_copy_bg(
                device,
                channel_buffer,
                ch_offset,
                &self.conv_buffer,
                conv_offset,
                ch_bytes,
            );

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ml::copy_to_conv"),
            });
            cpass.set_pipeline(&self.copy_to_conv_pipeline);
            cpass.set_bind_group(0, &copy_bg, &[]);
            let wg = (total_elements as u32 + 255) / 256;
            cpass.dispatch_workgroups(wg, 1, 1);
        }

        // ================================================================
        // Step 2: FFT all channels (row + col)
        // ================================================================
        for c in 0..num_channels {
            let conv_offset = (c as u64) * ch_complex_bytes;
            let fft_bg = self.make_fft_bg(
                device,
                &self.conv_buffer,
                conv_offset,
                ch_complex_bytes,
                &self.fft_params_buffer,
            );

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ml::fft_row"),
                });
                cpass.set_pipeline(&self.fft_row_pipeline);
                cpass.set_bind_group(0, &fft_bg, &[]);
                cpass.dispatch_workgroups(width, 1, 1);
            }

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ml::fft_col"),
                });
                cpass.set_pipeline(&self.fft_col_pipeline);
                cpass.set_bind_group(0, &fft_bg, &[]);
                cpass.dispatch_workgroups(width, 1, 1);
            }
        }

        // ================================================================
        // Step 3: For each kernel, copy FFT'd input channel to kernel's slot
        // Uses temp_buffer as intermediary (wgpu forbids same-buffer copy)
        // ================================================================
        for k in 0..num_kernels {
            let in_ch = c0[k] as usize;
            let src_offset = (in_ch as u64) * ch_complex_bytes;
            let dst_offset = (k as u64) * ch_complex_bytes;

            // conv_buffer[src_offset] -> temp_buffer
            encoder.copy_buffer_to_buffer(
                &self.conv_buffer,
                src_offset,
                &self.temp_buffer,
                0,
                ch_complex_bytes,
            );

            // temp_buffer -> conv_buffer[dst_offset]
            encoder.copy_buffer_to_buffer(
                &self.temp_buffer,
                0,
                &self.conv_buffer,
                dst_offset,
                ch_complex_bytes,
            );
        }

        // ================================================================
        // Step 4: Complex multiply each kernel with its FFT'd kernel
        // ================================================================
        for k in 0..num_kernels {
            let conv_offset = (k as u64) * ch_complex_bytes;
            let kernel_offset = (k as u64) * ch_complex_bytes;

            let cmul_bg = self.make_cmul_bg(
                device,
                &self.conv_buffer,
                conv_offset,
                kernel_buffer,
                kernel_offset,
                ch_complex_bytes,
            );

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ml::cmul"),
            });
            cpass.set_pipeline(&self.complex_mul_pipeline);
            cpass.set_bind_group(0, &cmul_bg, &[]);
            let wg = (total_elements as u32 + 255) / 256;
            cpass.dispatch_workgroups(wg, 1, 1);
        }

        // ================================================================
        // Step 5: IFFT all kernels (col + row)
        // ================================================================
        for k in 0..num_kernels {
            let conv_offset = (k as u64) * ch_complex_bytes;

            let inv_fft_bg = self.make_fft_bg(
                device,
                &self.conv_buffer,
                conv_offset,
                ch_complex_bytes,
                &self.inv_fft_params_buffer,
            );

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ml::ifft_col"),
                });
                cpass.set_pipeline(&self.ifft_col_pipeline);
                cpass.set_bind_group(0, &inv_fft_bg, &[]);
                cpass.dispatch_workgroups(width, 1, 1);
            }

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ml::ifft_row"),
                });
                cpass.set_pipeline(&self.ifft_row_pipeline);
                cpass.set_bind_group(0, &inv_fft_bg, &[]);
                cpass.dispatch_workgroups(width, 1, 1);
            }
        }
    }
}
