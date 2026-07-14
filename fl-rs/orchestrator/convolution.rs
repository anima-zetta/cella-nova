// ===========================================================================
// Phase 1: FFT-based convolution + growth
//
// For each unique source channel:
//   1. copy_to_conv  (real channel → complex FFT buffer)
//   2. FFT row pass
//   3. FFT col pass
//   4. save frequency-domain result
//   5. for each kernel sharing this channel:
//        a. restore frequency-domain data
//        b. complex_mul (channel × kernel in freq domain)
//        c. IFFT col pass
//        d. IFFT row pass
//        e. normalize_growth (scale + apply G(x))
// ===========================================================================

use super::GP_STRIDE;

pub struct ConvolutionPhase {
    // FFT pipelines + bind groups
    fft_row_pipeline: wgpu::ComputePipeline,
    fft_col_pipeline: wgpu::ComputePipeline,
    fft_bg: wgpu::BindGroup,
    inv_fft_bg: wgpu::BindGroup,

    // FFT working buffer: [X*Y] vec2<f32>
    conv_buffer: wgpu::Buffer,
    /// Saved frequency-domain data for sharing forward FFTs across kernels.
    conv_saved_buffer: wgpu::Buffer,

    // copy_to_conv
    copy_to_conv_pipeline: wgpu::ComputePipeline,
    copy_to_conv_bgl: wgpu::BindGroupLayout,

    // complex_mul
    complex_mul_pipeline: wgpu::ComputePipeline,
    complex_mul_bgl: wgpu::BindGroupLayout,

    // normalize_growth
    normalize_growth_pipeline: wgpu::ComputePipeline,

    // Per-kernel normalize+growth bind groups
    normalize_growth_bgs: Vec<wgpu::BindGroup>,

    // Cached per-kernel bind groups (created once in new(), reused in run())
    copy_bgs_cache: Vec<Vec<wgpu::BindGroup>>,
    cmul_bgs_cache: Vec<Vec<wgpu::BindGroup>>,

    // Cached channel-to-kernel mapping
    ch_to_kernels_cache: Vec<Vec<usize>>,
}

impl ConvolutionPhase {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        compute_shader: &str,
        shape: &[usize],
        total_elements: usize,
        num_kernels: usize,
        num_channels: usize,
        kernel_m: &[f32],
        kernel_s: &[f32],
        kernel_h: &[f32],
        c0: &[u32],
        // Owned buffers (only used within this phase)
        conv_buffer: wgpu::Buffer,
        conv_saved_buffer: wgpu::Buffer,
        // Shared buffers (references: owned by GpuFlowLenia)
        channel_buffer: &wgpu::Buffer,
        kernel_buffer: &wgpu::Buffer,
        u_buffer: &wgpu::Buffer,
        param_buffer: &wgpu::Buffer,
    ) -> Self {
        // --- Twiddle factors (Stockham arrangement) ---
        let n = shape[0] as usize;
        let num_stages = (n as f64).log2() as u32;
        let mut twiddles: Vec<[f32; 2]> = Vec::new();
        for stage in 0..num_stages {
            let block_size = 1u64 << (stage + 1);
            for k in 0..(block_size / 2) {
                let angle = -2.0 * std::f64::consts::PI * (k as f64) / (block_size as f64);
                twiddles.push([angle.cos() as f32, angle.sin() as f32]);
            }
        }
        while twiddles.len() < n {
            twiddles.push([0.0, 0.0]);
        }
        let twiddle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fl::twiddles"),
            size: (twiddles.len() * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&twiddle_buffer, 0, bytemuck::cast_slice(&twiddles));

        // --- Uniform buffers ---
        let make_uniform = |label: &str, size: u64| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let normalize_params_buffer = make_uniform("fl::normalize_params", 4);
        let fft_params_buffer = make_uniform("fl::fft_params", 8);
        let inv_fft_params_buffer = make_uniform("fl::inv_fft_params", 8);

        // Growth params buffer: per-kernel slots (storage, 256-byte aligned)
        let growth_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fl::growth_params"),
            size: (num_kernels as u64) * GP_STRIDE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        for k in 0..num_kernels {
            let gp: [f32; 3] = [kernel_m[k], kernel_s[k], kernel_h[k]];
            queue.write_buffer(
                &growth_params_buffer,
                (k as u64) * GP_STRIDE,
                bytemuck::cast_slice(&gp),
            );
        }

        // --- Write uniform params ---
        let padded_n = shape[0] as f32;
        let norm_factor = 1.0 / padded_n.powi(shape.len() as i32);
        queue.write_buffer(
            &normalize_params_buffer,
            0,
            bytemuck::cast_slice(&[norm_factor]),
        );

        queue.write_buffer(
            &fft_params_buffer,
            0,
            bytemuck::cast_slice(&[shape[0] as u32, 0u32]),
        );
        queue.write_buffer(
            &inv_fft_params_buffer,
            0,
            bytemuck::cast_slice(&[shape[0] as u32, 1u32]),
        );

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

        // --- Bind group layouts ---
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
                entries: &[srw(8), srw(9), unif(11), sro(12), sro(13)],
            });
        let fft_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fl::fft bgl"),
            entries: &[srw(0), sro(2), unif(3)],
        });

        // --- Pipeline layouts & pipelines ---
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

        let copy_to_conv_pipeline = cp(
            "fl::copy_to_conv",
            &pl("fl::copy_to_conv pl", &copy_to_conv_bgl),
            "copy_to_conv_main",
        );
        let complex_mul_pipeline = cp(
            "fl::complex_mul",
            &pl("fl::complex_mul pl", &complex_mul_bgl),
            "complex_mul_main",
        );
        let normalize_growth_pipeline = cp(
            "fl::normalize_growth",
            &pl("fl::normalize_growth pl", &normalize_growth_bgl),
            "normalize_growth_main",
        );
        let fft_row_pipeline = cp(
            "fl::fft_row",
            &pl("fl::fft_row pl", &fft_bgl),
            "fft_row_main",
        );
        let fft_col_pipeline = cp(
            "fl::fft_col",
            &pl("fl::fft_col pl", &fft_bgl),
            "fft_col_main",
        );

        // --- Cached bind groups ---
        let fft_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::fft bg"),
            layout: &fft_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: conv_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: twiddle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: fft_params_buffer.as_entire_binding(),
                },
            ],
        });

        let inv_fft_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fl::inv_fft bg"),
            layout: &fft_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: conv_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: twiddle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &inv_fft_params_buffer,
                        offset: 0,
                        size: std::num::NonZeroU64::new(8),
                    }),
                },
            ],
        });

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
                            buffer: u_buffer,
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
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: param_buffer,
                            offset: u_offset,
                            size: u_size,
                        }),
                    },
                ],
            }));
        }

        // --- Pre-compute per-kernel bind groups (cached, reused every iteration) ---
        let mut ch_to_kernels_cache: Vec<Vec<usize>> = Vec::new();
        ch_to_kernels_cache.resize_with(num_channels, Vec::new);
        for k in 0..num_kernels {
            ch_to_kernels_cache[c0[k] as usize].push(k);
        }

        let mut copy_bgs_cache: Vec<Vec<wgpu::BindGroup>> = Vec::new();
        let mut cmul_bgs_cache: Vec<Vec<wgpu::BindGroup>> = Vec::new();
        for kernels in &ch_to_kernels_cache {
            if kernels.is_empty() {
                copy_bgs_cache.push(Vec::new());
                cmul_bgs_cache.push(Vec::new());
                continue;
            }
            let mut copy_bgs = Vec::with_capacity(kernels.len());
            let mut cmul_bgs = Vec::with_capacity(kernels.len());
            for &k in kernels {
                let src_c = c0[k] as usize;
                let ch_offset = (src_c * total_elements * 4) as u64;
                let k_offset = (k * total_elements * 8) as u64;
                let ch_size = std::num::NonZeroU64::new((total_elements * 4) as u64);
                let k_size = std::num::NonZeroU64::new((total_elements * 8) as u64);

                copy_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("fl::copy_bg k={k}")),
                    layout: &copy_to_conv_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: channel_buffer,
                                offset: ch_offset,
                                size: ch_size,
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: conv_buffer.as_entire_binding(),
                        },
                    ],
                }));

                cmul_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("fl::cmul_bg k={k}")),
                    layout: &complex_mul_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: conv_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: kernel_buffer,
                                offset: k_offset,
                                size: k_size,
                            }),
                        },
                    ],
                }));
            }
            copy_bgs_cache.push(copy_bgs);
            cmul_bgs_cache.push(cmul_bgs);
        }

        Self {
            fft_row_pipeline,
            fft_col_pipeline,
            fft_bg,
            inv_fft_bg,
            copy_to_conv_pipeline,
            copy_to_conv_bgl,
            complex_mul_pipeline,
            complex_mul_bgl,
            normalize_growth_pipeline,
            normalize_growth_bgs,
            conv_buffer,
            conv_saved_buffer,
            copy_bgs_cache,
            cmul_bgs_cache,
            ch_to_kernels_cache,
        }
    }

    /// Run Phase 1: for each unique source channel, FFT-convolve with all
    /// kernels that share that channel, then apply the growth function.
    pub fn run(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        _channel_buffer: &wgpu::Buffer,
        _kernel_buffer: &wgpu::Buffer,
        _c0: &[u32],
        _num_kernels: usize,
        _num_channels: usize,
        total_elements: usize,
        shape: &[usize],
    ) {
        let wg = ((total_elements as u32) + 255) / 256;
        let sz = shape[0] as u32;
        let conv_buf_size = (total_elements * 8) as u64;

        for (gi, kernels) in self.ch_to_kernels_cache.iter().enumerate() {
            if kernels.is_empty() {
                continue;
            }

            let copy_bgs = &self.copy_bgs_cache[gi];
            let cmul_bgs = &self.cmul_bgs_cache[gi];

            // Step 1: Copy channel → conv (first kernel's source channel)
            self.dispatch(
                encoder,
                "copy",
                &self.copy_to_conv_pipeline,
                &copy_bgs[0],
                wg,
            );

            // Step 2: Forward FFT — row pass then column pass
            self.dispatch(encoder, "fft_row", &self.fft_row_pipeline, &self.fft_bg, sz);
            self.dispatch(encoder, "fft_col", &self.fft_col_pipeline, &self.fft_bg, sz);

            // Step 3: Save frequency-domain data for reuse across kernels
            encoder.copy_buffer_to_buffer(
                &self.conv_buffer,
                0,
                &self.conv_saved_buffer,
                0,
                conv_buf_size,
            );

            // Step 4: For each kernel sharing this channel...
            for (i, k) in kernels.iter().enumerate() {
                // Step 4a: Restore frequency-domain data (skip for first kernel)
                if i > 0 {
                    encoder.copy_buffer_to_buffer(
                        &self.conv_saved_buffer,
                        0,
                        &self.conv_buffer,
                        0,
                        conv_buf_size,
                    );
                }

                // Step 4b: Complex multiply (channel × kernel in frequency domain)
                self.dispatch(
                    encoder,
                    "cmul",
                    &self.complex_mul_pipeline,
                    &cmul_bgs[i],
                    wg,
                );

                // Step 4c: Inverse FFT — column pass then row pass
                self.dispatch(
                    encoder,
                    "inv_fft_col",
                    &self.fft_col_pipeline,
                    &self.inv_fft_bg,
                    sz,
                );
                self.dispatch(
                    encoder,
                    "inv_fft_row",
                    &self.fft_row_pipeline,
                    &self.inv_fft_bg,
                    sz,
                );

                // Step 4d: Normalize IFFT result + apply growth function (fused)
                self.dispatch(
                    encoder,
                    "norm_grow",
                    &self.normalize_growth_pipeline,
                    &self.normalize_growth_bgs[*k],
                    wg,
                );
            }
        }
    }

    /// Helper: begin a compute pass, set pipeline + bind group, dispatch.
    fn dispatch(
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
}
