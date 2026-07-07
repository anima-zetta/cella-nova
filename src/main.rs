use lenia_ca::gpu_flow_lenia::{generate_flow_kernels, GpuFlowLenia};
use lenia_ca::wfft::WgpuContext;
use ndarray::Array2;
use rand::Rng;
use std::sync::Arc;
use std::time::Instant;
use winit::event::{Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

const GRID_SIZE: usize = 1024;

// ---------------------------------------------------------------------------
// Render shaders
// ---------------------------------------------------------------------------

const RENDER_VERTEX_SHADER: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi & 2u) * 2 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}
"#;

const RENDER_FRAGMENT_SHADER: &str = r#"
struct Params {
    grid_size: u32,
    screen_width: f32,
    screen_height: f32,
}

@group(0) @binding(0) var<storage, read> channel: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let cell_w = params.screen_width / f32(params.grid_size);
    let cell_h = params.screen_height / f32(params.grid_size);
    let col = u32(pos.x / cell_w);
    let row = u32(pos.y / cell_h);

    if (col >= params.grid_size || row >= params.grid_size) {
        return vec4<f32>(0.01, 0.01, 0.02, 1.0);
    }

    // Read all 3 channels from packed buffer
    let total_pixels = params.grid_size * params.grid_size;
    let c0 = channel[row * params.grid_size + col];
    let c1 = channel[total_pixels + row * params.grid_size + col];
    let c2 = channel[2u * total_pixels + row * params.grid_size + col];

    // RGB mapping with boosted contrast
    let r = clamp(c0 * 1.5, 0.0, 1.0);
    let g = clamp(c1 * 1.5, 0.0, 1.0);
    let b = clamp(c2 * 1.5, 0.0, 1.0);
    let intensity = r + g + b;

    if (intensity < 0.005) {
        return vec4<f32>(0.01, 0.01, 0.02, 1.0);
    }

    return vec4<f32>(
        pow(r, 0.5),
        pow(g, 0.5),
        pow(b, 0.5),
        1.0,
    );
}
"#;

// ---------------------------------------------------------------------------
// Initialization helpers
// ---------------------------------------------------------------------------

fn add_blob(array: &mut Array2<f64>, cx: usize, cy: usize, radius: f64, strength: f64) {
    let shape = [array.shape()[0], array.shape()[1]];
    let r = radius as i32;
    let i_min = (cx as i32 - r).max(0) as usize;
    let i_max = (cx as usize + r as usize).min(shape[0]);
    let j_min = (cy as i32 - r).max(0) as usize;
    let j_max = (cy as usize + r as usize).min(shape[1]);

    for i in i_min..i_max {
        for j in j_min..j_max {
            let dx = i as f64 - cx as f64;
            let dy = j as f64 - cy as f64;
            let dist = (dx * dx + dy * dy).sqrt() / radius;
            if dist < 1.0 {
                let val = (-((dist - 0.5) * (dist - 0.5)) / (2.0 * 0.15 * 0.15)).exp();
                array[[i, j]] = (array[[i, j]] + val * strength).min(1.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Flow Lenia: GPU-Accelerated Mass-Conserving Life")
        .with_inner_size(winit::dpi::LogicalSize::new(
            GRID_SIZE as f64,
            GRID_SIZE as f64,
        ))
        .build(&event_loop)
        .unwrap();

    // --- wgpu setup ---
    let instance = wgpu::Instance::default();
    let surface = unsafe { instance.create_surface(&window) }.unwrap();

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .expect("No suitable GPU adapter found!");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Flow Lenia Device"),
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("Failed to request GPU device!");

    let window_size = window.inner_size();
    let mut surface_config = surface
        .get_default_config(&adapter, window_size.width, window_size.height)
        .unwrap();
    surface_config.present_mode = wgpu::PresentMode::AutoVsync;
    surface.configure(&device, &surface_config);

    let context = Arc::new(WgpuContext::from_device(device, queue));

    // --- Render pipeline ---
    let vertex_shader = context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vertex shader"),
            source: wgpu::ShaderSource::Wgsl(RENDER_VERTEX_SHADER.into()),
        });
    let fragment_shader = context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fragment shader"),
            source: wgpu::ShaderSource::Wgsl(RENDER_FRAGMENT_SHADER.into()),
        });

    let render_bgl = context
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let render_pl = context
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render pl"),
            bind_group_layouts: &[&render_bgl],
            push_constant_ranges: &[],
        });

    let render_pipeline = context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render pipeline"),
            layout: Some(&render_pl),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

    // --- Render params ---
    let render_params_buf = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("render params"),
        size: 12,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // --- Flow Lenia setup ---
    let shape = GRID_SIZE.next_power_of_two();
    let num_channels: usize = 3;
    let num_kernels: usize = 9;

    // Channel mapping: 3x3 matrix
    //   c0: which channel each kernel reads from
    //   c1: which kernels contribute to each channel
    // Matrix: [[2,1,0],[0,2,1],[1,0,2]] (cyclic food chain)
    let c0: Vec<u32> = vec![2, 1, 0, 0, 2, 1, 1, 0, 2];
    let c1: Vec<Vec<u32>> = vec![
        vec![2, 3, 7], // channel 0 gets kernels 2,3,7
        vec![1, 5, 6], // channel 1 gets kernels 1,5,6
        vec![0, 4, 8], // channel 2 gets kernels 0,4,8
    ];

    // Per-kernel growth parameters (randomized for diversity)
    let mut rng = rand::thread_rng();
    let kernel_m: Vec<f32> = (0..num_kernels).map(|_| rng.gen_range(0.05..0.5)).collect();
    let kernel_s: Vec<f32> = (0..num_kernels)
        .map(|_| rng.gen_range(0.001..0.18))
        .collect();
    let kernel_h: Vec<f32> = (0..num_kernels).map(|_| rng.gen_range(0.01..1.0)).collect();

    let dt: f32 = 0.2;
    let dd: i32 = 5;
    let sigma: f32 = 0.65;

    let game = GpuFlowLenia::new(
        Arc::clone(&context),
        &[shape, shape],
        num_channels,
        num_kernels,
        &c0,
        &c1,
        &kernel_m,
        &kernel_s,
        &kernel_h,
        dt,
        dd,
        sigma,
    );

    // Generate Flow Lenia kernels
    let global_r: f32 = rng.gen_range(2.0..25.0);
    let radii: Vec<f32> = (0..num_kernels).map(|_| rng.gen_range(0.2..1.0)).collect();
    let a: Vec<[f32; 3]> = (0..num_kernels)
        .map(|_| {
            [
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
            ]
        })
        .collect();
    let w: Vec<[f32; 3]> = (0..num_kernels)
        .map(|_| {
            [
                rng.gen_range(0.01..0.5),
                rng.gen_range(0.01..0.5),
                rng.gen_range(0.01..0.5),
            ]
        })
        .collect();
    let b: Vec<[f32; 3]> = (0..num_kernels)
        .map(|_| {
            [
                rng.gen_range(0.001..1.0),
                rng.gen_range(0.001..1.0),
                rng.gen_range(0.001..1.0),
            ]
        })
        .collect();

    let kernels_fft = generate_flow_kernels(shape, global_r, &radii, &a, &w, &b);
    for (k, kfft) in kernels_fft.iter().enumerate() {
        game.set_kernel(kfft, k);
    }

    // Initialize channels with random blobs
    for c in 0..num_channels {
        let mut ch = Array2::<f64>::zeros([shape; 2]);
        for _ in 0..20 {
            let x = rng.gen_range(30..shape - 30);
            let y = rng.gen_range(30..shape - 30);
            let radius = rng.gen_range(10.0..30.0);
            add_blob(&mut ch, x, y, radius, 0.9);
        }
        let flat: Vec<f64> = ch.iter().copied().collect();
        let max_val = flat.iter().cloned().fold(0.0f64, f64::max);
        println!("Channel {c}: max initial value = {max_val:.4}");
        game.upload_channel(&flat, c);
    }

    // --- Render bind group ---
    let render_bg = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render bg"),
            layout: &render_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: game.channel_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: render_params_buf.as_entire_binding(),
                },
            ],
        });

    // --- State ---
    let mut paused = false;
    let mut last_fps_time = Instant::now();
    let mut frame_count: u32 = 0;
    let mut debug_printed = false;

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║   🌊 Flow Lenia: GPU-Accelerated Mass-Conserving CA  🌊 ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║  {} channels, {} kernels, {}×{} grid                 ║",
        num_channels, num_kernels, GRID_SIZE, GRID_SIZE
    );
    println!(
        "║  Reintegration tracking: dd={}, σ={}                ║",
        dd, sigma
    );
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Controls:                                              ║");
    println!("║    [Space]     Pause/Resume simulation                 ║");
    println!("║    [R]         Reset with new random state             ║");
    println!("║    [Q/Esc]     Quit                                    ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,

            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            winit::event::KeyboardInput {
                                virtual_keycode: Some(key),
                                state: winit::event::ElementState::Pressed,
                                ..
                            },
                        ..
                    },
                ..
            } => match key {
                VirtualKeyCode::Space => paused = !paused,
                VirtualKeyCode::R => {
                    let mut rng = rand::thread_rng();
                    for c in 0..num_channels {
                        let mut ch = Array2::<f64>::zeros([shape; 2]);
                        for _ in 0..20 {
                            let x = rng.gen_range(30..shape - 30);
                            let y = rng.gen_range(30..shape - 30);
                            let radius = rng.gen_range(10.0..30.0);
                            add_blob(&mut ch, x, y, radius, 0.9);
                        }
                        let flat: Vec<f64> = ch.iter().copied().collect();
                        game.upload_channel(&flat, c);
                    }
                }
                VirtualKeyCode::Q | VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                _ => {}
            },

            Event::MainEventsCleared => {
                if !paused {
                    game.iterate();
                }
                window.request_redraw();
            }

            Event::RedrawRequested(_) => {
                // Debug: print channel stats after a few frames
                if !debug_printed && frame_count == 10 {
                    debug_printed = true;
                    for c in 0..num_channels {
                        let data = game.download_channel(c);
                        let max_val = data.iter().cloned().fold(0.0f32, f32::max);
                        let sum: f32 = data.iter().sum();
                        let non_zero = data.iter().filter(|&&v| v > 0.001).count();
                        println!("Frame {frame_count} ch{c}: max={max_val:.4}, sum={sum:.2}, non_zero={non_zero}/{}", data.len());
                    }
                }
                frame_count += 1;
                let now = Instant::now();
                let elapsed = now.duration_since(last_fps_time).as_secs_f64();
                if elapsed >= 1.0 {
                    let fps = frame_count as f64 / elapsed;
                    window.set_title(&format!(
                        "Flow Lenia GPU — {:.0} FPS ({}×{} grid, {}ch, {}kernels)",
                        fps, GRID_SIZE, GRID_SIZE, num_channels, num_kernels
                    ));
                    last_fps_time = now;
                    frame_count = 0;
                }

                // Update render params
                {
                    let mut data = Vec::with_capacity(12);
                    data.extend_from_slice(&(GRID_SIZE as u32).to_le_bytes());
                    data.extend_from_slice(&(surface_config.width as f32).to_le_bytes());
                    data.extend_from_slice(&(surface_config.height as f32).to_le_bytes());
                    context.queue.write_buffer(&render_params_buf, 0, &data);
                }

                let frame = surface.get_current_texture().unwrap();
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder =
                    context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("render encoder"),
                        });

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("render pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.01,
                                    g: 0.01,
                                    b: 0.02,
                                    a: 1.0,
                                }),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: None,
                    });
                    rpass.set_pipeline(&render_pipeline);
                    rpass.set_bind_group(0, &render_bg, &[]);
                    rpass.draw(0..3, 0..1);
                }

                context.queue.submit(Some(encoder.finish()));
                frame.present();
            }

            _ => {}
        }
    });
}
