use lenia_ca::orchestrator::GpuFlowLenia;
use lenia_ca::wfft::WgpuContext;
use serde::Deserialize;
use std::sync::Arc;
use std::time::Instant;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

const GRID_SIZE: usize = 512;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn creature_name() -> String {
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--creature") {
        args.get(pos + 1)
            .cloned()
            .unwrap_or_else(|| "glider".to_string())
    } else {
        panic!("Provide creature name");
    }
}

// ---------------------------------------------------------------------------
// Kernel loading (pre-FFT'd at GRID_SIZE)
// ---------------------------------------------------------------------------

/// Load pre-computed FFT kernels from binary file.
/// Format: for each kernel, `size * size` complex values as interleaved f32 (real, imag).
fn load_kernels_fft(
    path: &str,
    num_kernels: usize,
    size: usize,
) -> Vec<Vec<num_complex::Complex32>> {
    let data = std::fs::read(path).expect("Failed to read kernel FFT file");
    let expected = num_kernels * size * size * 8;
    assert_eq!(data.len(), expected, "Kernel FFT file size mismatch");

    let mut kernels = Vec::with_capacity(num_kernels);
    let kernel_len = size * size;
    for k in 0..num_kernels {
        let mut kernel = Vec::with_capacity(kernel_len);
        let base = k * kernel_len * 8;
        for i in 0..kernel_len {
            let offset = base + i * 8;
            let re = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            let im = f32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap());
            kernel.push(num_complex::Complex32::new(re, im));
        }
        kernels.push(kernel);
    }
    kernels
}

// ---------------------------------------------------------------------------
// Seed loading (grid-size independent: stored at seed_size, padded to grid)
// ---------------------------------------------------------------------------

fn load_seed(creature: &str, grid_size: usize) -> (Vec<Vec<f64>>, Vec<f32>, Vec<f32>, Vec<f32>) {
    #[derive(Deserialize)]
    struct GrowthParams {
        m: Vec<f32>,
        s: Vec<f32>,
        h: Vec<f32>,
    }

    #[derive(Deserialize)]
    struct CreatureConfig {
        seed_size: usize,
        seed_channels: Vec<Vec<f64>>,
        growth_params: GrowthParams,
    }

    let path = format!("seed/{}.json", creature);
    let config_str = std::fs::read_to_string(&path).expect(&format!("Failed to read {}", path));
    let config: CreatureConfig =
        serde_json::from_str(&config_str).expect(&format!("Failed to parse {}", path));

    let seed_size = config.seed_size;
    let pad = (grid_size - seed_size) / 2;

    let mut padded: Vec<Vec<f64>> = Vec::with_capacity(config.seed_channels.len());
    for ch in &config.seed_channels {
        let mut p = vec![0.0f64; grid_size * grid_size];
        for iy in 0..seed_size {
            for ix in 0..seed_size {
                let src_idx = iy * seed_size + ix;
                let dst_idx = (pad + iy) * grid_size + (pad + ix);
                p[dst_idx] = ch[src_idx];
            }
        }
        padded.push(p);
    }

    let gp = config.growth_params;
    (padded, gp.m, gp.s, gp.h)
}

// ---------------------------------------------------------------------------
// Render shaders
// ---------------------------------------------------------------------------

const RENDER_SHADER: &str = include_str!("shaders/render.wgsl");

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // --- Interactive mode ---
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Flow Lenia: GPU-Accelerated Mass-Conserving Life")
        .with_inner_size(winit::dpi::LogicalSize::new(
            GRID_SIZE as f64,
            GRID_SIZE as f64,
        ))
        .with_resizable(false)
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

    // --- Render params ---
    let render_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("render params"),
        size: 12,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut data = Vec::with_capacity(12);
    data.extend_from_slice(&(GRID_SIZE as u32).to_le_bytes());
    data.extend_from_slice(&(surface_config.width as f32).to_le_bytes());
    data.extend_from_slice(&(surface_config.height as f32).to_le_bytes());
    queue.write_buffer(&render_params_buf, 0, &data);

    let context = Arc::new(WgpuContext::from_device(device, queue));

    // --- Render pipeline ---
    let render_shader = context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render shader"),
            source: wgpu::ShaderSource::Wgsl(RENDER_SHADER.into()),
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
                module: &render_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
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

    // --- Flow Lenia setup ---
    let shape = GRID_SIZE.next_power_of_two();
    let num_channels: usize = 3;
    let num_kernels: usize = 9;

    let c0: Vec<u32> = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let c1: Vec<Vec<u32>> = vec![vec![0, 1, 6], vec![2, 3, 4], vec![5, 7, 8]];

    let dt: f32 = 0.2;
    let dd: i32 = 5;
    let sigma: f32 = 0.65;

    let creature = creature_name();
    println!("Creature: {}", creature);

    // Load seed + growth params (grid-size independent, padded to GRID_SIZE)
    let (seed_channels, kernel_m, kernel_s, kernel_h) = load_seed(&creature, GRID_SIZE);

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

    // Load pre-FFT'd kernels from file (generated by Python at GRID_SIZE)
    let kernel_path = format!("kernels/{}_512.bin", creature);
    let kernels_fft = load_kernels_fft(&kernel_path, num_kernels, shape);
    for (k, kfft) in kernels_fft.iter().enumerate() {
        game.set_kernel(kfft, k);
    }

    // Upload padded seed
    for (c, data) in seed_channels.iter().enumerate() {
        game.upload_channel(data, c);
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

    let mut last_fps_time = Instant::now();
    let mut frame_count: u32 = 0;

    println!("Flow Lenia: GPU-Accelerated Mass-Conserving CA");
    println!(
        "{} channels, {} kernels, {}x{} grid",
        num_channels, num_kernels, GRID_SIZE, GRID_SIZE
    );
    println!("Reintegration tracking: dd={}, sigma={}", dd, sigma);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,

            Event::MainEventsCleared => {
                game.iterate();
                window.request_redraw();
            }

            Event::RedrawRequested(_) => {
                frame_count += 1;
                let now = Instant::now();
                let elapsed = now.duration_since(last_fps_time).as_secs_f64();
                if elapsed >= 1.0 {
                    let fps = frame_count as f64 / elapsed;
                    window.set_title(&format!(
                        "Flow Lenia GPU -- {:.0} FPS ({}x{} grid, {}ch, {}kernels)",
                        fps, GRID_SIZE, GRID_SIZE, num_channels, num_kernels
                    ));
                    last_fps_time = now;
                    frame_count = 0;
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
