use lenia_ca::gpu_flow_lenia::GpuFlowLenia;
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
        "glider".to_string()
    }
}

// ---------------------------------------------------------------------------
// Kernel loading
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
// Seed loading (pre-computed channels from seed/glider.json)
// ---------------------------------------------------------------------------

fn load_seed(creature: &str) -> Vec<Vec<f64>> {
    #[derive(Deserialize)]
    struct GliderConfig {
        seed_channels: Vec<Vec<f64>>,
    }

    let path = format!("seed/{}.json", creature);
    let config_str = std::fs::read_to_string(&path).expect(&format!("Failed to read {}", path));
    let config: GliderConfig =
        serde_json::from_str(&config_str).expect(&format!("Failed to parse {}", path));

    config.seed_channels
}

// ---------------------------------------------------------------------------
// Render shaders
// ---------------------------------------------------------------------------

const RENDER_VERTEX_SHADER: &str = include_str!("shaders/render_vertex.wgsl");

const RENDER_FRAGMENT_SHADER: &str = include_str!("shaders/render_fragment.wgsl");

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

    // Write render params once (window is not resizable)
    let mut data = Vec::with_capacity(12);
    data.extend_from_slice(&(GRID_SIZE as u32).to_le_bytes());
    data.extend_from_slice(&(surface_config.width as f32).to_le_bytes());
    data.extend_from_slice(&(surface_config.height as f32).to_le_bytes());
    queue.write_buffer(&render_params_buf, 0, &data);

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

    // --- Flow Lenia setup ---
    let shape = GRID_SIZE.next_power_of_two();
    let num_channels: usize = 3;
    let num_kernels: usize = 9;

    // Channel mapping: M = [[2,1,0],[0,2,1],[1,0,2]]
    //   c0: source channel for each kernel
    //   c1: which kernels contribute to each target channel
    let c0: Vec<u32> = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let c1: Vec<Vec<u32>> = vec![
        vec![0, 1, 6], // ch0: 2 self + 1 from ch2
        vec![2, 3, 4], // ch1: 1 from ch0 + 2 self
        vec![5, 7, 8], // ch2: 1 from ch1 + 2 self
    ];

    // Growth parameters matching Python's FlowLeniaTorch: μ=0, σ=5, h=1
    let kernel_m: Vec<f32> = vec![0.0; 9];
    let kernel_s: Vec<f32> = vec![5.0; 9];
    let kernel_h: Vec<f32> = vec![1.0; 9];

    let dt: f32 = 0.2;
    let dd: i32 = 5;
    let sigma: f32 = 0.65;
    let basal_metabolic_rate: f32 = 0.001;
    let kinetic_cost: f32 = 0.0005;

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
        basal_metabolic_rate,
        kinetic_cost,
    );

    let creature = creature_name();
    println!("Creature: {}", creature);

    // Load pre-computed kernels from file
    let kernel_path = format!("kernels/{}.bin", creature);
    let kernels_fft = load_kernels_fft(&kernel_path, num_kernels, shape);
    for (k, kfft) in kernels_fft.iter().enumerate() {
        game.set_kernel(kfft, k);
    }

    // Load pre-computed seed
    let seed_channels = load_seed(&creature);
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

    // --- State ---
    let mut last_fps_time = Instant::now();
    let mut frame_count: u32 = 0;

    println!("Flow Lenia: GPU-Accelerated Mass-Conserving CA");
    println!(
        "{} channels, {} kernels, {}×{} grid",
        num_channels, num_kernels, GRID_SIZE, GRID_SIZE
    );
    println!("Reintegration tracking: dd={}, σ={}", dd, sigma);
    println!(
        "Metabolism: basal={}, kinetic={}",
        basal_metabolic_rate, kinetic_cost
    );

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
                // Debug: print channel stats + center of mass at key frames
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
