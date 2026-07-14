mod orchestrator;
mod wfft;

use orchestrator::GpuFlowLenia;
use serde::Deserialize;
use std::sync::Arc;
use std::time::Instant;
use wfft::WgpuContext;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

const GRID_SIZE: usize = 512;
const CELL_SIZE: usize = 64;
const CELLS_PER_ROW: usize = GRID_SIZE / CELL_SIZE; // 8

// ---------------------------------------------------------------------------
// Creature config (subset of the JSON)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct GrowthParams {
    m: Vec<f32>,
    s: Vec<f32>,
    h: Vec<f32>,
}

#[derive(Deserialize)]
struct BumpParams {
    num_kernels: usize,
}

#[derive(Deserialize)]
struct CreatureConfig {
    seed_size: usize,
    num_channels: usize,
    seed_channels: Vec<Vec<f64>>,
    bump_params: BumpParams,
    growth_params: GrowthParams,
}

// ---------------------------------------------------------------------------
// Load all random creatures from seed/
// ---------------------------------------------------------------------------

fn discover_creatures() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir("seed")
        .expect("Failed to read seed/ directory")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let file_name = path.file_stem()?.to_str()?.to_string();
            if file_name.starts_with("random_") && path.extension()? == "json" {
                Some(file_name)
            } else {
                None
            }
        })
        .collect();
    names.sort();
    names
}

fn load_creature_config(name: &str) -> CreatureConfig {
    let path = format!("seed/{}.json", name);
    let config_str =
        std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("Failed to read {}", path));
    serde_json::from_str(&config_str).unwrap_or_else(|_| panic!("Failed to parse {}", path))
}

// ---------------------------------------------------------------------------
// Kernel loading (pre-FFT'd at GRID_SIZE)
// ---------------------------------------------------------------------------

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
// Render shaders
// ---------------------------------------------------------------------------

const RENDER_SHADER: &str = include_str!("shaders/render.wgsl");

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // --- Discover creatures ---
    let creature_names = discover_creatures();
    let num_creatures = creature_names.len();
    assert!(
        num_creatures > 0,
        "No random_* creatures found in seed/ directory"
    );
    println!("Found {} creatures", num_creatures);

    // Load all creature configs
    let configs: Vec<CreatureConfig> = creature_names
        .iter()
        .map(|name| load_creature_config(name))
        .collect();

    // Compute totals
    let num_channels: usize = configs[0].num_channels; // all creatures have the same
    let num_kernels: usize = configs.iter().map(|c| c.bump_params.num_kernels).sum();

    // Build flat arrays
    let mut all_kernel_m: Vec<f32> = Vec::with_capacity(num_kernels);
    let mut all_kernel_s: Vec<f32> = Vec::with_capacity(num_kernels);
    let mut all_kernel_h: Vec<f32> = Vec::with_capacity(num_kernels);
    for config in &configs {
        all_kernel_m.extend(&config.growth_params.m);
        all_kernel_s.extend(&config.growth_params.s);
        all_kernel_h.extend(&config.growth_params.h);
    }

    // Build C0/C1 mapping: simple round-robin across channels
    let c0: Vec<u32> = (0..num_kernels as u32)
        .map(|k| k % num_channels as u32)
        .collect();
    let c1: Vec<Vec<u32>> = (0..num_channels)
        .map(|c| {
            (0..num_kernels as u32)
                .filter(|&k| k % num_channels as u32 == c as u32)
                .collect()
        })
        .collect();

    println!(
        "{} channels, {} kernels, {} creatures",
        num_channels, num_kernels, num_creatures
    );

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
        &all_kernel_m,
        &all_kernel_s,
        &all_kernel_h,
        dt,
        dd,
        sigma,
    );

    // --- Upload seeds: place each creature in its 8x8 cell ---
    // Accumulate all creatures into a single buffer per channel, then upload once.
    let seed_size = configs[0].seed_size; // 64
    let mut channel_buffers: Vec<Vec<f64>> = (0..num_channels)
        .map(|_| vec![0.0f64; GRID_SIZE * GRID_SIZE])
        .collect();
    for (ci, config) in configs.iter().enumerate() {
        let row = ci / CELLS_PER_ROW;
        let col = ci % CELLS_PER_ROW;
        let offset_x = col * CELL_SIZE;
        let offset_y = row * CELL_SIZE;

        for c in 0..config.num_channels {
            let src = &config.seed_channels[c];
            for iy in 0..seed_size {
                for ix in 0..seed_size {
                    let dst_idx = (offset_y + iy) * GRID_SIZE + (offset_x + ix);
                    let src_idx = iy * seed_size + ix;
                    channel_buffers[c][dst_idx] = src[src_idx];
                }
            }
        }
    }
    for (c, buf) in channel_buffers.iter().enumerate() {
        game.upload_channel(buf, c);
    }

    // --- Load and upload kernels ---
    let mut kernel_offset = 0;
    for (ci, config) in configs.iter().enumerate() {
        let name = &creature_names[ci];
        let nk = config.bump_params.num_kernels;
        let kernel_path = format!("kernels/{}_512.bin", name);
        let kernels_fft = load_kernels_fft(&kernel_path, nk, shape);
        for (k, kfft) in kernels_fft.iter().enumerate() {
            game.set_kernel(kfft, kernel_offset + k);
        }
        kernel_offset += nk;
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
    let mut last_compute_ms: f64 = 0.0;

    println!("Flow Lenia: GPU-Accelerated Mass-Conserving CA");
    println!(
        "{} channels, {} kernels, {}x{} grid, {} creatures in {}x{} grid",
        num_channels,
        num_kernels,
        GRID_SIZE,
        GRID_SIZE,
        num_creatures,
        CELLS_PER_ROW,
        CELLS_PER_ROW
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
                window.request_redraw();
            }

            Event::RedrawRequested(_) => {
                frame_count += 1;
                let now = Instant::now();
                let elapsed = now.duration_since(last_fps_time).as_secs_f64();
                if elapsed >= 1.0 {
                    let fps = frame_count as f64 / elapsed;
                    window.set_title(&format!(
                        "Flow Lenia GPU -- {:.0} FPS | compute {:.1}ms ({}x{} grid, {}ch, {}kernels, {} creatures)",
                        fps, last_compute_ms, GRID_SIZE, GRID_SIZE, num_channels, num_kernels, num_creatures
                    ));
                    last_fps_time = now;
                    frame_count = 0;
                }

                // Get the surface texture first (may wait for vsync)
                let frame = surface.get_current_texture().unwrap();
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                // Single encoder: compute + render in one submit
                let mut encoder =
                    context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("frame encoder"),
                        });

                // Compute pass
                let compute_start = Instant::now();
                game.iterate_with_encoder(&mut encoder);
                last_compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

                // Render pass
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
