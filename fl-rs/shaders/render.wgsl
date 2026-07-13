// --- Fullscreen triangle vertex shader ---

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi & 2u) * 2 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

// --- Channel visualization fragment shader ---

struct RenderParams {
    grid_size: u32,
    screen_width: f32,
    screen_height: f32,
}

@group(0) @binding(0) var<storage, read> channel: array<f32>;
@group(0) @binding(1) var<uniform> render_params: RenderParams;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let cell_w = render_params.screen_width / f32(render_params.grid_size);
    let cell_h = render_params.screen_height / f32(render_params.grid_size);
    let col = u32(pos.x / cell_w);
    let row = u32(pos.y / cell_h);

    if (col >= render_params.grid_size || row >= render_params.grid_size) {
        return vec4<f32>(0.01, 0.01, 0.02, 1.0);
    }

    let idx = row * render_params.grid_size + col;

    // Read all 3 channels from packed buffer
    let total_pixels = render_params.grid_size * render_params.grid_size;
    let c0 = channel[idx];
    let c1 = channel[total_pixels + idx];
    let c2 = channel[2u * total_pixels + idx];

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
