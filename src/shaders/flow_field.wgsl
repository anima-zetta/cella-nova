struct Params { width: u32, num_channels: u32, num_channels_f32: f32 }
@group(0) @binding(0) var<storage, read> channels: array<f32>;
@group(0) @binding(1) var<storage, read> nabla_u_x: array<f32>;
@group(0) @binding(2) var<storage, read> nabla_u_y: array<f32>;
@group(0) @binding(3) var<storage, read> nabla_a_x: array<f32>;
@group(0) @binding(4) var<storage, read> nabla_a_y: array<f32>;
@group(0) @binding(5) var<storage, read_write> flow_x: array<f32>;
@group(0) @binding(6) var<storage, read_write> flow_y: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    let total: u32 = params.width * params.width * params.num_channels;
    if (i >= total) { return; }
    let c: u32 = i / (params.width * params.width);
    let pixel: u32 = i % (params.width * params.width);
    let a: f32 = channels[i];
    let alpha: f32 = clamp((a / params.num_channels_f32) * (a / params.num_channels_f32), 0.0, 1.0);
    let nux: f32 = nabla_u_x[i];
    let nuy: f32 = nabla_u_y[i];
    let nax: f32 = nabla_a_x[pixel];
    let nay: f32 = nabla_a_y[pixel];
    flow_x[i] = nux * (1.0 - alpha) - nax * alpha;
    flow_y[i] = nuy * (1.0 - alpha) - nay * alpha;
}
