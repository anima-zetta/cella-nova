struct Params { width: u32, height: u32, dd: i32, sigma: f32, dt: f32, num_channels: u32, ma: f32, basal_rate: f32, kinetic_cost: f32 }
@group(0) @binding(0) var<storage, read> channel: array<f32>;
@group(0) @binding(1) var<storage, read> flow_x: array<f32>;
@group(0) @binding(2) var<storage, read> flow_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> new_channel: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx: u32 = id.x;
    let total: u32 = params.width * params.height * params.num_channels;
    if (idx >= total) { return; }
    let c: u32 = idx / (params.width * params.height);
    let pixel: u32 = idx % (params.width * params.height);
    let x: u32 = pixel % params.width;
    let y: u32 = pixel / params.width;
    let pos_x: f32 = f32(x) + 0.5;
    let pos_y: f32 = f32(y) + 0.5;
    let dd: i32 = params.dd;
    let sigma: f32 = params.sigma;
    let dt: f32 = params.dt;
    let ma: f32 = params.ma;
    let w: u32 = params.width;
    let h: u32 = params.height;
    let w_i32: i32 = i32(w);
    let h_i32: i32 = i32(h);
    let max_sz: f32 = min(1.0, 2.0 * sigma);
    let area_norm: f32 = 4.0 * sigma * sigma;
    let c_base: u32 = c * w * h;
    var sum: f32 = 0.0;
    for (var dx: i32 = -dd; dx <= dd; dx = dx + 1) {
        for (var dy: i32 = -dd; dy <= dd; dy = dy + 1) {
            let nx: i32 = i32(x) + dx;
            let ny: i32 = i32(y) + dy;
            if (nx < 0 || nx >= w_i32 || ny < 0 || ny >= h_i32) { continue; }
            let n_idx: u32 = u32(ny) * w + u32(nx);
            let a: f32 = channel[c_base + n_idx];
            if (a <= 0.0) { continue; }
            let n_pos_x: f32 = f32(nx) + 0.5;
            let n_pos_y: f32 = f32(ny) + 0.5;
            let fx: f32 = clamp(flow_x[c_base + n_idx], -ma, ma);
            let fy: f32 = clamp(flow_y[c_base + n_idx], -ma, ma);
            let mu_x: f32 = clamp(n_pos_x + fx * dt, sigma, f32(w) - sigma);
            let mu_y: f32 = clamp(n_pos_y + fy * dt, sigma, f32(h) - sigma);
            let dpx: f32 = abs(pos_x - mu_x);
            let dpy: f32 = abs(pos_y - mu_y);
            let sz_x: f32 = clamp(0.5 - dpx + sigma, 0.0, max_sz);
            let sz_y: f32 = clamp(0.5 - dpy + sigma, 0.0, max_sz);
            let area: f32 = (sz_x * sz_y) / area_norm;
            sum = sum + a * area;
        }
    }
    // Metabolic costs: basal decay + kinetic cost proportional to local flow
    let fx_self: f32 = clamp(flow_x[idx], -ma, ma);
    let fy_self: f32 = clamp(flow_y[idx], -ma, ma);
    let flow_mag: f32 = sqrt(fx_self * fx_self + fy_self * fy_self);
    sum = sum * (1.0 - params.basal_rate * dt) - params.kinetic_cost * flow_mag * dt;
    new_channel[idx] = max(sum, 0.0);
}
