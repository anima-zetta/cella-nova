// =============================================================================
// compute.wgsl — All compute shaders for the Flow Lenia simulation.
//
// Each entry point uses unique (group=0, binding) pairs so they can coexist
// in one module without conflicts.
// =============================================================================

// ---------------------------------------------------------------------------
// FFT: bit-reversal permutation  (bindings 0-1)
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read_write> fft_data: array<vec2<f32>>;

struct BitRevParams {
    n: u32,
    num_lanes: u32,
    lane_stride: u32,
    element_stride: u32,
}
@group(0) @binding(1) var<uniform> bit_rev_params: BitRevParams;

fn bit_reverse(x: u32, bits: u32) -> u32 {
    var result: u32 = 0u;
    for (var i: u32 = 0u; i < bits; i = i + 1u) {
        result = (result << 1u) | ((x >> i) & 1u);
    }
    return result;
}

@compute @workgroup_size(256)
fn bit_reverse_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let total: u32 = bit_rev_params.n * bit_rev_params.num_lanes;
    let i: u32 = id.x;
    if (i >= total) { return; }
    let lane: u32 = i / bit_rev_params.n;
    let offset_in_lane: u32 = i % bit_rev_params.n;
    let bits: u32 = u32(log2(f32(bit_rev_params.n)));
    let j: u32 = bit_reverse(offset_in_lane, bits);
    if (offset_in_lane < j) {
        let base: u32 = lane * bit_rev_params.lane_stride;
        let a: u32 = base + offset_in_lane * bit_rev_params.element_stride;
        let b: u32 = base + j * bit_rev_params.element_stride;
        let tmp: vec2<f32> = fft_data[a];
        fft_data[a] = fft_data[b];
        fft_data[b] = tmp;
    }
}

// ---------------------------------------------------------------------------
// FFT: butterfly stage  (bindings 0, 2-3)
// ---------------------------------------------------------------------------
@group(0) @binding(2) var<storage, read> twiddles: array<vec2<f32>>;

struct FftStageParams {
    n: u32,
    stage: u32,
    inverse: u32,
    num_lanes: u32,
    lane_stride: u32,
    element_stride: u32,
}
@group(0) @binding(3) var<uniform> fft_stage_params: FftStageParams;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(256)
fn fft_stage_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let half_n: u32 = fft_stage_params.n / 2u;
    let butterflies_per_lane: u32 = half_n;
    let total_butterflies: u32 = butterflies_per_lane * fft_stage_params.num_lanes;
    let i: u32 = id.x;
    if (i >= total_butterflies) { return; }
    let lane: u32 = i / butterflies_per_lane;
    let butterfly: u32 = i % butterflies_per_lane;
    let base: u32 = lane * fft_stage_params.lane_stride;
    let es: u32 = fft_stage_params.element_stride;
    let stride: u32 = 1u << fft_stage_params.stage;
    let block_size: u32 = stride * 2u;
    let block: u32 = butterfly / stride;
    let offset: u32 = butterfly % stride;
    let j: u32 = base + (block * block_size + offset) * es;
    let k: u32 = j + stride * es;
    let stage_offset: u32 = (1u << fft_stage_params.stage) - 1u;
    let w: vec2<f32> = twiddles[stage_offset + offset];
    let even: vec2<f32> = fft_data[j];
    let odd: vec2<f32> = complex_mul(w, fft_data[k]);
    fft_data[j] = even + odd;
    fft_data[k] = even - odd;
}

// ---------------------------------------------------------------------------
// copy_to_conv: real channel → complex conv buffer  (bindings 4-5)
// ---------------------------------------------------------------------------
@group(0) @binding(4) var<storage, read> ctc_channel: array<f32>;
@group(0) @binding(5) var<storage, read_write> ctc_conv: array<vec2<f32>>;

@compute @workgroup_size(256)
fn copy_to_conv_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    if (i >= arrayLength(&ctc_channel)) { return; }
    ctc_conv[i] = vec2<f32>(ctc_channel[i], 0.0);
}

// ---------------------------------------------------------------------------
// complex_mul: element-wise conv *= kernel  (bindings 6-7)
// ---------------------------------------------------------------------------
@group(0) @binding(6) var<storage, read_write> cm_conv: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read> cm_kernel: array<vec2<f32>>;

@compute @workgroup_size(256)
fn complex_mul_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    if (i >= arrayLength(&cm_conv)) { return; }
    cm_conv[i] = complex_mul(cm_conv[i], cm_kernel[i]);
}

// ---------------------------------------------------------------------------
// normalize_growth: normalize IFFT + apply growth  (bindings 8-12)
// ---------------------------------------------------------------------------
struct NormalizeParams { norm_factor: f32 }
struct GrowthParams { m: f32, s: f32, h: f32 }

@group(0) @binding(8) var<storage, read_write> ng_data: array<vec2<f32>>;
@group(0) @binding(9) var<storage, read_write> ng_result: array<f32>;
@group(0) @binding(10) var<storage, read_write> ng_conv_x: array<f32>;
@group(0) @binding(11) var<uniform> ng_norm_params: NormalizeParams;
@group(0) @binding(12) var<storage, read> ng_growth_params: GrowthParams;

@compute @workgroup_size(256)
fn normalize_growth_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    if (i >= arrayLength(&ng_result)) { return; }
    ng_data[i] = ng_data[i] * ng_norm_params.norm_factor;
    let x: f32 = ng_data[i].x;
    let diff: f32 = x - ng_growth_params.m;
    let g: f32 = exp(-(diff * diff) / (2.0 * ng_growth_params.s * ng_growth_params.s));
    ng_result[i] = (2.0 * g - 1.0) * ng_growth_params.h;
    ng_conv_x[i] = x;
}

// ---------------------------------------------------------------------------
// channel_aggregate: per-kernel growth → per-channel  (bindings 13-17)
// ---------------------------------------------------------------------------
struct CaParams { width: u32, num_kernels: u32, num_channels: u32 }

@group(0) @binding(13) var<storage, read> ca_u_all: array<f32>;
@group(0) @binding(14) var<storage, read_write> ca_u_channels: array<f32>;
@group(0) @binding(15) var<storage, read> ca_c1_flat: array<u32>;
@group(0) @binding(16) var<storage, read> ca_c1_offsets: array<u32>;
@group(0) @binding(17) var<uniform> ca_params: CaParams;

@compute @workgroup_size(256)
fn channel_aggregate_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    let total: u32 = ca_params.width * ca_params.width * ca_params.num_channels;
    if (i >= total) { return; }
    let c: u32 = i / (ca_params.width * ca_params.width);
    let pixel: u32 = i % (ca_params.width * ca_params.width);
    let start: u32 = ca_c1_offsets[c];
    let end: u32 = ca_c1_offsets[c + 1u];
    var sum: f32 = 0.0;
    for (var j: u32 = start; j < end; j = j + 1u) {
        let k: u32 = ca_c1_flat[j];
        sum = sum + ca_u_all[k * ca_params.width * ca_params.width + pixel];
    }
    ca_u_channels[i] = sum;
}

// ---------------------------------------------------------------------------
// sum_channels: sum all channels into one field  (bindings 18-20)
// ---------------------------------------------------------------------------
struct ScParams { width: u32, num_channels: u32 }

@group(0) @binding(18) var<storage, read> sc_channels: array<f32>;
@group(0) @binding(19) var<storage, read_write> sc_sum_out: array<f32>;
@group(0) @binding(20) var<uniform> sc_params: ScParams;

@compute @workgroup_size(256)
fn sum_channels_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    if (i >= sc_params.width * sc_params.width) { return; }
    var sum: f32 = 0.0;
    for (var c: u32 = 0u; c < sc_params.num_channels; c = c + 1u) {
        sum = sum + sc_channels[c * sc_params.width * sc_params.width + i];
    }
    sc_sum_out[i] = sum;
}

// ---------------------------------------------------------------------------
// sobel: 3×3 gradient  (bindings 21-24)
// ---------------------------------------------------------------------------
struct SobelParams { width: u32, height: u32, num_fields: u32 }

@group(0) @binding(21) var<storage, read> sobel_input: array<f32>;
@group(0) @binding(22) var<storage, read_write> sobel_grad_x: array<f32>;
@group(0) @binding(23) var<storage, read_write> sobel_grad_y: array<f32>;
@group(0) @binding(24) var<uniform> sobel_params: SobelParams;

@compute @workgroup_size(256)
fn sobel_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    let total: u32 = sobel_params.width * sobel_params.height * sobel_params.num_fields;
    if (i >= total) { return; }
    let field: u32 = i / (sobel_params.width * sobel_params.height);
    let pixel: u32 = i % (sobel_params.width * sobel_params.height);
    let x: u32 = pixel % sobel_params.width;
    let y: u32 = pixel / sobel_params.width;
    let w: u32 = sobel_params.width;
    let h: u32 = sobel_params.height;
    let base: u32 = field * w * h;
    let xl: u32 = select(x - 1u, 0u, x == 0u);
    let xr: u32 = select(x + 1u, w - 1u, x + 1u >= w);
    let yu: u32 = select(y - 1u, 0u, y == 0u);
    let yd: u32 = select(y + 1u, h - 1u, y + 1u >= h);
    let tl = sobel_input[base + yu * w + xl];
    let tc = sobel_input[base + yu * w + x];
    let tr = sobel_input[base + yu * w + xr];
    let ml = sobel_input[base + y * w + xl];
    let mr = sobel_input[base + y * w + xr];
    let bl = sobel_input[base + yd * w + xl];
    let bc = sobel_input[base + yd * w + x];
    let br = sobel_input[base + yd * w + xr];
    sobel_grad_x[i] = (-tl + tr) + 2.0 * (-ml + mr) + (-bl + br);
    sobel_grad_y[i] = (-tl - 2.0 * tc - tr) + (bl + 2.0 * bc + br);
}

// ---------------------------------------------------------------------------
// flow_field: compute flow from gradients  (bindings 25-32)
// ---------------------------------------------------------------------------
struct FlowFieldParams { width: u32, num_channels: u32, num_channels_f32: f32 }

@group(0) @binding(25) var<storage, read> ff_channels: array<f32>;
@group(0) @binding(26) var<storage, read> ff_nabla_u_x: array<f32>;
@group(0) @binding(27) var<storage, read> ff_nabla_u_y: array<f32>;
@group(0) @binding(28) var<storage, read> ff_nabla_a_x: array<f32>;
@group(0) @binding(29) var<storage, read> ff_nabla_a_y: array<f32>;
@group(0) @binding(30) var<storage, read_write> ff_flow_x: array<f32>;
@group(0) @binding(31) var<storage, read_write> ff_flow_y: array<f32>;
@group(0) @binding(32) var<uniform> ff_params: FlowFieldParams;

@compute @workgroup_size(256)
fn flow_field_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    let total: u32 = ff_params.width * ff_params.width * ff_params.num_channels;
    if (i >= total) { return; }
    let c: u32 = i / (ff_params.width * ff_params.width);
    let pixel: u32 = i % (ff_params.width * ff_params.width);
    let a: f32 = ff_channels[i];
    let alpha: f32 = clamp((a / ff_params.num_channels_f32) * (a / ff_params.num_channels_f32), 0.0, 1.0);
    let nux: f32 = ff_nabla_u_x[i];
    let nuy: f32 = ff_nabla_u_y[i];
    let nax: f32 = ff_nabla_a_x[pixel];
    let nay: f32 = ff_nabla_a_y[pixel];
    ff_flow_x[i] = nux * (1.0 - alpha) - nax * alpha;
    ff_flow_y[i] = nuy * (1.0 - alpha) - nay * alpha;
}

// ---------------------------------------------------------------------------
// reintegration: semi-Lagrangian advection  (bindings 33-37)
// ---------------------------------------------------------------------------
struct ReintegrationParams {
    width: u32, height: u32, dd: i32, sigma: f32, dt: f32,
    num_channels: u32, ma: f32, basal_rate: f32, kinetic_cost: f32,
}

@group(0) @binding(33) var<storage, read> ri_channel: array<f32>;
@group(0) @binding(34) var<storage, read> ri_flow_x: array<f32>;
@group(0) @binding(35) var<storage, read> ri_flow_y: array<f32>;
@group(0) @binding(36) var<storage, read_write> ri_new_channel: array<f32>;
@group(0) @binding(37) var<uniform> ri_params: ReintegrationParams;

@compute @workgroup_size(256)
fn reintegration_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx: u32 = id.x;
    let total: u32 = ri_params.width * ri_params.height * ri_params.num_channels;
    if (idx >= total) { return; }
    let c: u32 = idx / (ri_params.width * ri_params.height);
    let pixel: u32 = idx % (ri_params.width * ri_params.height);
    let x: u32 = pixel % ri_params.width;
    let y: u32 = pixel / ri_params.width;
    let pos_x: f32 = f32(x) + 0.5;
    let pos_y: f32 = f32(y) + 0.5;
    let dd: i32 = ri_params.dd;
    let sigma: f32 = ri_params.sigma;
    let dt: f32 = ri_params.dt;
    let ma: f32 = ri_params.ma;
    let w: u32 = ri_params.width;
    let h: u32 = ri_params.height;
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
            let a: f32 = ri_channel[c_base + n_idx];
            if (a <= 0.0) { continue; }
            let n_pos_x: f32 = f32(nx) + 0.5;
            let n_pos_y: f32 = f32(ny) + 0.5;
            let fx: f32 = clamp(ri_flow_x[c_base + n_idx], -ma, ma);
            let fy: f32 = clamp(ri_flow_y[c_base + n_idx], -ma, ma);
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
    let fx_self: f32 = clamp(ri_flow_x[idx], -ma, ma);
    let fy_self: f32 = clamp(ri_flow_y[idx], -ma, ma);
    let flow_mag: f32 = sqrt(fx_self * fx_self + fy_self * fy_self);
    sum = sum * (1.0 - ri_params.basal_rate * dt) - ri_params.kinetic_cost * flow_mag * dt;
    ri_new_channel[idx] = max(sum, 0.0);
}
