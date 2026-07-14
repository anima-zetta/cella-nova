// =============================================================================
// compute_512.wgsl — All compute shaders for the Flow Lenia simulation.
// 512x512 grid variant: 32 threads process 512 elements per workgroup.
//
// Each entry point uses unique (group=0, binding) pairs so they can coexist
// in one module without conflicts.
// =============================================================================

// ---------------------------------------------------------------------------
// Stockham FFT: single-pass row/column using shared memory  (bindings 0, 2-3)
// 32 threads cooperatively process 512 elements entirely in workgroup memory.
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read_write> fft_data: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> twiddles: array<vec2<f32>>;

struct FftParams { width: u32, inverse: u32 }
@group(0) @binding(3) var<uniform> fft_params: FftParams;

// Precomputed 9-bit reversal permutation (0..511), uploaded once by the host.
// Replaces the per-invocation bit_reverse() loop with a single cached buffer read.
@group(0) @binding(41) var<storage, read> bitrev_lut: array<u32>;

var<workgroup> ping: array<vec2<f32>, 512>;
var<workgroup> pong: array<vec2<f32>, 512>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn load_twiddle(stage: u32, k: u32) -> vec2<f32> {
    let stage_offset = (1u << stage) - 1u;
    let w = twiddles[stage_offset + k];
    return select(w, vec2<f32>(w.x, -w.y), fft_params.inverse == 1u);
}

fn stockham_butterfly(stage: u32, t: u32, src: ptr<workgroup, array<vec2<f32>, 512>>, dst: ptr<workgroup, array<vec2<f32>, 512>>) {
    let R = 1u << stage;
    let b = t / R;
    let k = t % R;
    let src0 = b * 2u * R + k;
    let src1 = src0 + R;
    let w = load_twiddle(stage, k);
    let even = (*src)[src0];
    let odd = complex_mul(w, (*src)[src1]);
    (*dst)[src0] = even + odd;
    (*dst)[src1] = even - odd;
}

@compute @workgroup_size(128)
fn fft_row_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let row = wg_id.x;
    let base = row * fft_params.width;
    let t = local_id.x;

    let rev_t   = bitrev_lut[t];
    let rev_t2  = bitrev_lut[t + 128u];
    let rev_t3  = bitrev_lut[t + 256u];
    let rev_t4  = bitrev_lut[t + 384u];
    ping[rev_t]  = fft_data[base + t];
    ping[rev_t2] = fft_data[base + t + 128u];
    ping[rev_t3] = fft_data[base + t + 256u];
    ping[rev_t4] = fft_data[base + t + 384u];
    workgroupBarrier();

    stockham_butterfly(0u, t + 0u, &ping, &pong);
    stockham_butterfly(0u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(1u, t + 0u, &pong, &ping);
    stockham_butterfly(1u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(2u, t + 0u, &ping, &pong);
    stockham_butterfly(2u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(3u, t + 0u, &pong, &ping);
    stockham_butterfly(3u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(4u, t + 0u, &ping, &pong);
    stockham_butterfly(4u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(5u, t + 0u, &pong, &ping);
    stockham_butterfly(5u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(6u, t + 0u, &ping, &pong);
    stockham_butterfly(6u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(7u, t + 0u, &pong, &ping);
    stockham_butterfly(7u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(8u, t + 0u, &ping, &pong);
    stockham_butterfly(8u, t + 128u, &ping, &pong);

    fft_data[base + t]          = pong[t];
    fft_data[base + t + 128u]   = pong[t + 128u];
    fft_data[base + t + 256u]   = pong[t + 256u];
    fft_data[base + t + 384u]   = pong[t + 384u];
}

@compute @workgroup_size(128)
fn fft_col_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let col = wg_id.x;
    let w = fft_params.width;
    let t = local_id.x;

    let rev_t   = bitrev_lut[t];
    let rev_t2  = bitrev_lut[t + 128u];
    let rev_t3  = bitrev_lut[t + 256u];
    let rev_t4  = bitrev_lut[t + 384u];
    ping[rev_t]  = fft_data[t * w + col];
    ping[rev_t2] = fft_data[(t + 128u) * w + col];
    ping[rev_t3] = fft_data[(t + 256u) * w + col];
    ping[rev_t4] = fft_data[(t + 384u) * w + col];
    workgroupBarrier();

    stockham_butterfly(0u, t + 0u, &ping, &pong);
    stockham_butterfly(0u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(1u, t + 0u, &pong, &ping);
    stockham_butterfly(1u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(2u, t + 0u, &ping, &pong);
    stockham_butterfly(2u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(3u, t + 0u, &pong, &ping);
    stockham_butterfly(3u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(4u, t + 0u, &ping, &pong);
    stockham_butterfly(4u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(5u, t + 0u, &pong, &ping);
    stockham_butterfly(5u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(6u, t + 0u, &ping, &pong);
    stockham_butterfly(6u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(7u, t + 0u, &pong, &ping);
    stockham_butterfly(7u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(8u, t + 0u, &ping, &pong);
    stockham_butterfly(8u, t + 128u, &ping, &pong);

    fft_data[t * w + col]            = pong[t];
    fft_data[(t + 128u) * w + col]   = pong[t + 128u];
    fft_data[(t + 256u) * w + col]   = pong[t + 256u];
    fft_data[(t + 384u) * w + col]   = pong[t + 384u];
}

// Fused complex multiply + IFFT column pass.
// Saves 1 dispatch per kernel by combining cmul and IFFT col.
// Uses bindings 0 (data), 2 (twiddles), 3 (params), 7 (kernel), 41 (bitrev).
@compute @workgroup_size(128)
fn fused_cmul_ifft_col_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let col = wg_id.x;
    let w = fft_params.width;
    let t = local_id.x;

    // Step 1: Complex multiply for elements in this column
    fft_data[t * w + col] = complex_mul(fft_data[t * w + col], cm_kernel[t * w + col]);
    fft_data[(t + 128u) * w + col] = complex_mul(fft_data[(t + 128u) * w + col], cm_kernel[(t + 128u) * w + col]);
    fft_data[(t + 256u) * w + col] = complex_mul(fft_data[(t + 256u) * w + col], cm_kernel[(t + 256u) * w + col]);
    fft_data[(t + 384u) * w + col] = complex_mul(fft_data[(t + 384u) * w + col], cm_kernel[(t + 384u) * w + col]);
    workgroupBarrier();

    // Step 2: IFFT column pass (same as fft_col_main with inverse=1)
    let rev_t   = bitrev_lut[t];
    let rev_t2  = bitrev_lut[t + 128u];
    let rev_t3  = bitrev_lut[t + 256u];
    let rev_t4  = bitrev_lut[t + 384u];
    ping[rev_t]  = fft_data[t * w + col];
    ping[rev_t2] = fft_data[(t + 128u) * w + col];
    ping[rev_t3] = fft_data[(t + 256u) * w + col];
    ping[rev_t4] = fft_data[(t + 384u) * w + col];
    workgroupBarrier();

    stockham_butterfly(0u, t + 0u, &ping, &pong);
    stockham_butterfly(0u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(1u, t + 0u, &pong, &ping);
    stockham_butterfly(1u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(2u, t + 0u, &ping, &pong);
    stockham_butterfly(2u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(3u, t + 0u, &pong, &ping);
    stockham_butterfly(3u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(4u, t + 0u, &ping, &pong);
    stockham_butterfly(4u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(5u, t + 0u, &pong, &ping);
    stockham_butterfly(5u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(6u, t + 0u, &ping, &pong);
    stockham_butterfly(6u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(7u, t + 0u, &pong, &ping);
    stockham_butterfly(7u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(8u, t + 0u, &ping, &pong);
    stockham_butterfly(8u, t + 128u, &ping, &pong);

    fft_data[t * w + col]            = pong[t];
    fft_data[(t + 128u) * w + col]   = pong[t + 128u];
    fft_data[(t + 256u) * w + col]   = pong[t + 256u];
    fft_data[(t + 384u) * w + col]   = pong[t + 384u];
}

// ---------------------------------------------------------------------------
// copy_to_conv: real channel -> complex conv buffer  (bindings 4-5)
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
// cmul_from_saved: element-wise conv = saved_conv * kernel  (bindings 5, 7)
// Reads from saved frequency data (binding 5), writes to conv (binding 6).
// Eliminates the restore buffer copy by combining restore + cmul into one pass.
// ---------------------------------------------------------------------------
@group(0) @binding(5) var<storage, read_write> cms_conv: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read> cms_kernel: array<vec2<f32>>;
@group(0) @binding(44) var<storage, read> cms_saved: array<vec2<f32>>;

@compute @workgroup_size(256)
fn cmul_from_saved_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    if (i >= arrayLength(&cms_conv)) { return; }
    cms_conv[i] = complex_mul(cms_saved[i], cms_kernel[i]);
}

// ---------------------------------------------------------------------------
// normalize_growth: normalize IFFT + apply growth  (bindings 8-12)
// ---------------------------------------------------------------------------
struct NormalizeParams { norm_factor: f32 }
struct GrowthParams { m: f32, s: f32, h: f32 }

@group(0) @binding(8) var<storage, read_write> ng_data: array<vec2<f32>>;
@group(0) @binding(9) var<storage, read_write> ng_result: array<f32>;
@group(0) @binding(11) var<uniform> ng_norm_params: NormalizeParams;
@group(0) @binding(12) var<storage, read> ng_growth_params: GrowthParams;
@group(0) @binding(13) var<storage, read> ng_params_field: array<f32>;

@compute @workgroup_size(256)
fn normalize_growth_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    if (i >= arrayLength(&ng_result)) { return; }
    ng_data[i] = ng_data[i] * ng_norm_params.norm_factor;
    let x: f32 = ng_data[i].x;
    let diff: f32 = x - ng_growth_params.m;
    let g: f32 = exp(-(diff * diff) / (2.0 * ng_growth_params.s * ng_growth_params.s));
    ng_result[i] = (2.0 * g - 1.0) * ng_growth_params.h * ng_params_field[i];
}

// ---------------------------------------------------------------------------
// channel_aggregate: per-kernel growth -> per-channel  (bindings 14-18)
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
// sum_channels: sum all channels into one field  (bindings 19-21)
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
// sobel: 3x3 gradient  (bindings 22-25)
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
// flow_field: compute flow from gradients  (bindings 26-33)
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
// reintegration: semi-Lagrangian advection  (bindings 34-40)
// ---------------------------------------------------------------------------
struct ReintegrationParams {
    width: u32, height: u32, dd: i32, sigma: f32, dt: f32,
    num_channels: u32, num_kernels: u32, ma: f32,
}

@group(0) @binding(33) var<storage, read> ri_channel: array<f32>;
@group(0) @binding(34) var<storage, read> ri_flow_x: array<f32>;
@group(0) @binding(35) var<storage, read> ri_flow_y: array<f32>;
@group(0) @binding(36) var<storage, read_write> ri_new_channel: array<f32>;
@group(0) @binding(37) var<uniform> ri_params: ReintegrationParams;
@group(0) @binding(38) var<storage, read> ri_params_field: array<f32>;
@group(0) @binding(39) var<storage, read_write> ri_new_params_field: array<f32>;

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
            let n_pos_x: f32 = f32(nx) + 0.5;
            let n_pos_y: f32 = f32(ny) + 0.5;
            let fx: f32 = ri_flow_x[c_base + n_idx];
            let fy: f32 = ri_flow_y[c_base + n_idx];
            let mu_x: f32 = clamp(n_pos_x + clamp(fx * dt, -ma, ma), sigma, f32(w) - sigma);
            let mu_y: f32 = clamp(n_pos_y + clamp(fy * dt, -ma, ma), sigma, f32(h) - sigma);
            let dpx: f32 = abs(pos_x - mu_x);
            let dpy: f32 = abs(pos_y - mu_y);
            let sz_x: f32 = clamp(0.5 - dpx + sigma, 0.0, max_sz);
            let sz_y: f32 = clamp(0.5 - dpy + sigma, 0.0, max_sz);
            let area: f32 = (sz_x * sz_y) / area_norm;
            sum = sum + a * area;
        }
    }
    ri_new_channel[idx] = max(sum, 0.0);
}

// ---------------------------------------------------------------------------
// reintegration_params: semi-Lagrangian advection of parameter field
// Uses the total mass (sum of all channels) as the weighting factor.
// (bindings 33-35, 37-40)
// ---------------------------------------------------------------------------

@group(0) @binding(40) var<storage, read> ri_sum_a: array<f32>;

@compute @workgroup_size(256)
fn reintegration_params_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx: u32 = id.x + id.y * 16776960u;
    let total: u32 = ri_params.width * ri_params.height * ri_params.num_kernels;
    if (idx >= total) { return; }
    let k: u32 = idx / (ri_params.width * ri_params.height);
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
    let k_base: u32 = k * w * h;
    var sum_val: f32 = 0.0;
    var total_weight: f32 = 0.0;
    for (var dx: i32 = -dd; dx <= dd; dx = dx + 1) {
        for (var dy: i32 = -dd; dy <= dd; dy = dy + 1) {
            let nx: i32 = i32(x) + dx;
            let ny: i32 = i32(y) + dy;
            if (nx < 0 || nx >= w_i32 || ny < 0 || ny >= h_i32) { continue; }
            let n_idx: u32 = u32(ny) * w + u32(nx);
            let a: f32 = ri_sum_a[n_idx];
            let n_pos_x: f32 = f32(nx) + 0.5;
            let n_pos_y: f32 = f32(ny) + 0.5;
            let fx: f32 = ri_flow_x[n_idx];
            let fy: f32 = ri_flow_y[n_idx];
            let mu_x: f32 = clamp(n_pos_x + clamp(fx * dt, -ma, ma), sigma, f32(w) - sigma);
            let mu_y: f32 = clamp(n_pos_y + clamp(fy * dt, -ma, ma), sigma, f32(h) - sigma);
            let dpx: f32 = abs(pos_x - mu_x);
            let dpy: f32 = abs(pos_y - mu_y);
            let sz_x: f32 = clamp(0.5 - dpx + sigma, 0.0, max_sz);
            let sz_y: f32 = clamp(0.5 - dpy + sigma, 0.0, max_sz);
            let area: f32 = (sz_x * sz_y) / area_norm;
            sum_val = sum_val + ri_params_field[k_base + n_idx] * a * area;
            total_weight = total_weight + a * area;
        }
    }
    if (total_weight > 0.0) {
        ri_new_params_field[idx] = sum_val / total_weight;
    } else {
        ri_new_params_field[idx] = ri_params_field[idx];
    }
}
