// =============================================================================
// compute_64.wgsl — All compute shaders for the MaceLenia simulation.
// 64x64 grid variant: 16 threads process 64 elements per workgroup.
// =============================================================================

// ---------------------------------------------------------------------------
// Stockham FFT: single-pass row/column using shared memory  (bindings 0, 2-3)
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read_write> fft_data: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> twiddles: array<vec2<f32>>;

struct FftParams { width: u32, inverse: u32 }
@group(0) @binding(3) var<uniform> fft_params: FftParams;

@group(0) @binding(41) var<storage, read> bitrev_lut: array<u32>;

var<workgroup> ping: array<vec2<f32>, 64>;
var<workgroup> pong: array<vec2<f32>, 64>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn load_twiddle(stage: u32, k: u32) -> vec2<f32> {
    let stage_offset = (1u << stage) - 1u;
    let w = twiddles[stage_offset + k];
    return select(w, vec2<f32>(w.x, -w.y), fft_params.inverse == 1u);
}

fn stockham_butterfly(stage: u32, t: u32, src: ptr<workgroup, array<vec2<f32>, 64>>, dst: ptr<workgroup, array<vec2<f32>, 64>>) {
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

@compute @workgroup_size(16)
fn fft_row_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let row = wg_id.x;
    let base = row * fft_params.width;
    let t = local_id.x;

    ping[bitrev_lut[t]]       = fft_data[base + t];
    ping[bitrev_lut[t + 16u]] = fft_data[base + t + 16u];
    ping[bitrev_lut[t + 32u]] = fft_data[base + t + 32u];
    ping[bitrev_lut[t + 48u]] = fft_data[base + t + 48u];
    workgroupBarrier();

    stockham_butterfly(0u, t, &ping, &pong);
    stockham_butterfly(0u, t + 16u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(1u, t, &pong, &ping);
    stockham_butterfly(1u, t + 16u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(2u, t, &ping, &pong);
    stockham_butterfly(2u, t + 16u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(3u, t, &pong, &ping);
    stockham_butterfly(3u, t + 16u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(4u, t, &ping, &pong);
    stockham_butterfly(4u, t + 16u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(5u, t, &pong, &ping);
    stockham_butterfly(5u, t + 16u, &pong, &ping);

    fft_data[base + t]        = ping[t];
    fft_data[base + t + 16u]  = ping[t + 16u];
    fft_data[base + t + 32u]  = ping[t + 32u];
    fft_data[base + t + 48u]  = ping[t + 48u];
}

@compute @workgroup_size(16)
fn fft_col_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let col = wg_id.x;
    let w = fft_params.width;
    let t = local_id.x;

    ping[bitrev_lut[t]]       = fft_data[t * w + col];
    ping[bitrev_lut[t + 16u]] = fft_data[(t + 16u) * w + col];
    ping[bitrev_lut[t + 32u]] = fft_data[(t + 32u) * w + col];
    ping[bitrev_lut[t + 48u]] = fft_data[(t + 48u) * w + col];
    workgroupBarrier();

    stockham_butterfly(0u, t, &ping, &pong);
    stockham_butterfly(0u, t + 16u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(1u, t, &pong, &ping);
    stockham_butterfly(1u, t + 16u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(2u, t, &ping, &pong);
    stockham_butterfly(2u, t + 16u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(3u, t, &pong, &ping);
    stockham_butterfly(3u, t + 16u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(4u, t, &ping, &pong);
    stockham_butterfly(4u, t + 16u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(5u, t, &pong, &ping);
    stockham_butterfly(5u, t + 16u, &pong, &ping);

    fft_data[t * w + col]         = ping[t];
    fft_data[(t + 16u) * w + col] = ping[t + 16u];
    fft_data[(t + 32u) * w + col] = ping[t + 32u];
    fft_data[(t + 48u) * w + col] = ping[t + 48u];
}

@compute @workgroup_size(16)
fn ifft_row_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let row = wg_id.x;
    let base = row * fft_params.width;
    let t = local_id.x;

    ping[bitrev_lut[t]]       = fft_data[base + t];
    ping[bitrev_lut[t + 16u]] = fft_data[base + t + 16u];
    ping[bitrev_lut[t + 32u]] = fft_data[base + t + 32u];
    ping[bitrev_lut[t + 48u]] = fft_data[base + t + 48u];
    workgroupBarrier();

    stockham_butterfly(0u, t, &ping, &pong);
    stockham_butterfly(0u, t + 16u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(1u, t, &pong, &ping);
    stockham_butterfly(1u, t + 16u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(2u, t, &ping, &pong);
    stockham_butterfly(2u, t + 16u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(3u, t, &pong, &ping);
    stockham_butterfly(3u, t + 16u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(4u, t, &ping, &pong);
    stockham_butterfly(4u, t + 16u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(5u, t, &pong, &ping);
    stockham_butterfly(5u, t + 16u, &pong, &ping);

    fft_data[base + t]        = ping[t];
    fft_data[base + t + 16u]  = ping[t + 16u];
    fft_data[base + t + 32u]  = ping[t + 32u];
    fft_data[base + t + 48u]  = ping[t + 48u];
}

@compute @workgroup_size(16)
fn ifft_col_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let col = wg_id.x;
    let w = fft_params.width;
    let t = local_id.x;

    ping[bitrev_lut[t]]       = fft_data[t * w + col];
    ping[bitrev_lut[t + 16u]] = fft_data[(t + 16u) * w + col];
    ping[bitrev_lut[t + 32u]] = fft_data[(t + 32u) * w + col];
    ping[bitrev_lut[t + 48u]] = fft_data[(t + 48u) * w + col];
    workgroupBarrier();

    stockham_butterfly(0u, t, &ping, &pong);
    stockham_butterfly(0u, t + 16u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(1u, t, &pong, &ping);
    stockham_butterfly(1u, t + 16u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(2u, t, &ping, &pong);
    stockham_butterfly(2u, t + 16u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(3u, t, &pong, &ping);
    stockham_butterfly(3u, t + 16u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(4u, t, &ping, &pong);
    stockham_butterfly(4u, t + 16u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(5u, t, &pong, &ping);
    stockham_butterfly(5u, t + 16u, &pong, &ping);

    fft_data[t * w + col]         = ping[t];
    fft_data[(t + 16u) * w + col] = ping[t + 16u];
    fft_data[(t + 32u) * w + col] = ping[t + 32u];
    fft_data[(t + 48u) * w + col] = ping[t + 48u];
}

// Fused complex multiply + IFFT column pass
@compute @workgroup_size(16)
fn fused_cmul_ifft_col_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let col = wg_id.x;
    let w = fft_params.width;
    let t = local_id.x;

    fft_data[t * w + col]         = complex_mul(fft_data[t * w + col], cm_kernel[t * w + col]);
    fft_data[(t + 16u) * w + col] = complex_mul(fft_data[(t + 16u) * w + col], cm_kernel[(t + 16u) * w + col]);
    fft_data[(t + 32u) * w + col] = complex_mul(fft_data[(t + 32u) * w + col], cm_kernel[(t + 32u) * w + col]);
    fft_data[(t + 48u) * w + col] = complex_mul(fft_data[(t + 48u) * w + col], cm_kernel[(t + 48u) * w + col]);
    workgroupBarrier();

    ping[bitrev_lut[t]]       = fft_data[t * w + col];
    ping[bitrev_lut[t + 16u]] = fft_data[(t + 16u) * w + col];
    ping[bitrev_lut[t + 32u]] = fft_data[(t + 32u) * w + col];
    ping[bitrev_lut[t + 48u]] = fft_data[(t + 48u) * w + col];
    workgroupBarrier();

    stockham_butterfly(0u, t, &ping, &pong);
    stockham_butterfly(0u, t + 16u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(1u, t, &pong, &ping);
    stockham_butterfly(1u, t + 16u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(2u, t, &ping, &pong);
    stockham_butterfly(2u, t + 16u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(3u, t, &pong, &ping);
    stockham_butterfly(3u, t + 16u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(4u, t, &ping, &pong);
    stockham_butterfly(4u, t + 16u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(5u, t, &pong, &ping);
    stockham_butterfly(5u, t + 16u, &pong, &ping);

    fft_data[t * w + col]         = ping[t];
    fft_data[(t + 16u) * w + col] = ping[t + 16u];
    fft_data[(t + 32u) * w + col] = ping[t + 32u];
    fft_data[(t + 48u) * w + col] = ping[t + 48u];
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
// mcl_growth: growth function + weighted sum + Euler step  (bindings 8-14)
// ---------------------------------------------------------------------------
struct MclParams {
    width: u32,
    num_kernels: u32,
    num_channels: u32,
    dt: f32,
    norm_factor: f32,
}

@group(0) @binding(8) var<storage, read> mcl_conv: array<vec2<f32>>;
@group(0) @binding(9) var<storage, read> mcl_channel: array<f32>;
@group(0) @binding(10) var<storage, read_write> mcl_new_channel: array<f32>;
@group(0) @binding(11) var<storage, read> mcl_growth_params: array<vec2<f32>>;
@group(0) @binding(12) var<storage, read> mcl_weights: array<f32>;
@group(0) @binding(13) var<storage, read> mcl_c1: array<u32>;
@group(0) @binding(14) var<uniform> mcl_params: MclParams;

@compute @workgroup_size(256)
fn mcl_growth_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p: u32 = id.x;
    let total_pixels: u32 = mcl_params.width * mcl_params.width;
    if (p >= total_pixels) { return; }

    var dx: array<f32, 16>;
    for (var c: u32 = 0u; c < mcl_params.num_channels; c = c + 1u) {
        dx[c] = 0.0;
    }

    for (var k: u32 = 0u; k < mcl_params.num_kernels; k = k + 1u) {
        let u_val: f32 = mcl_conv[k * total_pixels + p].x * mcl_params.norm_factor;
        let gp: vec2<f32> = mcl_growth_params[k];
        let diff: f32 = u_val - gp.x;
        let g: f32 = 2.0 * exp(-(diff * diff) / (2.0 * gp.y * gp.y)) - 1.0;
        let out_ch: u32 = mcl_c1[k];
        dx[out_ch] = dx[out_ch] + g * mcl_weights[k];
    }

    for (var c: u32 = 0u; c < mcl_params.num_channels; c = c + 1u) {
        let old_val: f32 = mcl_channel[c * total_pixels + p];
        let new_val: f32 = min(max(old_val + mcl_params.dt * dx[c], 0.0), 1.0);
        mcl_new_channel[c * total_pixels + p] = new_val;
    }
}
