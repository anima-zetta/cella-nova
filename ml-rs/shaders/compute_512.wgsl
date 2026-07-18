// =============================================================================
// compute_512.wgsl — All compute shaders for the MaceLenia simulation.
// 512x512 grid variant: 128 threads process 512 elements per workgroup.
//
// Each entry point uses unique (group=0, binding) pairs so they can coexist
// in one module without conflicts.
// =============================================================================

// ---------------------------------------------------------------------------
// Stockham FFT: single-pass row/column using shared memory  (bindings 0, 2-3)
// 128 threads cooperatively process 512 elements entirely in workgroup memory.
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read_write> fft_data: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> twiddles: array<vec2<f32>>;

struct FftParams { width: u32, inverse: u32 }
@group(0) @binding(3) var<uniform> fft_params: FftParams;

// Precomputed bit-reversal permutation, uploaded once by the host.
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

    ping[bitrev_lut[t]]          = fft_data[base + t];
    ping[bitrev_lut[t + 128u]]   = fft_data[base + t + 128u];
    ping[bitrev_lut[t + 256u]]   = fft_data[base + t + 256u];
    ping[bitrev_lut[t + 384u]]   = fft_data[base + t + 384u];
    workgroupBarrier();

    stockham_butterfly(0u, t, &ping, &pong);
    stockham_butterfly(0u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(1u, t, &pong, &ping);
    stockham_butterfly(1u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(2u, t, &ping, &pong);
    stockham_butterfly(2u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(3u, t, &pong, &ping);
    stockham_butterfly(3u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(4u, t, &ping, &pong);
    stockham_butterfly(4u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(5u, t, &pong, &ping);
    stockham_butterfly(5u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(6u, t, &ping, &pong);
    stockham_butterfly(6u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(7u, t, &pong, &ping);
    stockham_butterfly(7u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(8u, t, &ping, &pong);
    stockham_butterfly(8u, t + 128u, &ping, &pong);

    fft_data[base + t]            = pong[t];
    fft_data[base + t + 128u]    = pong[t + 128u];
    fft_data[base + t + 256u]    = pong[t + 256u];
    fft_data[base + t + 384u]    = pong[t + 384u];
}

@compute @workgroup_size(128)
fn fft_col_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let col = wg_id.x;
    let w = fft_params.width;
    let t = local_id.x;

    ping[bitrev_lut[t]]            = fft_data[t * w + col];
    ping[bitrev_lut[t + 128u]]     = fft_data[(t + 128u) * w + col];
    ping[bitrev_lut[t + 256u]]     = fft_data[(t + 256u) * w + col];
    ping[bitrev_lut[t + 384u]]     = fft_data[(t + 384u) * w + col];
    workgroupBarrier();

    stockham_butterfly(0u, t, &ping, &pong);
    stockham_butterfly(0u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(1u, t, &pong, &ping);
    stockham_butterfly(1u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(2u, t, &ping, &pong);
    stockham_butterfly(2u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(3u, t, &pong, &ping);
    stockham_butterfly(3u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(4u, t, &ping, &pong);
    stockham_butterfly(4u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(5u, t, &pong, &ping);
    stockham_butterfly(5u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(6u, t, &ping, &pong);
    stockham_butterfly(6u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(7u, t, &pong, &ping);
    stockham_butterfly(7u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(8u, t, &ping, &pong);
    stockham_butterfly(8u, t + 128u, &ping, &pong);

    fft_data[t * w + col]            = pong[t];
    fft_data[(t + 128u) * w + col]   = pong[t + 128u];
    fft_data[(t + 256u) * w + col]   = pong[t + 256u];
    fft_data[(t + 384u) * w + col]   = pong[t + 384u];
}

@compute @workgroup_size(128)
fn ifft_row_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let row = wg_id.x;
    let base = row * fft_params.width;
    let t = local_id.x;

    ping[bitrev_lut[t]]          = fft_data[base + t];
    ping[bitrev_lut[t + 128u]]   = fft_data[base + t + 128u];
    ping[bitrev_lut[t + 256u]]   = fft_data[base + t + 256u];
    ping[bitrev_lut[t + 384u]]   = fft_data[base + t + 384u];
    workgroupBarrier();

    stockham_butterfly(0u, t, &ping, &pong);
    stockham_butterfly(0u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(1u, t, &pong, &ping);
    stockham_butterfly(1u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(2u, t, &ping, &pong);
    stockham_butterfly(2u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(3u, t, &pong, &ping);
    stockham_butterfly(3u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(4u, t, &ping, &pong);
    stockham_butterfly(4u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(5u, t, &pong, &ping);
    stockham_butterfly(5u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(6u, t, &ping, &pong);
    stockham_butterfly(6u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(7u, t, &pong, &ping);
    stockham_butterfly(7u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(8u, t, &ping, &pong);
    stockham_butterfly(8u, t + 128u, &ping, &pong);

    fft_data[base + t]            = pong[t];
    fft_data[base + t + 128u]    = pong[t + 128u];
    fft_data[base + t + 256u]    = pong[t + 256u];
    fft_data[base + t + 384u]    = pong[t + 384u];
}

@compute @workgroup_size(128)
fn ifft_col_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let col = wg_id.x;
    let w = fft_params.width;
    let t = local_id.x;

    ping[bitrev_lut[t]]            = fft_data[t * w + col];
    ping[bitrev_lut[t + 128u]]     = fft_data[(t + 128u) * w + col];
    ping[bitrev_lut[t + 256u]]     = fft_data[(t + 256u) * w + col];
    ping[bitrev_lut[t + 384u]]     = fft_data[(t + 384u) * w + col];
    workgroupBarrier();

    stockham_butterfly(0u, t, &ping, &pong);
    stockham_butterfly(0u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(1u, t, &pong, &ping);
    stockham_butterfly(1u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(2u, t, &ping, &pong);
    stockham_butterfly(2u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(3u, t, &pong, &ping);
    stockham_butterfly(3u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(4u, t, &ping, &pong);
    stockham_butterfly(4u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(5u, t, &pong, &ping);
    stockham_butterfly(5u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(6u, t, &ping, &pong);
    stockham_butterfly(6u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(7u, t, &pong, &ping);
    stockham_butterfly(7u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(8u, t, &ping, &pong);
    stockham_butterfly(8u, t + 128u, &ping, &pong);

    fft_data[t * w + col]            = pong[t];
    fft_data[(t + 128u) * w + col]   = pong[t + 128u];
    fft_data[(t + 256u) * w + col]   = pong[t + 256u];
    fft_data[(t + 384u) * w + col]   = pong[t + 384u];
}

// Fused complex multiply + IFFT column pass
@compute @workgroup_size(128)
fn fused_cmul_ifft_col_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let col = wg_id.x;
    let w = fft_params.width;
    let t = local_id.x;

    fft_data[t * w + col]              = complex_mul(fft_data[t * w + col], cm_kernel[t * w + col]);
    fft_data[(t + 128u) * w + col]     = complex_mul(fft_data[(t + 128u) * w + col], cm_kernel[(t + 128u) * w + col]);
    fft_data[(t + 256u) * w + col]     = complex_mul(fft_data[(t + 256u) * w + col], cm_kernel[(t + 256u) * w + col]);
    fft_data[(t + 384u) * w + col]     = complex_mul(fft_data[(t + 384u) * w + col], cm_kernel[(t + 384u) * w + col]);
    workgroupBarrier();

    ping[bitrev_lut[t]]            = fft_data[t * w + col];
    ping[bitrev_lut[t + 128u]]     = fft_data[(t + 128u) * w + col];
    ping[bitrev_lut[t + 256u]]     = fft_data[(t + 256u) * w + col];
    ping[bitrev_lut[t + 384u]]     = fft_data[(t + 384u) * w + col];
    workgroupBarrier();

    stockham_butterfly(0u, t, &ping, &pong);
    stockham_butterfly(0u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(1u, t, &pong, &ping);
    stockham_butterfly(1u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(2u, t, &ping, &pong);
    stockham_butterfly(2u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(3u, t, &pong, &ping);
    stockham_butterfly(3u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(4u, t, &ping, &pong);
    stockham_butterfly(4u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(5u, t, &pong, &ping);
    stockham_butterfly(5u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(6u, t, &ping, &pong);
    stockham_butterfly(6u, t + 128u, &ping, &pong);
    workgroupBarrier();
    stockham_butterfly(7u, t, &pong, &ping);
    stockham_butterfly(7u, t + 128u, &pong, &ping);
    workgroupBarrier();
    stockham_butterfly(8u, t, &ping, &pong);
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
// mcl_growth: growth function + weighted sum -> affinity  (bindings 8-14)
// ---------------------------------------------------------------------------
struct MclParams {
    width: u32,
    num_kernels: u32,
    num_channels: u32,
    dt: f32,
    norm_factor: f32,
}

@group(0) @binding(8) var<storage, read> mcl_conv: array<vec2<f32>>;
@group(0) @binding(10) var<storage, read_write> mcl_affinity: array<f32>;
@group(0) @binding(11) var<storage, read> mcl_growth_params: array<vec2<f32>>;
@group(0) @binding(12) var<storage, read> mcl_weights: array<f32>;
@group(0) @binding(13) var<storage, read> mcl_c1: array<u32>;
@group(0) @binding(14) var<uniform> mcl_params: MclParams;

@compute @workgroup_size(256)
fn mcl_growth_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p: u32 = id.x;
    let total_pixels: u32 = mcl_params.width * mcl_params.width;
    if (p >= total_pixels) { return; }

    // Accumulate growth contribution for each output channel
    // (max 16 channels supported)
    var dx: array<f32, 16>;
    for (var c: u32 = 0u; c < mcl_params.num_channels; c = c + 1u) {
        dx[c] = 0.0;
    }

    // For each kernel: growth -> weighted sum
    for (var k: u32 = 0u; k < mcl_params.num_kernels; k = k + 1u) {
        let u_val: f32 = mcl_conv[k * total_pixels + p].x * mcl_params.norm_factor;
        let gp: vec2<f32> = mcl_growth_params[k];
        let diff: f32 = u_val - gp.x;
        let g: f32 = 2.0 * exp(-(diff * diff) / (2.0 * gp.y * gp.y)) - 1.0;
        let out_ch: u32 = mcl_c1[k];
        dx[out_ch] = dx[out_ch] + g * mcl_weights[k];
    }

    for (var c: u32 = 0u; c < mcl_params.num_channels; c = c + 1u) {
        mcl_affinity[c * total_pixels + p] = dx[c];
    }
}

// ---------------------------------------------------------------------------
// DiffusionLenia: affinity exponential + local normalization  (bindings 15-17)
// Pass 1: compute aff_exp = exp(temp * affinity) and Z = 3x3 sum of aff_exp
// ---------------------------------------------------------------------------
struct DiffusionParams {
    width: u32,
    num_channels: u32,
    temp: f32,
}

@group(0) @binding(15) var<storage, read_write> diff_affinity: array<f32>;
@group(0) @binding(16) var<storage, read_write> diff_Z: array<f32>;
@group(0) @binding(17) var<uniform> diff_params: DiffusionParams;

@compute @workgroup_size(256)
fn diffusion_pass1_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p: u32 = id.x;
    let w: u32 = diff_params.width;
    let total_pixels: u32 = w * w;
    if (p >= total_pixels) { return; }

    let px: u32 = p % w;
    let py: u32 = p / w;

    for (var c: u32 = 0u; c < diff_params.num_channels; c = c + 1u) {
        let base: u32 = c * total_pixels;

        // Compute Z = sum of aff_exp over 3x3 neighborhood
        var Z: f32 = 0.0;
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                let nx: u32 = (w + px + u32(dx)) % w;
                let ny: u32 = (w + py + u32(dy)) % w;
                let n: u32 = ny * w + nx;
                let a: f32 = diff_affinity[base + n];
                Z = Z + exp(diff_params.temp * a);
            }
        }

        // Store Z to temp buffer (affinity buffer is NOT modified)
        diff_Z[base + p] = Z;
    }
}

// ---------------------------------------------------------------------------
// DiffusionLenia: mass redistribution  (bindings 15-19)
// Pass 2: new_state[p] = aff_exp[p] * sum over 3x3 of (state[n] / Z[n])
// ---------------------------------------------------------------------------
@group(0) @binding(15) var<storage, read> diff2_affinity: array<f32>;
@group(0) @binding(16) var<storage, read> diff2_Z: array<f32>;
@group(0) @binding(18) var<storage, read> diff2_channel: array<f32>;
@group(0) @binding(19) var<storage, read_write> diff2_new_channel: array<f32>;
@group(0) @binding(17) var<uniform> diff2_params: DiffusionParams;

@compute @workgroup_size(256)
fn diffusion_pass2_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p: u32 = id.x;
    let w: u32 = diff2_params.width;
    let total_pixels: u32 = w * w;
    if (p >= total_pixels) { return; }

    let px: u32 = p % w;
    let py: u32 = p / w;

    for (var c: u32 = 0u; c < diff2_params.num_channels; c = c + 1u) {
        let base: u32 = c * total_pixels;

        // Recompute aff_exp from original affinity (not overwritten by pass 1)
        let a: f32 = diff2_affinity[base + p];
        let aff_exp: f32 = exp(diff2_params.temp * a);

        // Sum state[n] / Z[n] over 3x3 neighborhood
        var sum: f32 = 0.0;
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                let nx: u32 = (w + px + u32(dx)) % w;
                let ny: u32 = (w + py + u32(dy)) % w;
                let n: u32 = ny * w + nx;
                let s: f32 = diff2_channel[base + n];
                let Z_n: f32 = diff2_Z[base + n];
                sum = sum + s / Z_n;
            }
        }

        diff2_new_channel[base + p] = aff_exp * sum;
    }
}
