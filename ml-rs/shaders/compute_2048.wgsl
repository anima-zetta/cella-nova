// =============================================================================
// compute_2048.wgsl — All compute shaders for the MaceLenia simulation.
// 2048x2048 grid variant: 256 threads process 2048 elements per workgroup.
//
// Each entry point uses unique (group=0, binding) pairs so they can coexist
// in one module without conflicts.
// =============================================================================

// ---------------------------------------------------------------------------
// Cooley-Tukey DIT FFT: uses only 256 elements of workgroup memory.
// Stages 0-7 (distance 1..128) run in workgroup memory per 256-element slice.
// Stages 8-10 (distance 256..1024) run in registers (same-thread pairs).
// 256 threads process 2048 elements (8 per thread, strided by 256).
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read_write> fft_data: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> twiddles: array<vec2<f32>>;

struct FftParams { width: u32, inverse: u32 }
@group(0) @binding(3) var<uniform> fft_params: FftParams;

// Precomputed bit-reversal permutation, uploaded once by the host.
@group(0) @binding(41) var<storage, read> bitrev_lut: array<u32>;

var<workgroup> wg_data: array<vec2<f32>, 256>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn load_twiddle(stage: u32, k: u32) -> vec2<f32> {
    let stage_offset = (1u << stage) - 1u;
    let w = twiddles[stage_offset + k];
    return select(w, vec2<f32>(w.x, -w.y), fft_params.inverse == 1u);
}


/// Run DIT stages 0-7 on a 256-element slice in workgroup memory.
/// After this, the slice is in normal order (stages 0-7 complete).
fn dit_stages_0_7(tid: u32) {
    for (var s = 0u; s < 8u; s++) {
        let dist = 1u << s;
        let pair = tid ^ dist;
        if (tid < pair) {
            let w = load_twiddle(s, tid & (dist - 1u));
            let even = wg_data[tid];
            let odd = wg_data[pair];
            wg_data[tid] = even + complex_mul(odd, w);
            wg_data[pair] = even - complex_mul(odd, w);
        }
        workgroupBarrier();
    }
}

/// Run DIT stages 8-10 on 8 elements in registers (indices 0..7).
/// Pairs are within the same thread: (0,1),(2,3),(4,5),(6,7) for s=8,
/// (0,2),(1,3),(4,6),(5,7) for s=9, (0,4),(1,5),(2,6),(3,7) for s=10.
fn dit_stages_8_10(reg: ptr<function, array<vec2<f32>, 8>>, t: u32) {
    // Stage 8: distance 256, pairs (0,1), (2,3), (4,5), (6,7)
    // All pairs use the same twiddle W_{512}^t (k = t for all groups)
    let w8 = load_twiddle(8u, t);
    for (var i = 0u; i < 8u; i += 2u) {
        let even = (*reg)[i];
        let odd = (*reg)[i + 1u];
        (*reg)[i] = even + complex_mul(odd, w8);
        (*reg)[i + 1u] = even - complex_mul(odd, w8);
    }
    // Stage 9: distance 512, pairs (0,2), (1,3), (4,6), (5,7)
    for (var i = 0u; i < 4u; i++) {
        let base = (i / 2u) * 4u + (i % 2u);
        let w = load_twiddle(9u, t + (i % 2u) * 256u);
        let even = (*reg)[base];
        let odd = (*reg)[base + 2u];
        (*reg)[base] = even + complex_mul(odd, w);
        (*reg)[base + 2u] = even - complex_mul(odd, w);
    }
    // Stage 10: distance 1024, pairs (0,4), (1,5), (2,6), (3,7)
    for (var i = 0u; i < 4u; i++) {
        let w = load_twiddle(10u, t + i * 256u);
        let even = (*reg)[i];
        let odd = (*reg)[i + 4u];
        (*reg)[i] = even + complex_mul(odd, w);
        (*reg)[i + 4u] = even - complex_mul(odd, w);
    }
}

/// Load 8 strided elements into registers with bit-reversal.
fn load_regs_row(base: u32, t: u32, reg: ptr<function, array<vec2<f32>, 8>>) {
    (*reg)[0] = fft_data[base + bitrev_lut[t]];
    (*reg)[1] = fft_data[base + bitrev_lut[t + 256u]];
    (*reg)[2] = fft_data[base + bitrev_lut[t + 512u]];
    (*reg)[3] = fft_data[base + bitrev_lut[t + 768u]];
    (*reg)[4] = fft_data[base + bitrev_lut[t + 1024u]];
    (*reg)[5] = fft_data[base + bitrev_lut[t + 1280u]];
    (*reg)[6] = fft_data[base + bitrev_lut[t + 1536u]];
    (*reg)[7] = fft_data[base + bitrev_lut[t + 1792u]];
}

/// Store 8 strided elements from registers (normal order).
fn store_regs_row(base: u32, t: u32, reg: array<vec2<f32>, 8>) {
    fft_data[base + t]          = reg[0];
    fft_data[base + t + 256u]   = reg[1];
    fft_data[base + t + 512u]   = reg[2];
    fft_data[base + t + 768u]   = reg[3];
    fft_data[base + t + 1024u]  = reg[4];
    fft_data[base + t + 1280u]  = reg[5];
    fft_data[base + t + 1536u]  = reg[6];
    fft_data[base + t + 1792u]  = reg[7];
}

/// Load 8 strided column elements into registers with bit-reversal.
fn load_regs_col(w: u32, col: u32, t: u32, reg: ptr<function, array<vec2<f32>, 8>>) {
    (*reg)[0] = fft_data[bitrev_lut[t] * w + col];
    (*reg)[1] = fft_data[bitrev_lut[t + 256u] * w + col];
    (*reg)[2] = fft_data[bitrev_lut[t + 512u] * w + col];
    (*reg)[3] = fft_data[bitrev_lut[t + 768u] * w + col];
    (*reg)[4] = fft_data[bitrev_lut[t + 1024u] * w + col];
    (*reg)[5] = fft_data[bitrev_lut[t + 1280u] * w + col];
    (*reg)[6] = fft_data[bitrev_lut[t + 1536u] * w + col];
    (*reg)[7] = fft_data[bitrev_lut[t + 1792u] * w + col];
}

/// Store 8 strided column elements from registers (normal order).
fn store_regs_col(w: u32, col: u32, t: u32, reg: array<vec2<f32>, 8>) {
    fft_data[t * w + col]          = reg[0];
    fft_data[(t + 256u) * w + col] = reg[1];
    fft_data[(t + 512u) * w + col] = reg[2];
    fft_data[(t + 768u) * w + col] = reg[3];
    fft_data[(t + 1024u) * w + col] = reg[4];
    fft_data[(t + 1280u) * w + col] = reg[5];
    fft_data[(t + 1536u) * w + col] = reg[6];
    fft_data[(t + 1792u) * w + col] = reg[7];
}

@compute @workgroup_size(256)
fn fft_row_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let row = wg_id.x;
    let base = row * fft_params.width;
    let t = local_id.x;

    var reg: array<vec2<f32>, 8>;
    load_regs_row(base, t, &reg);

    // Stages 0-7: process each 256-element slice in workgroup memory
    for (var slice = 0u; slice < 8u; slice++) {
        wg_data[t] = reg[slice];
        workgroupBarrier();
        dit_stages_0_7(t);
        reg[slice] = wg_data[t];
        workgroupBarrier();
    }

    // Stages 8-10: in registers
    dit_stages_8_10(&reg, t);

    store_regs_row(base, t, reg);
}

@compute @workgroup_size(256)
fn fft_col_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let col = wg_id.x;
    let w = fft_params.width;
    let t = local_id.x;

    var reg: array<vec2<f32>, 8>;
    load_regs_col(w, col, t, &reg);

    for (var slice = 0u; slice < 8u; slice++) {
        wg_data[t] = reg[slice];
        workgroupBarrier();
        dit_stages_0_7(t);
        reg[slice] = wg_data[t];
        workgroupBarrier();
    }

    dit_stages_8_10(&reg, t);

    store_regs_col(w, col, t, reg);
}

@compute @workgroup_size(256)
fn ifft_row_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let row = wg_id.x;
    let base = row * fft_params.width;
    let t = local_id.x;

    var reg: array<vec2<f32>, 8>;
    load_regs_row(base, t, &reg);

    for (var slice = 0u; slice < 8u; slice++) {
        wg_data[t] = reg[slice];
        workgroupBarrier();
        dit_stages_0_7(t);
        reg[slice] = wg_data[t];
        workgroupBarrier();
    }

    dit_stages_8_10(&reg, t);

    store_regs_row(base, t, reg);
}

@compute @workgroup_size(256)
fn ifft_col_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let col = wg_id.x;
    let w = fft_params.width;
    let t = local_id.x;

    var reg: array<vec2<f32>, 8>;
    load_regs_col(w, col, t, &reg);

    for (var slice = 0u; slice < 8u; slice++) {
        wg_data[t] = reg[slice];
        workgroupBarrier();
        dit_stages_0_7(t);
        reg[slice] = wg_data[t];
        workgroupBarrier();
    }

    dit_stages_8_10(&reg, t);

    store_regs_col(w, col, t, reg);
}

// Fused complex multiply + IFFT column pass
@compute @workgroup_size(256)
fn fused_cmul_ifft_col_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let col = wg_id.x;
    let w = fft_params.width;
    let t = local_id.x;

    var reg: array<vec2<f32>, 8>;
    // Load + complex multiply + bit-reverse in one step
    reg[0] = complex_mul(fft_data[bitrev_lut[t] * w + col], cm_kernel[bitrev_lut[t] * w + col]);
    reg[1] = complex_mul(fft_data[bitrev_lut[t + 256u] * w + col], cm_kernel[bitrev_lut[t + 256u] * w + col]);
    reg[2] = complex_mul(fft_data[bitrev_lut[t + 512u] * w + col], cm_kernel[bitrev_lut[t + 512u] * w + col]);
    reg[3] = complex_mul(fft_data[bitrev_lut[t + 768u] * w + col], cm_kernel[bitrev_lut[t + 768u] * w + col]);
    reg[4] = complex_mul(fft_data[bitrev_lut[t + 1024u] * w + col], cm_kernel[bitrev_lut[t + 1024u] * w + col]);
    reg[5] = complex_mul(fft_data[bitrev_lut[t + 1280u] * w + col], cm_kernel[bitrev_lut[t + 1280u] * w + col]);
    reg[6] = complex_mul(fft_data[bitrev_lut[t + 1536u] * w + col], cm_kernel[bitrev_lut[t + 1536u] * w + col]);
    reg[7] = complex_mul(fft_data[bitrev_lut[t + 1792u] * w + col], cm_kernel[bitrev_lut[t + 1792u] * w + col]);

    for (var slice = 0u; slice < 8u; slice++) {
        wg_data[t] = reg[slice];
        workgroupBarrier();
        dit_stages_0_7(t);
        reg[slice] = wg_data[t];
        workgroupBarrier();
    }

    dit_stages_8_10(&reg, t);

    // Store back (normal order)
    fft_data[t * w + col]          = reg[0];
    fft_data[(t + 256u) * w + col] = reg[1];
    fft_data[(t + 512u) * w + col] = reg[2];
    fft_data[(t + 768u) * w + col] = reg[3];
    fft_data[(t + 1024u) * w + col] = reg[4];
    fft_data[(t + 1280u) * w + col] = reg[5];
    fft_data[(t + 1536u) * w + col] = reg[6];
    fft_data[(t + 1792u) * w + col] = reg[7];
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

        let a: f32 = diff2_affinity[base + p];
        let aff_exp: f32 = exp(diff2_params.temp * a);

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

// ---------------------------------------------------------------------------
// Render: channel data → packed RGB  (bindings 20-22)
// ---------------------------------------------------------------------------
@group(0) @binding(20) var<storage, read> render_channels: array<f32>;
@group(0) @binding(21) var<storage, read_write> render_output: array<u32>;

struct RenderParams {
    width: u32,
    num_channels: u32,
}

@group(0) @binding(22) var<uniform> render_params: RenderParams;

@compute @workgroup_size(256)
fn render_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = render_params.width * render_params.width;
    if (idx >= total) { return; }

    let c0 = render_channels[idx];
    let c1 = render_channels[total + idx];
    let c2 = render_channels[2u * total + idx];

    let r = u32(sqrt(clamp(c0 * 1.5, 0.0, 1.0)) * 255.0);
    let g = u32(sqrt(clamp(c1 * 1.5, 0.0, 1.0)) * 255.0);
    let b = u32(sqrt(clamp(c2 * 1.5, 0.0, 1.0)) * 255.0);

    render_output[idx] = (r << 16u) | (g << 8u) | b;
}
