@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> twiddles: array<vec2<f32>>;

struct Params {
    n: u32,
    stage: u32,
    inverse: u32,
    num_lanes: u32,
    lane_stride: u32,
    element_stride: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x,
    );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let half_n: u32 = params.n / 2u;
    let butterflies_per_lane: u32 = half_n;
    let total_butterflies: u32 = butterflies_per_lane * params.num_lanes;
    let i: u32 = id.x;
    if (i >= total_butterflies) {
        return;
    }
    let lane: u32 = i / butterflies_per_lane;
    let butterfly: u32 = i % butterflies_per_lane;
    let base: u32 = lane * params.lane_stride;
    let es: u32 = params.element_stride;

    let stride: u32 = 1u << params.stage;
    let block_size: u32 = stride * 2u;
    let block: u32 = butterfly / stride;
    let offset: u32 = butterfly % stride;
    let j: u32 = base + (block * block_size + offset) * es;
    let k: u32 = j + stride * es;

    let stage_offset: u32 = (1u << params.stage) - 1u;
    let w: vec2<f32> = twiddles[stage_offset + offset];

    let even: vec2<f32> = data[j];
    let odd: vec2<f32> = complex_mul(w, data[k]);

    data[j] = even + odd;
    data[k] = even - odd;
}
