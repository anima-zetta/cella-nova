@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;

struct Params {
    n: u32,
    num_lanes: u32,
    lane_stride: u32,
    element_stride: u32,
}
@group(0) @binding(1) var<uniform> params: Params;

fn bit_reverse(x: u32, bits: u32) -> u32 {
    var result: u32 = 0u;
    for (var i: u32 = 0u; i < bits; i = i + 1u) {
        result = (result << 1u) | ((x >> i) & 1u);
    }
    return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let total: u32 = params.n * params.num_lanes;
    let i: u32 = id.x;
    if (i >= total) {
        return;
    }
    let lane: u32 = i / params.n;
    let offset_in_lane: u32 = i % params.n;
    let bits: u32 = u32(log2(f32(params.n)));
    let j: u32 = bit_reverse(offset_in_lane, bits);
    if (offset_in_lane < j) {
        let base: u32 = lane * params.lane_stride;
        let a: u32 = base + offset_in_lane * params.element_stride;
        let b: u32 = base + j * params.element_stride;
        let tmp: vec2<f32> = data[a];
        data[a] = data[b];
        data[b] = tmp;
    }
}
