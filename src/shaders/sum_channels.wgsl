struct Params { width: u32, num_channels: u32 }
@group(0) @binding(0) var<storage, read> channels: array<f32>;
@group(0) @binding(1) var<storage, read_write> sum_out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    if (i >= params.width * params.width) { return; }
    var sum: f32 = 0.0;
    for (var c: u32 = 0u; c < params.num_channels; c = c + 1u) {
        sum = sum + channels[c * params.width * params.width + i];
    }
    sum_out[i] = sum;
}
