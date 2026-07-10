struct Params { m: f32, s: f32, h: f32 }
@group(0) @binding(0) var<storage, read> conv: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<storage, read_write> conv_x: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    if (i >= arrayLength(&result)) { return; }
    let x: f32 = conv[i].x;
    let diff: f32 = x - params.m;
    let g: f32 = exp(-(diff * diff) / (2.0 * params.s * params.s));
    result[i] = (2.0 * g - 1.0) * params.h;
    conv_x[i] = x;
}
