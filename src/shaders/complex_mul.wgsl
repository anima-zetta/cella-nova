@group(0) @binding(0) var<storage, read_write> conv: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> kernel: array<vec2<f32>>;
fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    if (i >= arrayLength(&conv)) { return; }
    conv[i] = complex_mul(conv[i], kernel[i]);
}
