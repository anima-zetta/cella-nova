@group(0) @binding(0) var<storage, read> channel: array<f32>;
@group(0) @binding(1) var<storage, read_write> conv: array<vec2<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    if (i >= arrayLength(&channel)) { return; }
    conv[i] = vec2<f32>(channel[i], 0.0);
}
