struct Params { width: u32, num_kernels: u32, num_channels: u32 }
@group(0) @binding(0) var<storage, read> u_all: array<f32>;
@group(0) @binding(1) var<storage, read_write> u_channels: array<f32>;
@group(0) @binding(2) var<storage, read> c1_flat: array<u32>;
@group(0) @binding(3) var<storage, read> c1_offsets: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    let total: u32 = params.width * params.width * params.num_channels;
    if (i >= total) { return; }
    let c: u32 = i / (params.width * params.width);
    let pixel: u32 = i % (params.width * params.width);
    let start: u32 = c1_offsets[c];
    let end: u32 = c1_offsets[c + 1u];
    var sum: f32 = 0.0;
    for (var j: u32 = start; j < end; j = j + 1u) {
        let k: u32 = c1_flat[j];
        sum = sum + u_all[k * params.width * params.width + pixel];
    }
    u_channels[i] = sum;
}
