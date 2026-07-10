struct Params { width: u32, height: u32, num_fields: u32 }
@group(0) @binding(0) var<storage, read> input_field: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i: u32 = id.x;
    let total: u32 = params.width * params.height * params.num_fields;
    if (i >= total) { return; }
    let field: u32 = i / (params.width * params.height);
    let pixel: u32 = i % (params.width * params.height);
    let x: u32 = pixel % params.width;
    let y: u32 = pixel / params.width;
    let w: u32 = params.width;
    let h: u32 = params.height;
    let base: u32 = field * w * h;
    let xl: u32 = select(x - 1u, 0u, x == 0u);
    let xr: u32 = select(x + 1u, w - 1u, x + 1u >= w);
    let yu: u32 = select(y - 1u, 0u, y == 0u);
    let yd: u32 = select(y + 1u, h - 1u, y + 1u >= h);
    let tl = input_field[base + yu * w + xl];
    let tc = input_field[base + yu * w + x];
    let tr = input_field[base + yu * w + xr];
    let ml = input_field[base + y * w + xl];
    let mr = input_field[base + y * w + xr];
    let bl = input_field[base + yd * w + xl];
    let bc = input_field[base + yd * w + x];
    let br = input_field[base + yd * w + xr];
    grad_x[i] = (-tl + tr) + 2.0 * (-ml + mr) + (-bl + br);
    grad_y[i] = (-tl - 2.0 * tc - tr) + (bl + 2.0 * bc + br);
}
