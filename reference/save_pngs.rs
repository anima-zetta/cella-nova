// -*- coding: utf-8 -*-
// Generate PNG frames matching train/save_frames_png.py
#![allow(non_snake_case, dead_code)]

use ndarray::prelude::*;
use ndarray::{concatenate, Axis};
use num_complex::Complex;
use rustfft::FftPlanner;

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

fn sigmoid(x: f64) -> f64 {
    0.5 * ((x / 2.0).tanh() + 1.0)
}

fn ker_f(x: &Array2<f64>, a: &[f64], w: &[f64], b: &[f64]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros(x.dim());
    for p in 0..a.len() {
        let diff = x - a[p];
        out = out + b[p] * (&diff * &diff).mapv(|v| (-v / w[p]).exp());
    }
    out
}

fn bell(x: f64, m: f64, s: f64) -> f64 {
    (-((x - m) / s).powi(2) / 2.0).exp()
}

// ---------------------------------------------------------------------------
// Sobel operators
// ---------------------------------------------------------------------------

const KX: [[f64; 3]; 3] = [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]];
const KY: [[f64; 3]; 3] = [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]];

fn convolve2d_same_3x3(arr: &Array2<f64>, kernel: &[[f64; 3]; 3]) -> Array2<f64> {
    let (h, w) = arr.dim();
    let mut out = Array2::<f64>::zeros((h, w));
    for i in 0..h {
        for j in 0..w {
            let mut sum = 0.0;
            for di in 0..3 {
                for dj in 0..3 {
                    let ni = i as i32 - di as i32 + 1;
                    let nj = j as i32 - dj as i32 + 1;
                    if ni >= 0 && ni < h as i32 && nj >= 0 && nj < w as i32 {
                        sum += arr[[ni as usize, nj as usize]] * kernel[di][dj];
                    }
                }
            }
            out[[i, j]] = sum;
        }
    }
    out
}

fn sobel_x(A: &Array3<f64>) -> Array3<f64> {
    let (sx, sy, c) = A.dim();
    let mut out = Array3::<f64>::zeros((sx, sy, c));
    for ch in 0..c {
        let slice = A.slice(s![.., .., ch]);
        out.slice_mut(s![.., .., ch])
            .assign(&convolve2d_same_3x3(&slice.to_owned(), &KX));
    }
    out
}

fn sobel_y(A: &Array3<f64>) -> Array3<f64> {
    let (sx, sy, c) = A.dim();
    let mut out = Array3::<f64>::zeros((sx, sy, c));
    for ch in 0..c {
        let slice = A.slice(s![.., .., ch]);
        out.slice_mut(s![.., .., ch])
            .assign(&convolve2d_same_3x3(&slice.to_owned(), &KY));
    }
    out
}

fn sobel(A: &Array3<f64>) -> Array4<f64> {
    let sy = sobel_y(A).insert_axis(Axis(2));
    let sx = sobel_x(A).insert_axis(Axis(2));
    concatenate(Axis(2), &[sy.view(), sx.view()]).unwrap()
}

// ---------------------------------------------------------------------------
// FFT helpers
// ---------------------------------------------------------------------------

struct Fft2DPlanner {
    planner: FftPlanner<f64>,
}

impl Fft2DPlanner {
    fn new() -> Self {
        Self {
            planner: FftPlanner::<f64>::new(),
        }
    }

    fn fft2(&mut self, arr: &Array3<Complex<f64>>) -> Array3<Complex<f64>> {
        let (sx, sy, k) = arr.dim();
        let mut result = arr.clone();
        let fft_rows = self.planner.plan_fft_forward(sy);
        for k_idx in 0..k {
            for i in 0..sx {
                let mut row: Vec<Complex<f64>> = (0..sy).map(|j| result[[i, j, k_idx]]).collect();
                fft_rows.process(&mut row);
                for j in 0..sy {
                    result[[i, j, k_idx]] = row[j];
                }
            }
        }
        let fft_cols = self.planner.plan_fft_forward(sx);
        for k_idx in 0..k {
            for j in 0..sy {
                let mut col: Vec<Complex<f64>> = (0..sx).map(|i| result[[i, j, k_idx]]).collect();
                fft_cols.process(&mut col);
                for i in 0..sx {
                    result[[i, j, k_idx]] = col[i];
                }
            }
        }
        result
    }

    fn ifft2(&mut self, arr: &Array3<Complex<f64>>) -> Array3<Complex<f64>> {
        let (sx, sy, k) = arr.dim();
        let mut result = arr.clone();
        let ifft_rows = self.planner.plan_fft_inverse(sy);
        for k_idx in 0..k {
            for i in 0..sx {
                let mut row: Vec<Complex<f64>> = (0..sy).map(|j| result[[i, j, k_idx]]).collect();
                ifft_rows.process(&mut row);
                for j in 0..sy {
                    result[[i, j, k_idx]] = row[j];
                }
            }
        }
        let ifft_cols = self.planner.plan_fft_inverse(sx);
        for k_idx in 0..k {
            for j in 0..sy {
                let mut col: Vec<Complex<f64>> = (0..sx).map(|i| result[[i, j, k_idx]]).collect();
                ifft_cols.process(&mut col);
                for i in 0..sx {
                    result[[i, j, k_idx]] = col[i];
                }
            }
        }
        let norm = (sx * sy) as f64;
        result.mapv_inplace(|v| v / norm);
        result
    }
}

fn fftshift_3d(arr: &Array3<Complex<f64>>) -> Array3<Complex<f64>> {
    let (sx, sy, k) = arr.dim();
    let mut result = Array3::<Complex<f64>>::zeros((sx, sy, k));
    let half_rows = sx / 2;
    let half_cols = sy / 2;
    for i in 0..sx {
        for j in 0..sy {
            let ni = (i + half_rows) % sx;
            let nj = (j + half_cols) % sy;
            for k_idx in 0..k {
                result[[ni, nj, k_idx]] = arr[[i, j, k_idx]];
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Reintegration
// ---------------------------------------------------------------------------

fn build_reintegration(
    SX: usize,
    SY: usize,
    dt: f64,
    dd: i32,
    sigma: f64,
    border: &str,
) -> Box<dyn Fn(&Array3<f64>, &Array4<f64>) -> Array3<f64>> {
    let pos_y = Array2::from_shape_fn((SY, SX), |(i, _j)| i as f64 + 0.5);
    let pos_x = Array2::from_shape_fn((SY, SX), |(_i, j)| j as f64 + 0.5);
    let mut pos = Array3::<f64>::zeros((SY, SX, 2));
    pos.slice_mut(s![.., .., 0]).assign(&pos_y);
    pos.slice_mut(s![.., .., 1]).assign(&pos_x);

    let n_offsets = ((2 * dd + 1) * (2 * dd + 1)) as usize;
    let mut dxs = Array1::<i32>::zeros(n_offsets);
    let mut dys = Array1::<i32>::zeros(n_offsets);
    let mut idx = 0;
    for dx in -dd..=dd {
        for dy in -dd..=dd {
            dxs[idx] = dx;
            dys[idx] = dy;
            idx += 1;
        }
    }

    let border_owned = border.to_string();

    Box::new(move |X: &Array3<f64>, F: &Array4<f64>| {
        let (sx, sy, _two, c) = F.dim();
        let ma = dd as f64 - sigma;

        let mut mu = Array4::<f64>::zeros((sx, sy, 2, c));
        for ci in 0..c {
            for a in 0..2 {
                let p_slice = pos.slice(s![.., .., a]);
                let f_slice = F.slice(s![.., .., a, ci]);
                let clipped = f_slice.mapv(|v| (v * dt).clamp(-ma, ma));
                mu.slice_mut(s![.., .., a, ci])
                    .assign(&(&p_slice + &clipped));
            }
        }
        if border_owned == "wall" {
            for ci in 0..c {
                for a in 0..2 {
                    mu.slice_mut(s![.., .., a, ci])
                        .mapv_inplace(|v| v.clamp(sigma, SX as f64 - sigma));
                }
            }
        }

        let mut result = Array3::<f64>::zeros((sx, sy, c));
        for oi in 0..n_offsets {
            let dx = dxs[oi];
            let dy = dys[oi];
            let Xr = roll_3d_axes01(X, dx, dy);
            let mur = roll_4d_axes01(&mu, dx, dy);

            let mut dpmu = Array4::<f64>::zeros((sx, sy, 2, c));
            for ci in 0..c {
                for a in 0..2 {
                    let diff = (pos.slice(s![.., .., a]).to_owned()
                        - mur.slice(s![.., .., a, ci]).to_owned())
                    .mapv(|v| v.abs());
                    dpmu.slice_mut(s![.., .., a, ci]).assign(&diff);
                }
            }
            let sz = dpmu.mapv(|v| 0.5 - v + sigma);
            let clip_max = 1.0_f64.min(2.0 * sigma);
            let sz_clipped = sz.mapv(|v| v.clamp(0.0, clip_max));

            for ci in 0..c {
                for i in 0..sx {
                    for j in 0..sy {
                        let area = sz_clipped[[i, j, 0, ci]] * sz_clipped[[i, j, 1, ci]]
                            / (4.0 * sigma * sigma);
                        result[[i, j, ci]] += Xr[[i, j, ci]] * area;
                    }
                }
            }
        }
        result
    })
}

fn roll_3d_axes01(arr: &Array3<f64>, dx: i32, dy: i32) -> Array3<f64> {
    let (sx, sy, k) = arr.dim();
    let mut out = Array3::<f64>::zeros((sx, sy, k));
    for i in 0..sx {
        for j in 0..sy {
            let ni = ((i as i32 + dx).rem_euclid(sx as i32)) as usize;
            let nj = ((j as i32 + dy).rem_euclid(sy as i32)) as usize;
            for k_idx in 0..k {
                out[[ni, nj, k_idx]] = arr[[i, j, k_idx]];
            }
        }
    }
    out
}

fn roll_4d_axes01(arr: &Array4<f64>, dx: i32, dy: i32) -> Array4<f64> {
    let (sx, sy, a, b) = arr.dim();
    let mut out = Array4::<f64>::zeros((sx, sy, a, b));
    for i in 0..sx {
        for j in 0..sy {
            let ni = ((i as i32 + dx).rem_euclid(sx as i32)) as usize;
            let nj = ((j as i32 + dy).rem_euclid(sy as i32)) as usize;
            for a_idx in 0..a {
                for b_idx in 0..b {
                    out[[ni, nj, a_idx, b_idx]] = arr[[i, j, a_idx, b_idx]];
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Params {
    r: Array1<f64>,
    b: Array2<f64>,
    w: Array2<f64>,
    a: Array2<f64>,
    m: Array1<f64>,
    s: Array1<f64>,
    h: Array1<f64>,
    R: f64,
}

#[derive(Clone, Debug)]
struct CompiledParams {
    fK: Array3<Complex<f64>>,
    m: Array1<f64>,
    s: Array1<f64>,
    h: Array1<f64>,
}

#[derive(Clone, Debug)]
struct Config {
    SX: usize,
    SY: usize,
    nb_k: usize,
    C: usize,
    c0: Vec<usize>,
    c1: Vec<Vec<usize>>,
    dt: f64,
    dd: i32,
    sigma: f64,
    n: f64,
    theta_A: f64,
    border: String,
}

#[derive(Clone, Debug)]
struct State {
    A: Array3<f64>,
}

// ---------------------------------------------------------------------------
// Kernel computation
// ---------------------------------------------------------------------------

fn build_kernel_computer(
    SX: usize,
    SY: usize,
    nb_k: usize,
) -> impl FnMut(&Params, &mut Fft2DPlanner) -> CompiledParams {
    let mid = SX as i32 / 2;
    move |params: &Params, planner: &mut Fft2DPlanner| -> CompiledParams {
        let mut K = Array3::<f64>::zeros((SY, SX, nb_k));
        for k in 0..nb_k {
            let mut dist = Array2::<f64>::zeros((SY, SX));
            for i in 0..SY {
                for j in 0..SX {
                    let di = i as i32 - mid;
                    let dj = j as i32 - mid;
                    dist[[i, j]] =
                        ((di * di + dj * dj) as f64).sqrt() / ((params.R + 15.0) * params.r[k]);
                }
            }
            let sig = dist.mapv(|d| sigmoid(-(d - 1.0) * 10.0));
            let a: Vec<f64> = params.a.row(k).to_vec();
            let w: Vec<f64> = params.w.row(k).to_vec();
            let b: Vec<f64> = params.b.row(k).to_vec();
            let kf = ker_f(&dist, &a, &w, &b);
            let mut kslice = sig * kf;
            let total = kslice.sum();
            if total > 0.0 {
                kslice = kslice / total;
            }
            K.slice_mut(s![.., .., k]).assign(&kslice);
        }
        let K_complex = K.mapv(|v| Complex::new(v, 0.0));
        let K_shifted = fftshift_3d(&K_complex);
        let fK = planner.fft2(&K_shifted);
        CompiledParams {
            fK,
            m: params.m.clone(),
            s: params.s.clone(),
            h: params.h.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Step function
// ---------------------------------------------------------------------------

fn build_step_fn(
    config: &Config,
) -> impl FnMut(&State, &CompiledParams, &mut Fft2DPlanner) -> State + '_ {
    let advect = build_reintegration(
        config.SX,
        config.SY,
        config.dt,
        config.dd,
        config.sigma,
        &config.border,
    );
    let c0 = config.c0.clone();
    let c1 = config.c1.clone();
    let C = config.C;
    let nb_k = config.nb_k;
    let n = config.n;
    let theta_A = config.theta_A;
    let SX = config.SX;
    let SY = config.SY;

    move |state: &State, params: &CompiledParams, planner: &mut Fft2DPlanner| -> State {
        let A = &state.A;
        let A_complex = A.mapv(|v| Complex::new(v, 0.0));
        let fA = planner.fft2(&A_complex);

        let mut fA_selected = Array3::<Complex<f64>>::zeros((SX, SY, nb_k));
        for k in 0..nb_k {
            let src_ch = c0[k];
            for i in 0..SX {
                for j in 0..SY {
                    fA_selected[[i, j, k]] = fA[[i, j, src_ch]];
                }
            }
        }

        let mut fK_times_fA = Array3::<Complex<f64>>::zeros((SX, SY, nb_k));
        for k in 0..nb_k {
            for i in 0..SX {
                for j in 0..SY {
                    fK_times_fA[[i, j, k]] = params.fK[[i, j, k]] * fA_selected[[i, j, k]];
                }
            }
        }

        let U_complex = planner.ifft2(&fK_times_fA);
        let mut U = U_complex.mapv(|v| v.re);

        for k in 0..nb_k {
            for i in 0..SX {
                for j in 0..SY {
                    let u = U[[i, j, k]];
                    U[[i, j, k]] = (bell(u, params.m[k], params.s[k]) * 2.0 - 1.0) * params.h[k];
                }
            }
        }

        let mut U_new = Array3::<f64>::zeros((SX, SY, C));
        for c in 0..C {
            for &k_idx in &c1[c] {
                for i in 0..SX {
                    for j in 0..SY {
                        U_new[[i, j, c]] += U[[i, j, k_idx]];
                    }
                }
            }
        }

        let nabla_U = sobel(&U_new);
        let A_sum = A.sum_axis(Axis(2)).insert_axis(Axis(2));
        let nabla_A = sobel(&A_sum);

        let mut alpha = Array4::<f64>::zeros((SX, SY, 2, C));
        for c_idx in 0..C {
            for i in 0..SX {
                for j in 0..SY {
                    let val = (A[[i, j, c_idx]] / theta_A).powf(n);
                    let clamped = val.clamp(0.0, 1.0);
                    alpha[[i, j, 0, c_idx]] = clamped;
                    alpha[[i, j, 1, c_idx]] = clamped;
                }
            }
        }

        let mut flow = Array4::<f64>::zeros((SX, SY, 2, C));
        for c_idx in 0..C {
            for a in 0..2 {
                for i in 0..SX {
                    for j in 0..SY {
                        flow[[i, j, a, c_idx]] = nabla_U[[i, j, a, c_idx]]
                            * (1.0 - alpha[[i, j, a, c_idx]])
                            - nabla_A[[i, j, a, 0]] * alpha[[i, j, a, c_idx]];
                    }
                }
            }
        }

        State {
            A: advect(A, &flow),
        }
    }
}

// ---------------------------------------------------------------------------
// PNG saving
// ---------------------------------------------------------------------------

fn save_frame_as_png(arr: &Array3<f64>, path: &str) {
    let (sx, sy, _c) = arr.dim();
    // Sum channels
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    let mut pixels = vec![0u8; sx * sy];
    for i in 0..sx {
        for j in 0..sy {
            let mut sum = 0.0;
            for c in 0.._c {
                sum += arr[[i, j, c]];
            }
            pixels[i * sy + j] = 0; // placeholder
            if sum < min_val {
                min_val = sum;
            }
            if sum > max_val {
                max_val = sum;
            }
        }
    }
    let range = max_val - min_val;
    for i in 0..sx {
        for j in 0..sy {
            let mut sum = 0.0;
            for c in 0.._c {
                sum += arr[[i, j, c]];
            }
            let normalized = if range > 0.0 {
                (sum - min_val) / range
            } else {
                0.0
            };
            pixels[i * sy + j] = (normalized * 255.0) as u8;
        }
    }
    image::save_buffer(path, &pixels, sy as u32, sx as u32, image::ColorType::L8).unwrap();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let SX: usize = 64;
    let SY: usize = 64;
    let nb_k: usize = 2;
    let C: usize = 2;

    let params = Params {
        r: arr1(&[0.5, 0.8]),
        b: arr2(&[[0.5, 0.3, 0.0], [0.7, 0.2, 0.0]]),
        w: arr2(&[[0.1, 0.05, 0.01], [0.08, 0.06, 0.01]]),
        a: arr2(&[[0.0, 0.5, 0.0], [0.0, 0.4, 0.0]]),
        m: arr1(&[0.1, 0.15]),
        s: arr1(&[0.05, 0.08]),
        h: arr1(&[0.5, 0.8]),
        R: 10.0,
    };

    let c0: Vec<usize> = vec![0, 1];
    let c1: Vec<Vec<usize>> = vec![vec![0], vec![1]];

    let config = Config {
        SX,
        SY,
        nb_k,
        C,
        c0,
        c1,
        dt: 0.2,
        dd: 5,
        sigma: 0.65,
        n: 2.0,
        theta_A: 1.0,
        border: "wall".to_string(),
    };

    // Compute kernels
    let mut planner = Fft2DPlanner::new();
    let mut kernel_computer = build_kernel_computer(SX, SY, nb_k);
    let compiled = kernel_computer(&params, &mut planner);

    // Initialize state with Gaussian blob
    let mut A = Array3::<f64>::zeros((SX, SY, C));
    let cx = SX as f64 / 2.0;
    let cy = SY as f64 / 2.0;
    for i in 0..SX {
        for j in 0..SY {
            let dx = i as f64 - cx;
            let dy = j as f64 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let val = (-dist * dist / 64.0).exp();
            for c in 0..C {
                A[[i, j, c]] = val * (0.5 + 0.5 * c as f64 / C as f64);
            }
        }
    }

    // Save initial state
    std::fs::create_dir_all("pngs").unwrap();
    save_frame_as_png(&A, "pngs/rs_frame_0000.png");

    // Run 50 steps
    let mut step_fn = build_step_fn(&config);
    let mut state = State { A };
    for step in 1..=50 {
        state = step_fn(&state, &compiled, &mut planner);
        let path = format!("pngs/rs_frame_{:04}.png", step);
        save_frame_as_png(&state.A, &path);
    }

    println!("Saved 51 Rust frames (step 0 + 50 steps) to pngs/");
    println!("  rs_frame_0000.png through rs_frame_0050.png");
}
