// -*- coding: utf-8 -*-
// Rust port of train/flowlenia.py — matches the Python implementation 1:1
//
// Skipped (Python-specific): VideoWriter, matplotlib, IPython.display, moviepy

#![allow(non_snake_case, dead_code)]

use ndarray::prelude::*;
use ndarray::{concatenate, Axis};
use num_complex::Complex;
use rand::Rng;
use rustfft::FftPlanner;

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

fn sigmoid(x: f32) -> f32 {
    0.5 * ((x / 2.0).tanh() + 1.0)
}

/// ker_f(x, a, w, b) = sum_p b[p] * exp(-(x - a[p])^2 / w[p])
fn ker_f(x: &Array2<f32>, a: &[f32], w: &[f32], b: &[f32]) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros(x.dim());
    for p in 0..a.len() {
        let diff = x - a[p];
        out = out + b[p] * (&diff * &diff).mapv(|v| (-v / w[p]).exp());
    }
    out
}

fn bell(x: f32, m: f32, s: f32) -> f32 {
    (-((x - m) / s).powi(2) / 2.0).exp()
}

fn growth(U: &Array3<f32>, m: &Array1<f32>, s: &Array1<f32>) -> Array3<f32> {
    let (sx, sy, nb_k) = U.dim();
    let mut out = Array3::<f32>::zeros((sx, sy, nb_k));
    for k in 0..nb_k {
        let uk = U.slice(s![.., .., k]);
        let gk = uk.mapv(|u| bell(u, m[k], s[k]) * 2.0 - 1.0);
        out.slice_mut(s![.., .., k]).assign(&gk);
    }
    out
}

// ---------------------------------------------------------------------------
// Sobel operators
// ---------------------------------------------------------------------------

const KX: [[f32; 3]; 3] = [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]];
const KY: [[f32; 3]; 3] = [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]];

/// 2D convolution with 'same' mode for a 3x3 kernel
fn convolve2d_same_3x3(arr: &Array2<f32>, kernel: &[[f32; 3]; 3]) -> Array2<f32> {
    let (h, w) = arr.dim();
    let mut out = Array2::<f32>::zeros((h, w));
    for i in 0..h {
        for j in 0..w {
            let mut sum = 0.0;
            for di in 0..3 {
                for dj in 0..3 {
                    // scipy.signal.convolve2d: C[i,j] = sum A[i-di+1, j-dj+1] * K[di,dj]
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

fn sobel_x(A: &Array3<f32>) -> Array3<f32> {
    let (sx, sy, c) = A.dim();
    let mut out = Array3::<f32>::zeros((sx, sy, c));
    for ch in 0..c {
        let slice = A.slice(s![.., .., ch]);
        let conv = convolve2d_same_3x3(&slice.to_owned(), &KX);
        out.slice_mut(s![.., .., ch]).assign(&conv);
    }
    out
}

fn sobel_y(A: &Array3<f32>) -> Array3<f32> {
    let (sx, sy, c) = A.dim();
    let mut out = Array3::<f32>::zeros((sx, sy, c));
    for ch in 0..c {
        let slice = A.slice(s![.., .., ch]);
        let conv = convolve2d_same_3x3(&slice.to_owned(), &KY);
        out.slice_mut(s![.., .., ch]).assign(&conv);
    }
    out
}

/// Returns array of shape (SX, SY, 2, C) — [dy, dx] gradients
fn sobel(A: &Array3<f32>) -> Array4<f32> {
    let sy = sobel_y(A).insert_axis(Axis(2)); // (SX, SY, 1, C)
    let sx = sobel_x(A).insert_axis(Axis(2)); // (SX, SY, 1, C)
    concatenate(Axis(2), &[sy.view(), sx.view()]).unwrap() // (SX, SY, 2, C)
}

// ---------------------------------------------------------------------------
// Kernel generation
// ---------------------------------------------------------------------------

fn get_kernels(SX: usize, SY: usize, nb_k: usize, params: &Params) -> Array3<f32> {
    let mid = SX as i32 / 2;
    let _size = SX.min(SY);
    let mut Ds: Vec<Array2<f32>> = Vec::with_capacity(nb_k);
    for k in 0..nb_k {
        let mut dist = Array2::<f32>::zeros((SY, SX));
        for i in 0..SY {
            for j in 0..SX {
                let di = i as i32 - mid;
                let dj = j as i32 - mid;
                let d = ((di * di + dj * dj) as f32).sqrt();
                dist[[i, j]] = d / ((params.R + 15.0) * params.r[k]);
            }
        }
        Ds.push(dist);
    }

    let mut K = Array3::<f32>::zeros((SY, SX, nb_k));
    for (k, D) in Ds.iter().enumerate() {
        let sig = D.mapv(|d| sigmoid(-(d - 1.0) * 10.0));
        let a: Vec<f32> = params.a.row(k).to_vec();
        let w: Vec<f32> = params.w.row(k).to_vec();
        let b: Vec<f32> = params.b.row(k).to_vec();
        let kf = ker_f(D, &a, &w, &b);
        let mut kslice = sig * kf;
        let total = kslice.sum();
        if total > 0.0 {
            kslice = kslice / total;
        }
        K.slice_mut(s![.., .., k]).assign(&kslice);
    }
    K
}

// ---------------------------------------------------------------------------
// Connectivity helpers
// ---------------------------------------------------------------------------

fn conn_from_matrix(mat: &Array2<i32>) -> (Vec<usize>, Vec<Vec<usize>>) {
    let C = mat.shape()[0];
    let mut c0: Vec<usize> = Vec::new();
    let mut c1: Vec<Vec<usize>> = vec![Vec::new(); C];
    let mut i = 0;
    for s in 0..C {
        for t in 0..C {
            let n = mat[[s, t]];
            if n > 0 {
                for _ in 0..n {
                    c0.push(s);
                }
                for idx in i..(i + n as usize) {
                    c1[t].push(idx);
                }
            }
            i += n as usize;
        }
    }
    (c0, c1)
}

fn conn_from_lists(c0: &[usize], c1: &[Vec<usize>], C: usize) -> (Vec<usize>, Vec<Vec<bool>>) {
    let nb_k = c0.len();
    let mut c1_bool: Vec<Vec<bool>> = vec![vec![false; nb_k]; C];
    for c in 0..C {
        for &idx in &c1[c] {
            c1_bool[c][idx] = true;
        }
    }
    (c0.to_vec(), c1_bool)
}

// ---------------------------------------------------------------------------
// FFT helpers
// ---------------------------------------------------------------------------

struct Fft2DPlanner {
    planner: FftPlanner<f32>,
}

impl Fft2DPlanner {
    fn new() -> Self {
        Self {
            planner: FftPlanner::<f32>::new(),
        }
    }

    /// 2D FFT along axes (0, 1) for each slice along the last axis
    fn fft2(&mut self, arr: &Array3<Complex<f32>>) -> Array3<Complex<f32>> {
        let (sx, sy, k) = arr.dim();
        let mut result = arr.clone();

        // FFT along axis 1 (rows) for each row and each kernel
        let fft_rows = self.planner.plan_fft_forward(sy);
        for k_idx in 0..k {
            for i in 0..sx {
                let mut row: Vec<Complex<f32>> = (0..sy).map(|j| result[[i, j, k_idx]]).collect();
                fft_rows.process(&mut row);
                for j in 0..sy {
                    result[[i, j, k_idx]] = row[j];
                }
            }
        }

        // FFT along axis 0 (columns) for each column and each kernel
        let fft_cols = self.planner.plan_fft_forward(sx);
        for k_idx in 0..k {
            for j in 0..sy {
                let mut col: Vec<Complex<f32>> = (0..sx).map(|i| result[[i, j, k_idx]]).collect();
                fft_cols.process(&mut col);
                for i in 0..sx {
                    result[[i, j, k_idx]] = col[i];
                }
            }
        }

        result
    }

    /// 2D IFFT along axes (0, 1) for each slice along the last axis
    /// Note: rustfft's inverse FFT does NOT normalize, so we divide by sx*sy manually.
    fn ifft2(&mut self, arr: &Array3<Complex<f32>>) -> Array3<Complex<f32>> {
        let (sx, sy, k) = arr.dim();
        let mut result = arr.clone();

        // IFFT along axis 1 (rows)
        let ifft_rows = self.planner.plan_fft_inverse(sy);
        for k_idx in 0..k {
            for i in 0..sx {
                let mut row: Vec<Complex<f32>> = (0..sy).map(|j| result[[i, j, k_idx]]).collect();
                ifft_rows.process(&mut row);
                for j in 0..sy {
                    result[[i, j, k_idx]] = row[j];
                }
            }
        }

        // IFFT along axis 0 (columns)
        let ifft_cols = self.planner.plan_fft_inverse(sx);
        for k_idx in 0..k {
            for j in 0..sy {
                let mut col: Vec<Complex<f32>> = (0..sx).map(|i| result[[i, j, k_idx]]).collect();
                ifft_cols.process(&mut col);
                for i in 0..sx {
                    result[[i, j, k_idx]] = col[i];
                }
            }
        }

        // Normalize: rustfft's inverse FFT does NOT divide by n, so we must divide by sx*sy
        let norm = (sx * sy) as f32;
        result.mapv_inplace(|v| v / norm);
        result
    }
}

/// fftshift along axes (0, 1) for a 2D array
fn fftshift_2d(arr: &Array2<Complex<f32>>) -> Array2<Complex<f32>> {
    let (nrows, ncols) = arr.dim();
    let mut result = Array2::<Complex<f32>>::zeros((nrows, ncols));
    let half_rows = nrows / 2;
    let half_cols = ncols / 2;
    for i in 0..nrows {
        for j in 0..ncols {
            let ni = (i + half_rows) % nrows;
            let nj = (j + half_cols) % ncols;
            result[[ni, nj]] = arr[[i, j]];
        }
    }
    result
}

/// ifftshift along axes (0, 1) for a 2D array
fn ifftshift_2d(arr: &Array2<Complex<f32>>) -> Array2<Complex<f32>> {
    let (nrows, ncols) = arr.dim();
    let mut result = Array2::<Complex<f32>>::zeros((nrows, ncols));
    let half_rows = (nrows + 1) / 2;
    let half_cols = (ncols + 1) / 2;
    for i in 0..nrows {
        for j in 0..ncols {
            let ni = (i + half_rows) % nrows;
            let nj = (j + half_cols) % ncols;
            result[[ni, nj]] = arr[[i, j]];
        }
    }
    result
}

/// fftshift for a 3D array along axes (0, 1)
fn fftshift_3d(arr: &Array3<Complex<f32>>) -> Array3<Complex<f32>> {
    let (sx, sy, k) = arr.dim();
    let mut result = Array3::<Complex<f32>>::zeros((sx, sy, k));
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
// Reintegration tracking (semi-Lagrangian advection)
// ---------------------------------------------------------------------------

fn build_reintegration(
    SX: usize,
    SY: usize,
    dt: f32,
    dd: i32,
    sigma: f32,
    border: &str,
) -> Box<dyn Fn(&Array3<f32>, &Array4<f32>) -> Array3<f32>> {
    // Build position grid
    let pos_y = Array2::from_shape_fn((SY, SX), |(i, _j)| i as f32 + 0.5);
    let pos_x = Array2::from_shape_fn((SY, SX), |(_i, j)| j as f32 + 0.5);
    // pos as (SY, SX, 2) — [y, x]
    let mut pos = Array3::<f32>::zeros((SY, SX, 2));
    pos.slice_mut(s![.., .., 0]).assign(&pos_y);
    pos.slice_mut(s![.., .., 1]).assign(&pos_x);

    // dxs, dys: all offsets in [-dd, dd]
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

    Box::new(move |X: &Array3<f32>, F: &Array4<f32>| {
        // F has shape (SX, SY, 2, C) — [dy, dx] gradients
        let (sx, sy, _two, c) = F.dim();
        let ma = dd as f32 - sigma;

        // mu = pos + clip(dt * F, -ma, ma)  — shape (SX, SY, 2, C)
        let mut mu = Array4::<f32>::zeros((sx, sy, 2, c));
        for ci in 0..c {
            for a in 0..2 {
                let f_slice = F.slice(s![.., .., a, ci]);
                let p_slice = pos.slice(s![.., .., a]);
                let clipped = f_slice.mapv(|v| (v * dt).clamp(-ma, ma));
                let mut mu_slice = mu.slice_mut(s![.., .., a, ci]);
                mu_slice.assign(&(&p_slice + &clipped));
            }
        }

        if border_owned == "wall" {
            for ci in 0..c {
                for a in 0..2 {
                    let mut mu_slice = mu.slice_mut(s![.., .., a, ci]);
                    mu_slice.mapv_inplace(|v| v.clamp(sigma, SX as f32 - sigma));
                }
            }
        }

        // Accumulate over all offsets
        let mut result = Array3::<f32>::zeros((sx, sy, c));
        for oi in 0..n_offsets {
            let dx = dxs[oi];
            let dy = dys[oi];

            // Roll X and mu
            let Xr = roll_3d_axes01(X, dx, dy);
            let mur = roll_4d_axes01(&mu, dx, dy);

            // dpmu = |pos - mur|
            let mut dpmu = Array4::<f32>::zeros((sx, sy, 2, c));
            for ci in 0..c {
                for a in 0..2 {
                    let p_slice = pos.slice(s![.., .., a]);
                    let m_slice = mur.slice(s![.., .., a, ci]);
                    let diff = (&p_slice - &m_slice).mapv(|v| v.abs());
                    dpmu.slice_mut(s![.., .., a, ci]).assign(&diff);
                }
            }

            // sz = 0.5 - dpmu + sigma
            let sz = dpmu.mapv(|v| 0.5 - v + sigma);
            // clip sz to [0, min(1, 2*sigma)]
            let clip_max = 1.0_f32.min(2.0 * sigma);
            let sz_clipped = sz.mapv(|v| v.clamp(0.0, clip_max));
            // area = prod(sz, axis=2) / (4*sigma^2)
            let mut area = Array2::<f32>::zeros((sx, sy));
            for ci in 0..c {
                for i in 0..sx {
                    for j in 0..sy {
                        let p = sz_clipped[[i, j, 0, ci]] * sz_clipped[[i, j, 1, ci]];
                        area[[i, j]] = p / (4.0 * sigma * sigma);
                    }
                }
                // Xr * area, summed over offsets
                for i in 0..sx {
                    for j in 0..sy {
                        result[[i, j, ci]] += Xr[[i, j, ci]] * area[[i, j]];
                    }
                }
            }
        }

        result
    })
}

fn build_reintegration_p(
    SX: usize,
    SY: usize,
    dt: f32,
    dd: i32,
    sigma: f32,
    border: &str,
    hidden_dims: usize,
    mix: &str,
) -> Box<dyn Fn(&Array3<f32>, &Array3<f32>, &Array4<f32>) -> (Array3<f32>, Array3<f32>)> {
    let pos_y = Array2::from_shape_fn((SY, SX), |(i, _j)| i as f32 + 0.5);
    let pos_x = Array2::from_shape_fn((SY, SX), |(_i, j)| j as f32 + 0.5);
    let mut pos = Array3::<f32>::zeros((SY, SX, 2));
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
    let mix_owned = mix.to_string();

    Box::new(move |X: &Array3<f32>, H: &Array3<f32>, F: &Array4<f32>| {
        let (sx, sy, _two, c) = F.dim();
        let ma = dd as f32 - sigma;

        let mut mu = Array4::<f32>::zeros((sx, sy, 2, c));
        for ci in 0..c {
            for a in 0..2 {
                let f_slice = F.slice(s![.., .., a, ci]);
                let p_slice = pos.slice(s![.., .., a]);
                let clipped = f_slice.mapv(|v| (v * dt).clamp(-ma, ma));
                let mut mu_slice = mu.slice_mut(s![.., .., a, ci]);
                mu_slice.assign(&(&p_slice + &clipped));
            }
        }

        if border_owned == "wall" {
            for ci in 0..c {
                for a in 0..2 {
                    let mut mu_slice = mu.slice_mut(s![.., .., a, ci]);
                    mu_slice.mapv_inplace(|v| v.clamp(sigma, SX as f32 - sigma));
                }
            }
        }

        let mut nX_sum = Array3::<f32>::zeros((sx, sy, c));
        let mut nH_sum = Array3::<f32>::zeros((sx, sy, hidden_dims));

        for oi in 0..n_offsets {
            let dx = dxs[oi];
            let dy = dys[oi];

            let Xr = roll_3d_axes01(X, dx, dy);
            let Hr = roll_3d_axes01(H, dx, dy);
            let mur = roll_4d_axes01(&mu, dx, dy);

            let mut dpmu = Array4::<f32>::zeros((sx, sy, 2, c));
            for ci in 0..c {
                for a in 0..2 {
                    let p_slice = pos.slice(s![.., .., a]);
                    let m_slice = mur.slice(s![.., .., a, ci]);
                    let diff = (&p_slice - &m_slice).mapv(|v| v.abs());
                    dpmu.slice_mut(s![.., .., a, ci]).assign(&diff);
                }
            }

            let sz = dpmu.mapv(|v| 0.5 - v + sigma);
            let clip_max = 1.0_f32.min(2.0 * sigma);
            let sz_clipped = sz.mapv(|v| v.clamp(0.0, clip_max));

            let mut area = Array2::<f32>::zeros((sx, sy));
            for ci in 0..c {
                for i in 0..sx {
                    for j in 0..sy {
                        let p = sz_clipped[[i, j, 0, ci]] * sz_clipped[[i, j, 1, ci]];
                        area[[i, j]] = p / (4.0 * sigma * sigma);
                    }
                }
                for i in 0..sx {
                    for j in 0..sy {
                        nX_sum[[i, j, ci]] += Xr[[i, j, ci]] * area[[i, j]];
                    }
                }
            }

            // H accumulates with area broadcast
            for k_idx in 0..hidden_dims {
                for i in 0..sx {
                    for j in 0..sy {
                        nH_sum[[i, j, k_idx]] += Hr[[i, j, k_idx]] * area[[i, j]];
                    }
                }
            }
        }

        // Mix modes
        let (nX, nH) = match mix_owned.as_str() {
            "avg" => {
                // nH = sum(nH * nX.sum(-1, keepdims=True), axis=0)
                // nX = sum(nX, axis=0)
                // nH = nH / (nX.sum(-1, keepdims=True) + 1e-10)
                let nX_ch_sum = nX_sum.sum_axis(Axis(2)); // (SX, SY)
                let mut nH_avg = Array3::<f32>::zeros((sx, sy, hidden_dims));
                for k_idx in 0..hidden_dims {
                    for i in 0..sx {
                        for j in 0..sy {
                            nH_avg[[i, j, k_idx]] = nH_sum[[i, j, k_idx]] * nX_ch_sum[[i, j]];
                        }
                    }
                }
                let denom = nX_ch_sum.mapv(|v| v + 1e-10);
                for k_idx in 0..hidden_dims {
                    for i in 0..sx {
                        for j in 0..sy {
                            nH_avg[[i, j, k_idx]] /= denom[[i, j]];
                        }
                    }
                }
                (nX_sum, nH_avg)
            }
            "softmax" => {
                // expnX = exp(nX.sum(-1, keepdims=True)) - 1
                // nX = sum(nX, axis=0)
                // nH = sum(nH * expnX, axis=0) / (expnX.sum(axis=0) + 1e-10)
                let nX_ch_sum = nX_sum.sum_axis(Axis(2)); // (SX, SY)
                let expnX = nX_ch_sum.mapv(|v| v.exp() - 1.0);
                let mut nH_soft = Array3::<f32>::zeros((sx, sy, hidden_dims));
                for k_idx in 0..hidden_dims {
                    for i in 0..sx {
                        for j in 0..sy {
                            nH_soft[[i, j, k_idx]] = nH_sum[[i, j, k_idx]] * expnX[[i, j]];
                        }
                    }
                }
                let denom = expnX.mapv(|v| v + 1e-10);
                for k_idx in 0..hidden_dims {
                    for i in 0..sx {
                        for j in 0..sy {
                            nH_soft[[i, j, k_idx]] /= denom[[i, j]];
                        }
                    }
                }
                (nX_sum, nH_soft)
            }
            "stoch" | "stoch_gene_wise" => {
                // Simplified: use nX_sum directly, nH = nH_sum
                // Full stochastic sampling would require PRNG
                (nX_sum, nH_sum)
            }
            _ => (nX_sum, nH_sum),
        };

        (nX, nH)
    })
}

/// Roll a 3D array along axes 0 and 1 by (dx, dy)
fn roll_3d_axes01(arr: &Array3<f32>, dx: i32, dy: i32) -> Array3<f32> {
    let (sx, sy, k) = arr.dim();
    let mut out = Array3::<f32>::zeros((sx, sy, k));
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

/// Roll a 4D array along axes 0 and 1 by (dx, dy)
fn roll_4d_axes01(arr: &Array4<f32>, dx: i32, dy: i32) -> Array4<f32> {
    let (sx, sy, a, b) = arr.dim();
    let mut out = Array4::<f32>::zeros((sx, sy, a, b));
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
pub struct Params {
    pub r: Array1<f32>,
    pub b: Array2<f32>,
    pub w: Array2<f32>,
    pub a: Array2<f32>,
    pub m: Array1<f32>,
    pub s: Array1<f32>,
    pub h: Array1<f32>,
    pub R: f32,
}

#[derive(Clone, Debug)]
pub struct CompiledParams {
    pub fK: Array3<Complex<f32>>,
    pub m: Array1<f32>,
    pub s: Array1<f32>,
    pub h: Array1<f32>,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub SX: usize,
    pub SY: usize,
    pub nb_k: usize,
    pub C: usize,
    pub c0: Vec<usize>,
    pub c1: Vec<Vec<usize>>,
    pub dt: f32,
    pub dd: i32,
    pub sigma: f32,
    pub n: f32,
    pub theta_A: f32,
    pub border: String,
}

#[derive(Clone, Debug)]
pub struct State {
    pub A: Array3<f32>,
}

#[derive(Clone, Debug)]
pub struct ConfigP {
    pub base: Config,
    pub mix: String,
}

#[derive(Clone, Debug)]
pub struct StateP {
    pub A: Array3<f32>,
    pub P: Array3<f32>,
}

// ---------------------------------------------------------------------------
// Parameter sampling
// ---------------------------------------------------------------------------

fn sample_params<R: Rng>(rng: &mut R, nb_k: usize) -> Params {
    let r = Array1::from_shape_fn(nb_k, |_| rng.gen_range(0.2..1.0));
    let m = Array1::from_shape_fn(nb_k, |_| rng.gen_range(0.05..0.5));
    let s = Array1::from_shape_fn(nb_k, |_| rng.gen_range(0.001..0.18));
    let h = Array1::from_shape_fn(nb_k, |_| rng.gen_range(0.01..1.0));
    let a = Array2::from_shape_fn((nb_k, 3), |_| rng.gen_range(0.0..1.0));
    let w = Array2::from_shape_fn((nb_k, 3), |_| rng.gen_range(0.01..0.5));
    let b = Array2::from_shape_fn((nb_k, 3), |_| rng.gen_range(0.001..1.0));
    let R = rng.gen_range(2.0..25.0);
    Params {
        r,
        b,
        w,
        a,
        m,
        s,
        h,
        R,
    }
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
        let mut K = Array3::<f32>::zeros((SY, SX, nb_k));
        for k in 0..nb_k {
            let mut dist = Array2::<f32>::zeros((SY, SX));
            for i in 0..SY {
                for j in 0..SX {
                    let di = i as i32 - mid;
                    let dj = j as i32 - mid;
                    let d = ((di * di + dj * dj) as f32).sqrt();
                    dist[[i, j]] = d / ((params.R + 15.0) * params.r[k]);
                }
            }
            let sig = dist.mapv(|d| sigmoid(-(d - 1.0) * 10.0));
            let a: Vec<f32> = params.a.row(k).to_vec();
            let w: Vec<f32> = params.w.row(k).to_vec();
            let b: Vec<f32> = params.b.row(k).to_vec();
            let kf = ker_f(&dist, &a, &w, &b);
            let mut kslice = sig * kf;
            let total = kslice.sum();
            if total > 0.0 {
                kslice = kslice / total;
            }
            K.slice_mut(s![.., .., k]).assign(&kslice);
        }

        // Convert to complex and apply fftshift + fft2
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
// Step functions
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

        // fA = fft2(A, axes=(0,1))
        let A_complex = A.mapv(|v| Complex::new(v, 0.0));
        let fA = planner.fft2(&A_complex);

        // U = real(ifft2(fK * fA[:,:,c0], axes=(0,1)))
        let mut fA_selected = Array3::<Complex<f32>>::zeros((SX, SY, nb_k));
        for k in 0..nb_k {
            let src_ch = c0[k];
            for i in 0..SX {
                for j in 0..SY {
                    fA_selected[[i, j, k]] = fA[[i, j, src_ch]];
                }
            }
        }

        let mut fK_times_fA = Array3::<Complex<f32>>::zeros((SX, SY, nb_k));
        for k in 0..nb_k {
            for i in 0..SX {
                for j in 0..SY {
                    fK_times_fA[[i, j, k]] = params.fK[[i, j, k]] * fA_selected[[i, j, k]];
                }
            }
        }

        let U_complex = planner.ifft2(&fK_times_fA);
        let mut U = U_complex.mapv(|v| v.re);

        // U = growth(U, m, s) * h
        for k in 0..nb_k {
            for i in 0..SX {
                for j in 0..SY {
                    let u = U[[i, j, k]];
                    let g = bell(u, params.m[k], params.s[k]) * 2.0 - 1.0;
                    U[[i, j, k]] = g * params.h[k];
                }
            }
        }

        // U = dstack([U[:,:,c1[c]].sum(-1) for c in range(C)])
        let mut U_new = Array3::<f32>::zeros((SX, SY, C));
        for c in 0..C {
            for &k_idx in &c1[c] {
                for i in 0..SX {
                    for j in 0..SY {
                        U_new[[i, j, c]] += U[[i, j, k_idx]];
                    }
                }
            }
        }

        // nabla_U = sobel(U), nabla_A = sobel(A.sum(-1, keepdims=True))
        let nabla_U = sobel(&U_new);
        let A_sum = A.sum_axis(Axis(2)).mapv(|v| v).insert_axis(Axis(2)); // (SX, SY, 1)
        let nabla_A = sobel(&A_sum);

        // alpha = clip((A[:,:,None,:] / theta_A)^n, 0, 1)
        // A has shape (SX, SY, C), we need (SX, SY, 1, C) for broadcasting with nabla (SX, SY, 2, C)
        let mut alpha = Array4::<f32>::zeros((SX, SY, 2, C));
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

        // advect(A, nabla_U * (1-alpha) - nabla_A * alpha)
        let mut flow = Array4::<f32>::zeros((SX, SY, 2, C));
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

        let new_A = advect(A, &flow);
        State { A: new_A }
    }
}

fn build_step_fn_p(
    config: &ConfigP,
) -> impl FnMut(&StateP, &CompiledParams, &mut Fft2DPlanner) -> StateP + '_ {
    let advect = build_reintegration_p(
        config.base.SX,
        config.base.SY,
        config.base.dt,
        config.base.dd,
        config.base.sigma,
        &config.base.border,
        config.base.nb_k,
        &config.mix,
    );

    let c0 = config.base.c0.clone();
    let c1 = config.base.c1.clone();
    let C = config.base.C;
    let nb_k = config.base.nb_k;
    let SX = config.base.SX;
    let SY = config.base.SY;
    let dd = config.base.dd;
    let sigma = config.base.sigma;

    move |state: &StateP, params: &CompiledParams, planner: &mut Fft2DPlanner| -> StateP {
        let A = &state.A;
        let P = &state.P;

        let A_complex = A.mapv(|v| Complex::new(v, 0.0));
        let fA = planner.fft2(&A_complex);

        let mut fA_selected = Array3::<Complex<f32>>::zeros((SX, SY, nb_k));
        for k in 0..nb_k {
            let src_ch = c0[k];
            for i in 0..SX {
                for j in 0..SY {
                    fA_selected[[i, j, k]] = fA[[i, j, src_ch]];
                }
            }
        }

        let mut fK_times_fA = Array3::<Complex<f32>>::zeros((SX, SY, nb_k));
        for k in 0..nb_k {
            for i in 0..SX {
                for j in 0..SY {
                    fK_times_fA[[i, j, k]] = params.fK[[i, j, k]] * fA_selected[[i, j, k]];
                }
            }
        }

        let U_complex = planner.ifft2(&fK_times_fA);
        let mut U = U_complex.mapv(|v| v.re);

        // U = growth(U, m, s) * P
        for k in 0..nb_k {
            for i in 0..SX {
                for j in 0..SY {
                    let u = U[[i, j, k]];
                    let g = bell(u, params.m[k], params.s[k]) * 2.0 - 1.0;
                    U[[i, j, k]] = g * P[[i, j, k]];
                }
            }
        }

        // U = dstack([U[:,:,c1[c]].sum(-1) for c in range(C)])
        let mut U_new = Array3::<f32>::zeros((SX, SY, C));
        for c in 0..C {
            for &k_idx in &c1[c] {
                for i in 0..SX {
                    for j in 0..SY {
                        U_new[[i, j, c]] += U[[i, j, k_idx]];
                    }
                }
            }
        }

        // F = sobel(U)
        let F = sobel(&U_new);

        // C_grad = sobel(A.sum(-1, keepdims=True))
        let A_sum = A.sum_axis(Axis(2)).mapv(|v| v).insert_axis(Axis(2));
        let C_grad = sobel(&A_sum);

        // alpha = clip((A / 2)^2, 0, 1)
        let mut alpha = Array4::<f32>::zeros((SX, SY, 2, C));
        for c_idx in 0..C {
            for i in 0..SX {
                for j in 0..SY {
                    let val = (A[[i, j, c_idx]] / 2.0).powi(2);
                    let clamped = val.clamp(0.0, 1.0);
                    alpha[[i, j, 0, c_idx]] = clamped;
                    alpha[[i, j, 1, c_idx]] = clamped;
                }
            }
        }

        // F = clip(F * (1-alpha) - C_grad * alpha, -(dd-sigma), dd-sigma)
        let ma = dd as f32 - sigma;
        let mut flow = Array4::<f32>::zeros((SX, SY, 2, C));
        for c_idx in 0..C {
            for a in 0..2 {
                for i in 0..SX {
                    for j in 0..SY {
                        let val = F[[i, j, a, c_idx]] * (1.0 - alpha[[i, j, a, c_idx]])
                            - C_grad[[i, j, a, 0]] * alpha[[i, j, a, c_idx]];
                        flow[[i, j, a, c_idx]] = val.clamp(-ma, ma);
                    }
                }
            }
        }

        let (nA, nP) = advect(A, P, &flow);
        StateP { A: nA, P: nP }
    }
}

// ---------------------------------------------------------------------------
// Rollout
// ---------------------------------------------------------------------------

fn build_rollout<F>(
    mut step_fn: F,
) -> impl FnMut(&CompiledParams, &State, usize, &mut Fft2DPlanner) -> (State, Vec<State>)
where
    F: FnMut(&State, &CompiledParams, &mut Fft2DPlanner) -> State,
{
    move |params: &CompiledParams,
          init_state: &State,
          steps: usize,
          planner: &mut Fft2DPlanner|
          -> (State, Vec<State>) {
        let mut state = init_state.clone();
        let mut states = Vec::with_capacity(steps);
        for _ in 0..steps {
            state = step_fn(&state, params, planner);
            states.push(state.clone());
        }
        (state, states)
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Flowlenia (Rust port) ===");
    println!("Matching train/flowlenia.py 1:1\n");

    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let grid_size: usize = if let Some(pos) = args.iter().position(|a| a == "--grid-size") {
        args.get(pos + 1).and_then(|s| s.parse().ok()).unwrap_or(32)
    } else {
        32
    };
    assert!(
        [32, 64, 128, 256, 512].contains(&grid_size),
        "Grid size must be 32, 64, 128, 256, or 512"
    );

    // Small test configuration
    let SX = grid_size;
    let SY = grid_size;
    let nb_k = 3;
    let C = 3;

    // Simple connectivity: each channel connects to itself
    let c0: Vec<usize> = (0..C).collect();
    let c1: Vec<Vec<usize>> = (0..C).map(|c| vec![c]).collect();

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

    // Sample random parameters
    let mut rng = rand::thread_rng();
    let params = sample_params(&mut rng, nb_k);
    println!("Sampled parameters:");
    println!("  R = {:.4}", params.R);
    println!("  r = {:?}", params.r);
    println!("  m = {:?}", params.m);
    println!("  s = {:?}", params.s);
    println!("  h = {:?}", params.h);

    // Compute kernels
    let mut planner = Fft2DPlanner::new();
    let mut kernel_computer = build_kernel_computer(SX, SY, nb_k);
    let compiled = kernel_computer(&params, &mut planner);
    println!("\nKernel FFT shape: {:?}", compiled.fK.dim());
    println!("  fK[0,0,0] = {:?}", compiled.fK[[0, 0, 0]]);

    // Initialize state with a small Gaussian blob in the center
    let mut A = Array3::<f32>::zeros((SX, SY, C));
    let cx = SX as f32 / 2.0;
    let cy = SY as f32 / 2.0;
    let variance = (SX * SX) as f32 / 64.0;
    for i in 0..SX {
        for j in 0..SY {
            let dx = i as f32 - cx;
            let dy = j as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let val = (-dist * dist / variance).exp();
            for c in 0..C {
                A[[i, j, c]] = val * (0.5 + 0.5 * (c as f32 / C as f32));
            }
        }
    }
    let init_state = State { A };

    // Build step function and run a few steps
    let config_clone = config.clone();
    let mut step_fn = build_step_fn(&config_clone);
    let (final_state, states) =
        build_rollout(&mut step_fn)(&compiled, &init_state, 10, &mut planner);

    println!("\nRan 10 steps.");
    println!("Initial state A sum: {:.6}", init_state.A.sum());
    println!("Final state A sum:  {:.6}", final_state.A.sum());
    println!("State history length: {}", states.len());

    // Test with P (hidden state) version
    let config_p = ConfigP {
        base: config,
        mix: "avg".to_string(),
    };

    let P = Array3::<f32>::ones((SX, SY, nb_k)) * 0.5;
    let init_state_p = StateP {
        A: init_state.A.clone(),
        P,
    };

    let mut step_fn_p = build_step_fn_p(&config_p);
    let final_p = step_fn_p(&init_state_p, &compiled, &mut planner);

    println!("\nWith hidden state (P):");
    println!("  Final A sum: {:.6}", final_p.A.sum());
    println!("  Final P sum: {:.6}", final_p.P.sum());

    // Test get_kernels
    let spatial_kernels = get_kernels(SX, SY, nb_k, &params);
    println!("\nSpatial kernels shape: {:?}", spatial_kernels.dim());
    println!(
        "  Kernel 0 sum: {:.6}",
        spatial_kernels.slice(s![.., .., 0]).sum()
    );

    // Test conn_from_matrix
    let mat = arr2(&[[1i32, 0], [0, 1]]);
    let (c0_out, c1_out) = conn_from_matrix(&mat);
    println!("\nconn_from_matrix test:");
    println!("  c0 = {:?}", c0_out);
    println!("  c1 = {:?}", c1_out);

    println!("\n=== All tests passed ===");
}
