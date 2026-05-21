//! Core math operations for transformer inference.
//!
//! All operations are pure CPU, no BLAS dependency.
//! Quantized matmuls are in the `quantized` module.

use std::f32::consts;

/// Dot product of two f32 vectors.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Add two vectors in-place: `a += b`.
#[inline]
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += y;
    }
}

/// Multiply a vector by a scalar in-place: `a *= s`.
#[inline]
pub fn scale_inplace(a: &mut [f32], s: f32) {
    for x in a.iter_mut() {
        *x *= s;
    }
}

/// Copy values from `src` to `dst`.
#[inline]
pub fn copy(dst: &mut [f32], src: &[f32]) {
    dst.copy_from_slice(src);
}

// ── Matrix Multiply ────────────────────────────────────────────────

/// Naive matrix-multiply: C = A @ B
///
/// - A: [M, K] row-major
/// - B: [K, N] row-major
/// - C: [M, N] row-major (output)
pub fn matmul(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for t in 0..k {
                sum += a[i * k + t] * b[t * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Matrix-vector multiply: y = A @ x
///
/// - A: [M, K] row-major
/// - x: [K]
/// - y: [M] (output)
pub fn matvec(m: usize, k: usize, a: &[f32], x: &[f32], y: &mut [f32]) {
    assert_eq!(a.len(), m * k);
    assert_eq!(x.len(), k);
    assert_eq!(y.len(), m);

    for i in 0..m {
        y[i] = dot(&a[i * k..(i + 1) * k], x);
    }
}

/// Matrix-vector multiply, transposed: y = A^T @ x
///
/// - A: [K, M] row-major (so A^T is M x K)
/// - x: [K]
/// - y: [M] (output)
pub fn matvec_transpose(m: usize, k: usize, a: &[f32], x: &[f32], y: &mut [f32]) {
    // A is K x M stored row-major. A^T is M x K.
    // y[j] = sum_i A[i][j] * x[i]
    assert_eq!(a.len(), k * m);
    assert_eq!(x.len(), k);
    assert_eq!(y.len(), m);

    y.fill(0.0);
    for i in 0..k {
        let xi = x[i];
        for j in 0..m {
            y[j] += a[i * m + j] * xi;
        }
    }
}

// ── RMSNorm ────────────────────────────────────────────────────────

/// Root Mean Square Layer Normalization.
///
/// y = x * rms(x)^-1 * w
/// where rms(x) = sqrt(mean(x^2) + eps)
pub fn rms_norm(x: &mut [f32], w: &[f32], eps: f64) {
    let n = x.len();
    let mut ss = 0.0f64;
    for &v in x.iter() {
        ss += (v as f64) * (v as f64);
    }
    let rms = (ss / n as f64 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..n {
        x[i] = (x[i] as f64 * inv_rms * w[i] as f64) as f32;
    }
}

// ── RoPE (Rotary Position Embedding) ──────────────────────────────

/// Apply Rotary Position Embedding to a query or key vector.
///
/// For each pair (d, d+1) in the head_dim, with position `pos`:
///   x_d     = x_d * cos(θ) - x_{d+1} * sin(θ)
///   x_{d+1} = x_d * sin(θ) + x_{d+1} * cos(θ)
/// where θ = pos / base^(2d/head_dim)
pub fn apply_rope(x: &mut [f32], pos: usize, head_dim: usize, base: f32) {
    for i in (0..head_dim).step_by(2) {
        let theta = pos as f32 / base.powf(2.0 * (i as f32) / head_dim as f32);
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let x0 = x[i];
        let x1 = x[i + 1];
        x[i] = x0 * cos_theta - x1 * sin_theta;
        x[i + 1] = x0 * sin_theta + x1 * cos_theta;
    }
}

/// Apply RoPE to a full query/key matrix [n_heads, head_dim].
/// Each head gets its own RoPE application.
pub fn apply_rope_multi(x: &mut [f32], pos: usize, n_heads: usize, head_dim: usize, base: f32) {
    for h in 0..n_heads {
        let start = h * head_dim;
        apply_rope(&mut x[start..start + head_dim], pos, head_dim, base);
    }
}

// ── Activation Functions ──────────────────────────────────────────

/// SiLU (Swish) activation: x * sigmoid(x)
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Apply SiLU element-wise in-place.
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = silu(*v);
    }
}

/// GELU activation (Gaussian Error Linear Unit, tanh approximation).
#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((consts::SQRT_2 / consts::PI) * (x + 0.044715 * x * x * x)).tanh())
}

// ── Softmax ────────────────────────────────────────────────────────

/// Compute softmax in-place over a vector.
pub fn softmax(x: &mut [f32]) {
    let max_val = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

// ── Attention helpers ──────────────────────────────────────────────

/// Apply causal mask (upper triangular) to attention scores, in-place.
/// scores has shape [n_heads, q_len, kv_len].
/// The mask zeroes out positions where q_pos < kv_pos (future tokens).
pub fn apply_causal_mask(scores: &mut [f32], n_heads: usize, q_len: usize, kv_len: usize) {
    for h in 0..n_heads {
        for qi in 0..q_len {
            for ki in 0..kv_len {
                if qi < ki || (qi + (kv_len - q_len)) < ki {
                    // Position this q token shouldn't attend to this k token
                    let idx = h * q_len * kv_len + qi * kv_len + ki;
                    // Use a very negative value (effectively -inf)
                    scores[idx] = -65504.0;
                }
            }
        }
    }
}
