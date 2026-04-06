use rand_distr::{Distribution, Normal};

/// Data type for tensor storage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType {
    F32,
    BF16,
    F16,
}

impl DType {
    pub fn bytes_per_element(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::BF16 | DType::F16 => 2,
        }
    }
}

/// BF16 ↔ F32 conversion utilities.
/// BF16: 1 sign + 8 exponent + 7 mantissa (same range as f32, less precision).
#[inline(always)]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    // Round to nearest even
    let rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

#[inline(always)]
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// F16 ↔ F32 conversion utilities.
#[inline(always)]
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    if exp == 255 {
        // Inf/NaN
        return sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 };
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7C00; // overflow → Inf
    }
    if new_exp <= 0 {
        return sign; // underflow → 0
    }

    sign | ((new_exp as u16) << 10) | ((mantissa >> 13) as u16)
}

#[inline(always)]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 { return f32::from_bits(sign << 31); }
        // Subnormal
        let mut m = mantissa;
        let mut e: i32 = -14;
        while m & 0x400 == 0 { m <<= 1; e -= 1; }
        m &= 0x3FF;
        let f32_exp = ((e + 127) as u32) & 0xFF;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exp == 31 {
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mantissa << 13));
    }
    let f32_exp = (exp as i32 - 15 + 127) as u32;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
}

/// Multi-dimensional tensor. All compute happens in f32.
/// BF16/F16 is used for storage (GGUF, weight serialization, memory savings).
/// Data is always kept as Vec<f32> in memory for compute speed.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub dtype: DType,
}

impl Tensor {
    /// Create tensor filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape: shape.to_vec(),
            strides: compute_strides(shape),
            dtype: DType::F32,
        }
    }

    /// Create tensor filled with ones.
    pub fn ones(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![1.0; size],
            shape: shape.to_vec(),
            strides: compute_strides(shape),
            dtype: DType::F32,
        }
    }

    /// Create tensor from a flat slice with given shape.
    pub fn from_slice(data: &[f32], shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        assert_eq!(data.len(), size, "Data length {} != shape product {}", data.len(), size);
        Self {
            data: data.to_vec(),
            shape: shape.to_vec(),
            strides: compute_strides(shape),
            dtype: DType::F32,
        }
    }

    /// Create tensor with values from N(0, std).
    pub fn randn(shape: &[usize], std: f64) -> Self {
        let size: usize = shape.iter().product();
        let normal = Normal::new(0.0, std).unwrap();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng) as f32).collect();
        Self {
            data,
            shape: shape.to_vec(),
            strides: compute_strides(shape),
            dtype: DType::F32,
        }
    }

    /// Create tensor filled with a scalar value.
    pub fn full(shape: &[usize], value: f32) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![value; size],
            shape: shape.to_vec(),
            strides: compute_strides(shape),
            dtype: DType::F32,
        }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get element at flat index.
    pub fn get_flat(&self, idx: usize) -> f32 {
        self.data[idx]
    }

    /// Set element at flat index.
    pub fn set_flat(&mut self, idx: usize, val: f32) {
        self.data[idx] = val;
    }

    /// Reshape tensor (must have same total elements).
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(self.numel(), new_size, "Cannot reshape {} -> {}", self.numel(), new_size);
        Self {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            strides: compute_strides(new_shape),
            dtype: self.dtype,
        }
    }

    /// Transpose two dimensions.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        assert!(dim0 < self.ndim() && dim1 < self.ndim());
        if dim0 == dim1 {
            return self.clone();
        }

        let mut new_shape = self.shape.clone();
        new_shape.swap(dim0, dim1);
        let new_size: usize = new_shape.iter().product();
        let mut result = vec![0.0f32; new_size];
        let new_strides = compute_strides(&new_shape);

        // General transpose via index mapping
        for i in 0..self.numel() {
            let mut remaining = i;
            let mut src_indices = vec![0usize; self.ndim()];
            for d in 0..self.ndim() {
                src_indices[d] = remaining / self.strides[d];
                remaining %= self.strides[d];
            }
            // Swap dimensions for destination
            let mut dst_indices = src_indices.clone();
            dst_indices.swap(dim0, dim1);
            let dst_flat: usize = dst_indices.iter().zip(new_strides.iter()).map(|(i, s)| i * s).sum();
            result[dst_flat] = self.data[i];
        }

        Self {
            data: result,
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        }
    }

    /// Scale all elements by a scalar.
    pub fn scale(&self, s: f32) -> Self {
        Self {
            data: self.data.iter().map(|x| x * s).collect(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
        }
    }

    /// Scale in-place.
    pub fn scale_inplace(&mut self, s: f32) {
        for x in self.data.iter_mut() {
            *x *= s;
        }
    }

    /// Serialize data to BF16 bytes (2 bytes per element).
    pub fn to_bf16_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.numel() * 2);
        for &val in &self.data {
            let bf16 = f32_to_bf16(val);
            bytes.extend_from_slice(&bf16.to_le_bytes());
        }
        bytes
    }

    /// Create tensor from BF16 bytes.
    pub fn from_bf16_bytes(bytes: &[u8], shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(bytes.len(), numel * 2, "BF16 byte count mismatch");
        let data: Vec<f32> = bytes.chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        Self {
            data,
            shape: shape.to_vec(),
            strides: compute_strides(shape),
            dtype: DType::BF16,
        }
    }

    /// Memory size in bytes for the given dtype.
    pub fn memory_bytes(&self, dtype: DType) -> usize {
        self.numel() * dtype.bytes_per_element()
    }

    /// Add in-place: self += other
    pub fn add_inplace(&mut self, other: &Tensor) {
        assert_eq!(self.numel(), other.numel());
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
    }
}

// ---- Basic Operations ----

/// Element-wise addition. Shapes must match.
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape, b.shape, "Shape mismatch for add: {:?} vs {:?}", a.shape, b.shape);
    Tensor {
        data: a.data.iter().zip(b.data.iter()).map(|(x, y)| x + y).collect(),
        shape: a.shape.clone(),
        strides: a.strides.clone(),
        dtype: a.dtype,
    }
}

/// Element-wise multiplication. Shapes must match.
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape, b.shape, "Shape mismatch for mul: {:?} vs {:?}", a.shape, b.shape);
    Tensor {
        data: a.data.iter().zip(b.data.iter()).map(|(x, y)| x * y).collect(),
        shape: a.shape.clone(),
        strides: a.strides.clone(),
        dtype: a.dtype,
    }
}

/// Matrix multiplication. a: (M, K), b: (K, N) -> (M, N).
/// Also supports batched: (..., M, K) x (..., K, N) -> (..., M, N).
/// Uses tiled algorithm for cache efficiency (~5-10x faster than naive).
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(a.ndim() >= 2 && b.ndim() >= 2, "matmul requires at least 2D tensors");
    let m = a.shape[a.ndim() - 2];
    let k_a = a.shape[a.ndim() - 1];
    let k_b = b.shape[b.ndim() - 2];
    let n = b.shape[b.ndim() - 1];
    assert_eq!(k_a, k_b, "matmul inner dimensions must match: {} vs {}", k_a, k_b);

    if a.ndim() == 2 && b.ndim() == 2 {
        let mut result = vec![0.0f32; m * n];
        tiled_matmul(&a.data, &b.data, &mut result, m, k_a, n);
        Tensor::from_slice(&result, &[m, n])
    } else {
        let batch_dims_a = &a.shape[..a.ndim() - 2];
        let batch_dims_b = &b.shape[..b.ndim() - 2];
        assert_eq!(batch_dims_a, batch_dims_b, "Batch dimensions must match");
        let batch_size: usize = batch_dims_a.iter().product();
        let a_stride = m * k_a;
        let b_stride = k_a * n;
        let c_stride = m * n;

        let mut result = vec![0.0f32; batch_size * c_stride];
        for batch in 0..batch_size {
            let a_off = batch * a_stride;
            let b_off = batch * b_stride;
            let c_off = batch * c_stride;
            tiled_matmul(
                &a.data[a_off..a_off + a_stride],
                &b.data[b_off..b_off + b_stride],
                &mut result[c_off..c_off + c_stride],
                m, k_a, n,
            );
        }
        let mut out_shape = batch_dims_a.to_vec();
        out_shape.push(m);
        out_shape.push(n);
        Tensor::from_slice(&result, &out_shape)
    }
}

/// Tiled matmul: C += A @ B with cache-friendly tiling.
/// A: (M, K), B: (K, N), C: (M, N).
/// Tile size chosen for L1 cache (~64KB on M4).
fn tiled_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    const TILE: usize = 32; // 32x32 tiles fit in L1 cache

    // Loop order: tile_k → tile_i → tile_j for maximum reuse
    let mut kk = 0;
    while kk < k {
        let k_end = (kk + TILE).min(k);
        let mut ii = 0;
        while ii < m {
            let i_end = (ii + TILE).min(m);
            let mut jj = 0;
            while jj < n {
                let j_end = (jj + TILE).min(n);

                // Micro-kernel: multiply tile
                for i in ii..i_end {
                    for kt in kk..k_end {
                        let a_val = a[i * k + kt];
                        let c_row = i * n;
                        for j in jj..j_end {
                            c[c_row + j] += a_val * b[kt * n + j];
                        }
                    }
                }

                jj += TILE;
            }
            ii += TILE;
        }
        kk += TILE;
    }
}

/// Softmax along the last dimension.
pub fn softmax(a: &Tensor, dim: isize) -> Tensor {
    let ndim = a.ndim();
    let dim = if dim < 0 { (ndim as isize + dim) as usize } else { dim as usize };
    assert!(dim < ndim);

    let outer: usize = a.shape[..dim].iter().product();
    let inner = a.shape[dim];
    let trailing: usize = a.shape[dim + 1..].iter().product();

    let mut result = a.data.clone();

    for o in 0..outer {
        for t in 0..trailing {
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..inner {
                let idx = o * inner * trailing + i * trailing + t;
                max_val = max_val.max(result[idx]);
            }
            // Compute exp and sum
            let mut sum = 0.0f32;
            for i in 0..inner {
                let idx = o * inner * trailing + i * trailing + t;
                result[idx] = (result[idx] - max_val).exp();
                sum += result[idx];
            }
            // Normalize
            for i in 0..inner {
                let idx = o * inner * trailing + i * trailing + t;
                result[idx] /= sum;
            }
        }
    }

    Tensor {
        data: result,
        shape: a.shape.clone(),
        strides: a.strides.clone(),
        dtype: a.dtype,
    }
}

/// Element-wise sigmoid: 1 / (1 + exp(-x)).
pub fn sigmoid(a: &Tensor) -> Tensor {
    Tensor {
        data: a.data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect(),
        shape: a.shape.clone(),
        strides: a.strides.clone(),
        dtype: a.dtype,
    }
}

/// Element-wise GELU (approximate): x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(a: &Tensor) -> Tensor {
    let sqrt_2_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    Tensor {
        data: a.data.iter().map(|&x| {
            let inner = sqrt_2_pi * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }).collect(),
        shape: a.shape.clone(),
        strides: a.strides.clone(),
        dtype: a.dtype,
    }
}

/// Element-wise SiLU (swish): x * sigmoid(x).
pub fn silu(a: &Tensor) -> Tensor {
    Tensor {
        data: a.data.iter().map(|&x| x / (1.0 + (-x).exp())).collect(),
        shape: a.shape.clone(),
        strides: a.strides.clone(),
        dtype: a.dtype,
    }
}

/// RMS normalization: x / sqrt(mean(x^2) + eps) * weight.
pub fn rms_norm(a: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    // Normalize along last dimension
    let last_dim = *a.shape.last().unwrap();
    assert_eq!(weight.numel(), last_dim, "Weight size must match last dim");
    let n_vecs = a.numel() / last_dim;

    let mut result = vec![0.0f32; a.numel()];
    for v in 0..n_vecs {
        let offset = v * last_dim;
        // Compute RMS
        let mut sum_sq = 0.0f64;
        for i in 0..last_dim {
            let val = a.data[offset + i] as f64;
            sum_sq += val * val;
        }
        let rms = ((sum_sq / last_dim as f64) + eps).sqrt();
        // Normalize and scale
        for i in 0..last_dim {
            result[offset + i] = (a.data[offset + i] as f64 / rms * weight.data[i] as f64) as f32;
        }
    }

    Tensor {
        data: result,
        shape: a.shape.clone(),
        strides: a.strides.clone(),
        dtype: a.dtype,
    }
}

/// Cross-entropy loss: -mean(log(softmax(logits)[targets]))
/// logits: (batch, vocab), targets: (batch,) as indices
pub fn cross_entropy_loss(logits: &Tensor, targets: &[u32]) -> f32 {
    assert_eq!(logits.ndim(), 2);
    let batch = logits.shape[0];
    let vocab = logits.shape[1];
    assert_eq!(targets.len(), batch);

    let probs = softmax(logits, -1);
    let mut total_loss = 0.0f32;
    for b in 0..batch {
        let target = targets[b] as usize;
        assert!(target < vocab);
        let p = probs.data[b * vocab + target].max(1e-10);
        total_loss -= p.ln();
    }
    total_loss / batch as f32
}

// ---- Utilities ----

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_ones() {
        let z = Tensor::zeros(&[2, 3]);
        assert_eq!(z.shape, vec![2, 3]);
        assert_eq!(z.numel(), 6);
        assert!(z.data.iter().all(|&x| x == 0.0));

        let o = Tensor::ones(&[3, 4]);
        assert!(o.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_from_slice() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.data[0], 1.0);
        assert_eq!(t.data[5], 6.0);
    }

    #[test]
    fn test_matmul_2d() {
        // [1 2]   [5 6]   [1*5+2*7  1*6+2*8]   [19 22]
        // [3 4] x [7 8] = [3*5+4*7  3*6+4*8] = [43 50]
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = matmul(&a, &b);
        assert_eq!(c.shape, vec![2, 2]);
        assert!((c.data[0] - 19.0).abs() < 1e-5);
        assert!((c.data[1] - 22.0).abs() < 1e-5);
        assert!((c.data[2] - 43.0).abs() < 1e-5);
        assert!((c.data[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_rect() {
        // (2,3) x (3,4) -> (2,4)
        let a = Tensor::ones(&[2, 3]);
        let b = Tensor::ones(&[3, 4]);
        let c = matmul(&a, &b);
        assert_eq!(c.shape, vec![2, 4]);
        // Each element should be 3.0 (sum of 3 ones)
        assert!(c.data.iter().all(|&x| (x - 3.0).abs() < 1e-5));
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], &[1, 3]);
        let s = softmax(&t, -1);
        let sum: f32 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax must sum to 1, got {sum}");
        // Values should be monotonically increasing
        assert!(s.data[0] < s.data[1]);
        assert!(s.data[1] < s.data[2]);
    }

    #[test]
    fn test_sigmoid() {
        let t = Tensor::from_slice(&[0.0, 100.0, -100.0], &[3]);
        let s = sigmoid(&t);
        assert!((s.data[0] - 0.5).abs() < 1e-5);
        assert!((s.data[1] - 1.0).abs() < 1e-3);
        assert!(s.data[2].abs() < 1e-3);
    }

    #[test]
    fn test_gelu() {
        let t = Tensor::from_slice(&[0.0, 1.0, -1.0], &[3]);
        let g = gelu(&t);
        assert!((g.data[0] - 0.0).abs() < 1e-5, "GELU(0) should be ~0");
        assert!(g.data[1] > 0.8, "GELU(1) should be > 0.8");
        assert!(g.data[2] < 0.0, "GELU(-1) should be < 0");
    }

    #[test]
    fn test_silu() {
        let t = Tensor::from_slice(&[0.0, 10.0, -10.0], &[3]);
        let s = silu(&t);
        assert!((s.data[0] - 0.0).abs() < 1e-5, "SiLU(0) = 0");
        assert!((s.data[1] - 10.0).abs() < 0.01, "SiLU(large) ≈ large");
        assert!(s.data[2].abs() < 0.01, "SiLU(very negative) ≈ 0");
    }

    #[test]
    fn test_rms_norm() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let w = Tensor::ones(&[2]);
        let normed = rms_norm(&x, &w, 1e-6);
        // Check that each row has unit RMS
        for row in 0..2 {
            let mut sum_sq = 0.0f64;
            for col in 0..2 {
                let val = normed.data[row * 2 + col] as f64;
                sum_sq += val * val;
            }
            let rms = (sum_sq / 2.0).sqrt();
            assert!((rms - 1.0).abs() < 0.01, "RMS of row {row} should be ~1.0, got {rms}");
        }
    }

    #[test]
    fn test_cross_entropy() {
        // With uniform logits, loss should be ln(vocab_size)
        let vocab = 100;
        let batch = 4;
        let logits = Tensor::zeros(&[batch, vocab]);
        let targets: Vec<u32> = (0..batch as u32).collect();
        let loss = cross_entropy_loss(&logits, &targets);
        let expected = (vocab as f32).ln();
        assert!((loss - expected).abs() < 0.1, "Uniform logits loss should be ~ln({vocab}) = {expected}, got {loss}");
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let tt = t.transpose(0, 1);
        assert_eq!(tt.shape, vec![3, 2]);
        // Original: [[1,2,3],[4,5,6]], Transposed: [[1,4],[2,5],[3,6]]
        assert_eq!(tt.data[0], 1.0);
        assert_eq!(tt.data[1], 4.0);
        assert_eq!(tt.data[2], 2.0);
        assert_eq!(tt.data[3], 5.0);
    }

    #[test]
    fn test_add_mul() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]);
        let sum = add(&a, &b);
        assert_eq!(sum.data, vec![5.0, 7.0, 9.0]);
        let prod = mul(&a, &b);
        assert_eq!(prod.data, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_randn() {
        let t = Tensor::randn(&[1000], 1.0);
        // Mean should be roughly 0
        let mean: f32 = t.data.iter().sum::<f32>() / 1000.0;
        assert!(mean.abs() < 0.2, "Randn mean should be ~0, got {mean}");
    }

    #[test]
    fn test_bf16_roundtrip() {
        let values = vec![0.0f32, 1.0, -1.0, 3.14159, 0.001, 100.0, -50.5];
        for &v in &values {
            let bf16 = f32_to_bf16(v);
            let back = bf16_to_f32(bf16);
            let rel_err = if v.abs() > 1e-6 { ((back - v) / v).abs() } else { (back - v).abs() };
            assert!(
                rel_err < 0.01,
                "BF16 roundtrip: {v} → {bf16} → {back} (error {rel_err})"
            );
        }
    }

    #[test]
    fn test_bf16_special_values() {
        // Zero
        assert_eq!(bf16_to_f32(f32_to_bf16(0.0)), 0.0);
        // Negative zero
        assert_eq!(bf16_to_f32(f32_to_bf16(-0.0)).to_bits(), (-0.0f32).to_bits());
        // Infinity
        assert!(bf16_to_f32(f32_to_bf16(f32::INFINITY)).is_infinite());
        // NaN
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    }

    #[test]
    fn test_tensor_bf16_serialize() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let bytes = t.to_bf16_bytes();
        assert_eq!(bytes.len(), 8); // 4 elements * 2 bytes

        let loaded = Tensor::from_bf16_bytes(&bytes, &[2, 2]);
        assert_eq!(loaded.shape, vec![2, 2]);
        assert_eq!(loaded.dtype, DType::BF16);
        for i in 0..4 {
            assert!((loaded.data[i] - t.data[i]).abs() < 0.02, "BF16 roundtrip at {i}");
        }
    }

    #[test]
    fn test_f16_roundtrip() {
        let values = vec![0.0f32, 1.0, -1.0, 0.5, 2.0];
        for &v in &values {
            let f16 = f32_to_f16(v);
            let back = f16_to_f32(f16);
            let err = (back - v).abs();
            assert!(err < 0.01, "F16 roundtrip: {v} → {f16} → {back} (error {err})");
        }
    }

    #[test]
    fn test_tiled_matmul_large() {
        // Test that tiled matmul matches naive for larger matrices
        let m = 128;
        let k = 64;
        let n = 96;
        let a = Tensor::randn(&[m, k], 0.1);
        let b = Tensor::randn(&[k, n], 0.1);
        let c = matmul(&a, &b);
        assert_eq!(c.shape, vec![m, n]);

        // Spot-check a few values with naive computation
        for i in [0, m/2, m-1] {
            for j in [0, n/2, n-1] {
                let mut expected = 0.0f32;
                for ki in 0..k {
                    expected += a.data[i * k + ki] * b.data[ki * n + j];
                }
                assert!(
                    (c.data[i * n + j] - expected).abs() < 1e-3,
                    "Tiled matmul mismatch at [{i},{j}]: {} vs {expected}", c.data[i * n + j]
                );
            }
        }
    }

    #[test]
    fn test_add_inplace() {
        let mut a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]);
        a.add_inplace(&b);
        assert_eq!(a.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_memory_bytes() {
        let t = Tensor::zeros(&[1000]);
        assert_eq!(t.memory_bytes(DType::F32), 4000);
        assert_eq!(t.memory_bytes(DType::BF16), 2000);
        assert_eq!(t.memory_bytes(DType::F16), 2000);
    }
}
