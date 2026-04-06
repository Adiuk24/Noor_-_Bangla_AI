use crate::tensor::Tensor;

/// Rotary Position Embedding (RoPE).
/// Supports both standard RoPE and proportional RoPE (p-RoPE).
pub struct RoPE {
    /// Inverse frequencies: (head_dim / 2,)
    pub inv_freq: Vec<f32>,
    /// Number of dimensions to rotate (for p-RoPE, this may be < head_dim)
    pub rotate_dim: usize,
    /// Full head dimension
    pub head_dim: usize,
}

impl RoPE {
    /// Create standard RoPE: rotates all head dimensions.
    /// theta: base frequency (default 10000.0)
    pub fn new(head_dim: usize, theta: f64) -> Self {
        let rotate_dim = head_dim;
        let half = rotate_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / rotate_dim as f64) as f32)
            .collect();
        Self {
            inv_freq,
            rotate_dim,
            head_dim,
        }
    }

    /// Create proportional RoPE (p-RoPE): rotates only a fraction of dims.
    /// fraction: e.g. 0.25 means rotate 25% of head dims
    /// theta: base frequency (typically 1_000_000.0 for p-RoPE)
    pub fn new_proportional(head_dim: usize, theta: f64, fraction: f64) -> Self {
        let rotate_dim = ((head_dim as f64 * fraction) as usize / 2) * 2; // round to even
        let rotate_dim = rotate_dim.max(2); // at least 2 dims
        let half = rotate_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / rotate_dim as f64) as f32)
            .collect();
        Self {
            inv_freq,
            rotate_dim,
            head_dim,
        }
    }

    /// Apply RoPE to a tensor of shape (..., seq_len, head_dim).
    /// seq_offset: starting position (for KV cache continuation).
    pub fn apply(&self, x: &Tensor, seq_offset: usize) -> Tensor {
        let ndim = x.ndim();
        assert!(ndim >= 2);
        let seq_len = x.shape[ndim - 2];
        let hd = x.shape[ndim - 1];
        assert_eq!(hd, self.head_dim, "Last dim {} != head_dim {}", hd, self.head_dim);

        let batch_size: usize = x.shape[..ndim - 2].iter().product();
        let half = self.rotate_dim / 2;
        let mut result = x.data.clone();

        for b in 0..batch_size {
            for s in 0..seq_len {
                let pos = (seq_offset + s) as f32;
                let base = b * seq_len * hd + s * hd;

                // Rotate the first `rotate_dim` dimensions in pairs
                for i in 0..half {
                    let angle = pos * self.inv_freq[i];
                    let cos_val = angle.cos();
                    let sin_val = angle.sin();

                    let idx0 = base + 2 * i;
                    let idx1 = base + 2 * i + 1;

                    let x0 = x.data[idx0];
                    let x1 = x.data[idx1];

                    result[idx0] = x0 * cos_val - x1 * sin_val;
                    result[idx1] = x0 * sin_val + x1 * cos_val;
                }
                // Dimensions beyond rotate_dim are left unchanged (already copied)
            }
        }

        Tensor {
            data: result,
            shape: x.shape.clone(),
            strides: x.strides.clone(),
            dtype: x.dtype,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_preserves_magnitude() {
        let rope = RoPE::new(64, 10000.0);
        let x = Tensor::randn(&[2, 8, 64], 1.0); // (batch=2, seq=8, head_dim=64)

        let rotated = rope.apply(&x, 0);
        assert_eq!(rotated.shape, x.shape);

        // Magnitude of each pair should be preserved
        for i in (0..x.numel()).step_by(2) {
            let orig_mag = (x.data[i].powi(2) + x.data[i + 1].powi(2)).sqrt();
            let rot_mag = (rotated.data[i].powi(2) + rotated.data[i + 1].powi(2)).sqrt();
            assert!(
                (orig_mag - rot_mag).abs() < 1e-4,
                "Magnitude not preserved at {i}: {orig_mag} vs {rot_mag}"
            );
        }
    }

    #[test]
    fn test_rope_different_offsets_differ() {
        let rope = RoPE::new(64, 10000.0);
        let x = Tensor::ones(&[1, 1, 64]); // single position

        let r0 = rope.apply(&x, 0);
        let r1 = rope.apply(&x, 1);
        let r100 = rope.apply(&x, 100);

        // Different offsets should produce different results
        assert!(r0.data != r1.data, "Offset 0 and 1 should differ");
        assert!(r1.data != r100.data, "Offset 1 and 100 should differ");
    }

    #[test]
    fn test_rope_offset_zero_identity_like() {
        // At position 0, angle = 0 for all freqs, so cos=1, sin=0 → output ≈ input
        let rope = RoPE::new(4, 10000.0);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let r = rope.apply(&x, 0);
        // At pos=0: angle=0 for all, cos=1, sin=0 → no rotation
        for i in 0..4 {
            assert!(
                (r.data[i] - x.data[i]).abs() < 1e-5,
                "At pos 0, RoPE should be identity: {} vs {}",
                r.data[i], x.data[i]
            );
        }
    }

    #[test]
    fn test_prope_partial_rotation() {
        // p-RoPE with 25% rotation on head_dim=8: only dims 0,1 are rotated
        let prope = RoPE::new_proportional(8, 1_000_000.0, 0.25);
        assert_eq!(prope.rotate_dim, 2);

        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 1, 8]);
        let r = prope.apply(&x, 5); // non-zero position to ensure rotation

        // Dims 2-7 should be unchanged (not rotated)
        for i in 2..8 {
            assert_eq!(
                r.data[i], x.data[i],
                "Dim {i} should not be rotated by p-RoPE"
            );
        }
        // Dims 0-1 should be different (rotated)
        let changed = (r.data[0] - x.data[0]).abs() > 1e-6
            || (r.data[1] - x.data[1]).abs() > 1e-6;
        assert!(changed, "Dims 0-1 should be rotated by p-RoPE");
    }
}
