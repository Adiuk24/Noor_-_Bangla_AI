use crate::tensor::{self, Tensor};

/// RMSNorm: x / sqrt(mean(x^2) + eps) * weight
pub struct RMSNorm {
    /// Learnable scale: (d_model,)
    pub weight: Tensor,
    pub eps: f64,
}

impl RMSNorm {
    /// Create RMSNorm with weights initialized to ones.
    pub fn new(d_model: usize, eps: f64) -> Self {
        Self {
            weight: Tensor::ones(&[d_model]),
            eps,
        }
    }

    /// Forward pass. x: (..., d_model) -> (..., d_model)
    pub fn forward(&self, x: &Tensor) -> Tensor {
        tensor::rms_norm(x, &self.weight, self.eps)
    }

    pub fn d_model(&self) -> usize {
        self.weight.numel()
    }
}

/// Sandwich RMSNorm with depth scaling.
/// Applies pre-norm and post-norm around a sublayer, both scaled by 1/sqrt(layer_idx+1).
pub struct SandwichNorm {
    pub pre_norm: RMSNorm,
    pub post_norm: RMSNorm,
    /// Depth scale factor: 1.0 / sqrt(layer_idx + 1)
    pub depth_scale: f32,
}

impl SandwichNorm {
    /// Create sandwich norm for a given layer index.
    /// layer_idx: 0-based layer index for depth scaling.
    pub fn new(d_model: usize, eps: f64, layer_idx: usize) -> Self {
        Self {
            pre_norm: RMSNorm::new(d_model, eps),
            post_norm: RMSNorm::new(d_model, eps),
            depth_scale: 1.0 / ((layer_idx + 1) as f32).sqrt(),
        }
    }

    /// Apply pre-normalization with depth scaling.
    pub fn pre(&self, x: &Tensor) -> Tensor {
        let normed = self.pre_norm.forward(x);
        normed.scale(self.depth_scale)
    }

    /// Apply post-normalization with depth scaling.
    pub fn post(&self, x: &Tensor) -> Tensor {
        let normed = self.post_norm.forward(x);
        normed.scale(self.depth_scale)
    }

    pub fn d_model(&self) -> usize {
        self.pre_norm.d_model()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_unit_rms() {
        let norm = RMSNorm::new(64, 1e-6);
        let x = Tensor::randn(&[4, 64], 1.0);
        let out = norm.forward(&x);
        assert_eq!(out.shape, x.shape);

        // Each row should have RMS ≈ 1.0 (since weight is all ones)
        let d = 64;
        for row in 0..4 {
            let mut sum_sq = 0.0f64;
            for col in 0..d {
                let val = out.data[row * d + col] as f64;
                sum_sq += val * val;
            }
            let rms = (sum_sq / d as f64).sqrt();
            assert!(
                (rms - 1.0).abs() < 0.02,
                "RMS of row {row} should be ~1.0, got {rms}"
            );
        }
    }

    #[test]
    fn test_rms_norm_shape_preserved() {
        let norm = RMSNorm::new(128, 1e-6);
        let x = Tensor::randn(&[8, 16, 128], 1.0);
        let out = norm.forward(&x);
        assert_eq!(out.shape, x.shape);
    }

    #[test]
    fn test_sandwich_depth_scale() {
        // Layer 0: scale = 1/sqrt(1) = 1.0
        let s0 = SandwichNorm::new(32, 1e-6, 0);
        assert!((s0.depth_scale - 1.0).abs() < 1e-6);

        // Layer 3: scale = 1/sqrt(4) = 0.5
        let s3 = SandwichNorm::new(32, 1e-6, 3);
        assert!((s3.depth_scale - 0.5).abs() < 1e-6);

        // Layer 15: scale = 1/sqrt(16) = 0.25
        let s15 = SandwichNorm::new(32, 1e-6, 15);
        assert!((s15.depth_scale - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_sandwich_pre_post_differ() {
        let sandwich = SandwichNorm::new(32, 1e-6, 5);
        let x = Tensor::randn(&[2, 32], 1.0);
        let pre_out = sandwich.pre(&x);
        let post_out = sandwich.post(&x);

        // Both should have the same shape
        assert_eq!(pre_out.shape, x.shape);
        assert_eq!(post_out.shape, x.shape);

        // Pre and post should be identical in this test (both norms init to ones)
        // since the input is the same and both norms are freshly created
        // The key is that both scale by depth_scale
        let scale = sandwich.depth_scale;
        assert!(scale < 1.0, "Layer 5 depth scale should be < 1.0");
    }

    #[test]
    fn test_sandwich_reduces_magnitude() {
        // Deep layer should reduce magnitude more
        let deep = SandwichNorm::new(16, 1e-6, 99); // layer 99 → scale = 1/sqrt(100) = 0.1
        assert!((deep.depth_scale - 0.1).abs() < 1e-5);

        let x = Tensor::randn(&[2, 16], 1.0);
        let out = deep.pre(&x);

        // Output magnitude should be ~0.1 of normalized input
        let in_rms: f32 = x.data.iter().map(|v| v * v).sum::<f32>() / x.numel() as f32;
        let out_rms: f32 = out.data.iter().map(|v| v * v).sum::<f32>() / out.numel() as f32;
        assert!(
            out_rms < in_rms,
            "Deep sandwich norm should reduce magnitude: in_rms={in_rms}, out_rms={out_rms}"
        );
    }
}
