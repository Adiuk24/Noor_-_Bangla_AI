use crate::tensor::{self, Tensor};
use crate::layers::ffn::GeGLUFFN;
use crate::layers::moe::MoELayer;

/// Parallel Dense + MoE combiner (Gemma 4 pattern).
/// Runs both branches in parallel, sums outputs / sqrt(2).
pub struct ParallelFFN {
    pub dense: GeGLUFFN,
    pub moe: MoELayer,
}

impl ParallelFFN {
    pub fn new(
        d_model: usize,
        dense_ffn_dim: usize,
        expert_ffn_dim: usize,
        n_experts: usize,
        n_active: usize,
        has_shared: bool,
        init_std: f64,
    ) -> Self {
        Self {
            dense: GeGLUFFN::new(d_model, dense_ffn_dim, init_std),
            moe: MoELayer::new(d_model, expert_ffn_dim, n_experts, n_active, has_shared, init_std),
        }
    }

    /// Forward pass. x: (seq_len, d_model) -> (seq_len, d_model)
    /// Returns: (dense_out + moe_out) / sqrt(2)
    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        let dense_out = self.dense.forward(x);
        let moe_out = self.moe.forward(x);
        let combined = tensor::add(&dense_out, &moe_out);
        let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
        combined.scale(inv_sqrt2)
    }

    pub fn param_count(&self) -> usize {
        self.dense.param_count() + self.moe.param_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_ffn_shape() {
        let mut pffn = ParallelFFN::new(32, 64, 32, 8, 2, true, 0.02);
        let x = Tensor::randn(&[4, 32], 1.0);
        let out = pffn.forward(&x);
        assert_eq!(out.shape, vec![4, 32], "Output should match input shape");
    }

    #[test]
    fn test_parallel_ffn_scaling() {
        // Verify output variance is reasonable (sqrt(2) scaling prevents blowup)
        let mut pffn = ParallelFFN::new(64, 128, 64, 8, 2, true, 0.02);
        let x = Tensor::randn(&[16, 64], 0.5);
        let out = pffn.forward(&x);

        let in_var: f32 = x.data.iter().map(|v| v * v).sum::<f32>() / x.numel() as f32;
        let out_var: f32 = out.data.iter().map(|v| v * v).sum::<f32>() / out.numel() as f32;

        // Output variance should be in a reasonable range (not exploding)
        assert!(
            out_var < in_var * 100.0,
            "Output variance {out_var} too large vs input {in_var}"
        );
    }

    #[test]
    fn test_parallel_ffn_param_count() {
        let pffn = ParallelFFN::new(32, 64, 32, 4, 2, true, 0.02);
        let dense_params = 3 * 32 * 64; // GeGLU
        let expert_params = 4 * 3 * 32 * 32; // 4 experts
        let shared_params = 3 * 32 * 32; // shared expert
        let router_params = 32 * 4 + 4 + 4; // gate + biases + scales
        let expected = dense_params + expert_params + shared_params + router_params;
        assert_eq!(pffn.param_count(), expected);
    }
}
