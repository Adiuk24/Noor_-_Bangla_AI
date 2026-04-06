use crate::tensor::{self, Tensor};

/// GeGLU FFN: gate(x) * GELU(up(x)), then down-project.
/// Used for the dense branch in parallel dense+MoE layers.
pub struct GeGLUFFN {
    /// Gate projection: (d_model, d_ffn)
    pub w_gate: Tensor,
    /// Up projection: (d_model, d_ffn)
    pub w_up: Tensor,
    /// Down projection: (d_ffn, d_model)
    pub w_down: Tensor,
}

impl GeGLUFFN {
    pub fn new(d_model: usize, d_ffn: usize, init_std: f64) -> Self {
        Self {
            w_gate: Tensor::randn(&[d_model, d_ffn], init_std),
            w_up: Tensor::randn(&[d_model, d_ffn], init_std),
            w_down: Tensor::randn(&[d_ffn, d_model], init_std),
        }
    }

    /// Forward pass. x: (seq_len, d_model) -> (seq_len, d_model)
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // gate = x @ w_gate -> (seq_len, d_ffn)
        let gate = tensor::matmul(x, &self.w_gate);
        // up = x @ w_up -> (seq_len, d_ffn)
        let up = tensor::matmul(x, &self.w_up);
        // gated = GELU(gate) * up
        let gate_act = tensor::gelu(&gate);
        let gated = tensor::mul(&gate_act, &up);
        // down = gated @ w_down -> (seq_len, d_model)
        tensor::matmul(&gated, &self.w_down)
    }

    pub fn param_count(&self) -> usize {
        self.w_gate.numel() + self.w_up.numel() + self.w_down.numel()
    }
}

/// SwiGLU FFN: SiLU(gate(x)) * up(x), then down-project.
/// Used for MoE experts.
pub struct SwiGLUFFN {
    /// Gate projection: (d_model, d_ffn)
    pub w_gate: Tensor,
    /// Up projection: (d_model, d_ffn)
    pub w_up: Tensor,
    /// Down projection: (d_ffn, d_model)
    pub w_down: Tensor,
}

impl SwiGLUFFN {
    pub fn new(d_model: usize, d_ffn: usize, init_std: f64) -> Self {
        Self {
            w_gate: Tensor::randn(&[d_model, d_ffn], init_std),
            w_up: Tensor::randn(&[d_model, d_ffn], init_std),
            w_down: Tensor::randn(&[d_ffn, d_model], init_std),
        }
    }

    /// Forward pass. x: (seq_len, d_model) -> (seq_len, d_model)
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let gate = tensor::matmul(x, &self.w_gate);
        let up = tensor::matmul(x, &self.w_up);
        let gate_act = tensor::silu(&gate);
        let gated = tensor::mul(&gate_act, &up);
        tensor::matmul(&gated, &self.w_down)
    }

    pub fn param_count(&self) -> usize {
        self.w_gate.numel() + self.w_up.numel() + self.w_down.numel()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geglu_shape() {
        let ffn = GeGLUFFN::new(64, 128, 0.02);
        let x = Tensor::randn(&[4, 64], 1.0); // (seq_len=4, d_model=64)
        let out = ffn.forward(&x);
        assert_eq!(out.shape, vec![4, 64], "GeGLU output should match input shape");
    }

    #[test]
    fn test_swiglu_shape() {
        let ffn = SwiGLUFFN::new(64, 128, 0.02);
        let x = Tensor::randn(&[4, 64], 1.0);
        let out = ffn.forward(&x);
        assert_eq!(out.shape, vec![4, 64], "SwiGLU output should match input shape");
    }

    #[test]
    fn test_geglu_param_count() {
        let ffn = GeGLUFFN::new(2048, 1536, 0.02);
        // 3 matrices: gate + up + down = 2 * (2048*1536) + (1536*2048)
        let expected = 3 * 2048 * 1536;
        assert_eq!(ffn.param_count(), expected);
    }

    #[test]
    fn test_swiglu_param_count() {
        let ffn = SwiGLUFFN::new(2048, 512, 0.02);
        let expected = 3 * 2048 * 512;
        assert_eq!(ffn.param_count(), expected);
    }

    #[test]
    fn test_ffn_output_not_zero() {
        // With random weights and input, output should not be all zeros
        let ffn = GeGLUFFN::new(32, 64, 0.5);
        let x = Tensor::randn(&[2, 32], 1.0);
        let out = ffn.forward(&x);
        let sum: f32 = out.data.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "FFN output should not be all zeros");
    }
}
