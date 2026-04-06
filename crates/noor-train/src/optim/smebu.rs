//! SMEBU: Soft-clamped Momentum Expert Bias Updates.
//! Maintains per-expert bias terms with momentum to balance MoE load.
//! From Trinity (Arcee AI).

use noor_core::tensor::Tensor;

/// SMEBU optimizer for MoE routing biases.
pub struct SMEBU {
    /// Per-expert momentum: (n_experts,)
    pub momentum: Vec<f32>,
    /// Current bias values: (n_experts,)
    pub biases: Vec<f32>,
    /// Clamp range
    pub kappa: f32,
    /// Momentum coefficient
    pub beta: f32,
    /// Update rate
    pub lambda: f32,
    pub n_experts: usize,
}

impl SMEBU {
    pub fn new(n_experts: usize, kappa: f32, beta: f32, lambda: f32) -> Self {
        Self {
            momentum: vec![0.0; n_experts],
            biases: vec![0.0; n_experts],
            kappa,
            beta,
            lambda,
            n_experts,
        }
    }

    /// Update biases based on expert utilization fractions.
    /// fractions: per-expert fraction of tokens routed (should sum to n_active).
    pub fn update(&mut self, fractions: &[f32]) {
        assert_eq!(fractions.len(), self.n_experts);
        let f_target = 1.0 / self.n_experts as f32;

        for i in 0..self.n_experts {
            // Momentum update
            self.momentum[i] = self.beta * self.momentum[i]
                + self.lambda * (f_target - fractions[i]);

            // Soft clamp with tanh
            self.biases[i] = self.kappa * (self.momentum[i] / self.kappa).tanh();
        }
    }

    /// Get current biases as a tensor.
    pub fn bias_tensor(&self) -> Tensor {
        Tensor::from_slice(&self.biases, &[self.n_experts])
    }

    /// Apply biases to the router's expert_biases tensor.
    pub fn apply_to_router(&self, router_biases: &mut Tensor) {
        assert_eq!(router_biases.numel(), self.n_experts);
        for i in 0..self.n_experts {
            router_biases.data[i] = self.biases[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smebu_balanced_stays_zero() {
        let mut smebu = SMEBU::new(8, 5.0, 0.9, 0.01);
        // Perfectly balanced: each expert gets 1/8 = 0.125
        let fracs = vec![0.125; 8];
        for _ in 0..100 {
            smebu.update(&fracs);
        }
        // Biases should remain near zero
        for &b in &smebu.biases {
            assert!(b.abs() < 0.01, "Balanced utilization should keep bias near 0, got {b}");
        }
    }

    #[test]
    fn test_smebu_imbalanced_corrects() {
        let mut smebu = SMEBU::new(4, 5.0, 0.9, 0.05); // larger lambda for faster convergence in test
        // Expert 0 gets all traffic, rest get nothing
        let fracs = vec![1.0, 0.0, 0.0, 0.0];
        for _ in 0..500 {
            smebu.update(&fracs);
        }
        // Expert 0's bias should be negative (discourage), others positive (encourage)
        assert!(smebu.biases[0] < -0.1, "Overused expert bias should be negative: {}", smebu.biases[0]);
        assert!(smebu.biases[1] > 0.1, "Underused expert bias should be positive: {}", smebu.biases[1]);
        assert!(smebu.biases[2] > 0.1, "Underused expert bias should be positive: {}", smebu.biases[2]);
    }

    #[test]
    fn test_smebu_clamped() {
        let mut smebu = SMEBU::new(2, 5.0, 0.9, 0.1);
        // Extreme imbalance for many steps
        let fracs = vec![1.0, 0.0];
        for _ in 0..10000 {
            smebu.update(&fracs);
        }
        // Biases must be within [-kappa, kappa] due to tanh clamping
        for &b in &smebu.biases {
            assert!(
                b.abs() <= 5.0 + 0.01,
                "Bias {b} exceeds kappa=5.0"
            );
        }
    }

    #[test]
    fn test_apply_to_router() {
        let mut smebu = SMEBU::new(4, 5.0, 0.9, 0.01);
        smebu.biases = vec![0.1, -0.2, 0.3, -0.4];
        let mut router_biases = Tensor::zeros(&[4]);
        smebu.apply_to_router(&mut router_biases);
        assert_eq!(router_biases.data, vec![0.1, -0.2, 0.3, -0.4]);
    }
}
