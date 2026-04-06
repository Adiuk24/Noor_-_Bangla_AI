use crate::tensor::Tensor;
use crate::layers::ffn::SwiGLUFFN;

/// Sigmoid MoE Router with SMEBU bias support and per-expert scales.
pub struct MoERouter {
    /// Gate weights: (d_model, n_experts)
    pub gate: Tensor,
    /// Per-expert biases maintained by SMEBU: (n_experts,)
    pub expert_biases: Tensor,
    /// Per-expert learned output scales: (n_experts,)
    pub expert_scales: Tensor,
    pub n_experts: usize,
    pub n_active: usize,
}

/// Routing result for a single token.
pub struct RouteResult {
    /// Indices of selected experts
    pub indices: Vec<usize>,
    /// Weights for selected experts (sigmoid score * expert_scale)
    pub weights: Vec<f32>,
}

/// Per-layer expert utilization counters.
pub struct ExpertUtilization {
    /// How many tokens each expert processed
    pub counts: Vec<usize>,
    /// Total tokens routed
    pub total_tokens: usize,
}

impl ExpertUtilization {
    pub fn new(n_experts: usize) -> Self {
        Self {
            counts: vec![0; n_experts],
            total_tokens: 0,
        }
    }

    pub fn record(&mut self, indices: &[usize]) {
        self.total_tokens += 1;
        for &idx in indices {
            self.counts[idx] += 1;
        }
    }

    /// Fraction of tokens each expert got.
    pub fn fractions(&self) -> Vec<f32> {
        if self.total_tokens == 0 {
            return vec![0.0; self.counts.len()];
        }
        self.counts.iter().map(|&c| c as f32 / self.total_tokens as f32).collect()
    }

    /// Number of experts with > 1% utilization.
    pub fn active_expert_count(&self) -> usize {
        let fracs = self.fractions();
        fracs.iter().filter(|&&f| f > 0.01).count()
    }

    pub fn reset(&mut self) {
        self.counts.fill(0);
        self.total_tokens = 0;
    }
}

impl MoERouter {
    pub fn new(d_model: usize, n_experts: usize, n_active: usize, init_std: f64) -> Self {
        Self {
            gate: Tensor::randn(&[d_model, n_experts], init_std),
            expert_biases: Tensor::zeros(&[n_experts]),
            expert_scales: Tensor::ones(&[n_experts]),
            n_experts,
            n_active,
        }
    }

    /// Route a single token. x: (d_model,) as a slice of the full tensor.
    pub fn route(&self, x: &[f32]) -> RouteResult {
        let d = self.gate.shape[0];
        let ne = self.n_experts;
        assert_eq!(x.len(), d);

        // Compute sigmoid scores: sigmoid(x @ gate + bias)
        let mut scores = vec![0.0f32; ne];
        for e in 0..ne {
            let mut dot = 0.0f32;
            for i in 0..d {
                dot += x[i] * self.gate.data[i * ne + e];
            }
            dot += self.expert_biases.data[e];
            scores[e] = 1.0 / (1.0 + (-dot).exp()); // sigmoid
        }

        // Top-k selection
        let mut indexed: Vec<(usize, f32)> = scores.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let k = self.n_active.min(ne);
        let indices: Vec<usize> = indexed[..k].iter().map(|(i, _)| *i).collect();
        let weights: Vec<f32> = indexed[..k]
            .iter()
            .map(|(i, s)| s * self.expert_scales.data[*i])
            .collect();

        RouteResult { indices, weights }
    }
}

/// Full MoE layer: router + experts + shared expert.
pub struct MoELayer {
    pub router: MoERouter,
    pub experts: Vec<SwiGLUFFN>,
    pub shared_expert: Option<SwiGLUFFN>,
    pub utilization: ExpertUtilization,
}

impl MoELayer {
    pub fn new(
        d_model: usize,
        expert_ffn_dim: usize,
        n_experts: usize,
        n_active: usize,
        has_shared: bool,
        init_std: f64,
    ) -> Self {
        let experts: Vec<SwiGLUFFN> = (0..n_experts)
            .map(|_| SwiGLUFFN::new(d_model, expert_ffn_dim, init_std))
            .collect();
        let shared = if has_shared {
            Some(SwiGLUFFN::new(d_model, expert_ffn_dim, init_std))
        } else {
            None
        };

        Self {
            router: MoERouter::new(d_model, n_experts, n_active, init_std * 0.5),
            experts,
            shared_expert: shared,
            utilization: ExpertUtilization::new(n_experts),
        }
    }

    /// Forward pass. x: (seq_len, d_model) -> (seq_len, d_model)
    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        let seq_len = x.shape[0];
        let d = x.shape[1];
        let mut output_data = vec![0.0f32; seq_len * d];

        for s in 0..seq_len {
            let token_slice = &x.data[s * d..(s + 1) * d];
            let route = self.router.route(token_slice);
            self.utilization.record(&route.indices);

            // Create single-token tensor for expert forward
            let token = Tensor::from_slice(token_slice, &[1, d]);

            // Sum weighted expert outputs
            for (i, &expert_idx) in route.indices.iter().enumerate() {
                let expert_out = self.experts[expert_idx].forward(&token);
                let weight = route.weights[i];
                for j in 0..d {
                    output_data[s * d + j] += weight * expert_out.data[j];
                }
            }

            // Add shared expert (always active, unweighted)
            if let Some(ref shared) = self.shared_expert {
                let shared_out = shared.forward(&token);
                for j in 0..d {
                    output_data[s * d + j] += shared_out.data[j];
                }
            }
        }

        Tensor::from_slice(&output_data, &[seq_len, d])
    }

    pub fn param_count(&self) -> usize {
        let expert_params: usize = self.experts.iter().map(|e| e.param_count()).sum();
        let shared_params = self.shared_expert.as_ref().map_or(0, |e| e.param_count());
        let router_params = self.router.gate.numel()
            + self.router.expert_biases.numel()
            + self.router.expert_scales.numel();
        expert_params + shared_params + router_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_returns_k_indices() {
        let router = MoERouter::new(32, 8, 2, 0.5);
        let x = vec![1.0f32; 32];
        let result = router.route(&x);
        assert_eq!(result.indices.len(), 2, "Should return exactly k=2 experts");
        assert_eq!(result.weights.len(), 2);
    }

    #[test]
    fn test_router_indices_in_range() {
        let router = MoERouter::new(32, 16, 4, 0.5);
        let x = Tensor::randn(&[32], 1.0);
        let result = router.route(&x.data);
        for &idx in &result.indices {
            assert!(idx < 16, "Expert index {idx} out of range");
        }
    }

    #[test]
    fn test_router_weights_positive() {
        let router = MoERouter::new(32, 8, 3, 0.5);
        let x = Tensor::randn(&[32], 1.0);
        let result = router.route(&x.data);
        for &w in &result.weights {
            assert!(w > 0.0, "Sigmoid weights must be positive, got {w}");
        }
    }

    #[test]
    fn test_moe_layer_shape() {
        let mut moe = MoELayer::new(32, 64, 8, 2, true, 0.02);
        let x = Tensor::randn(&[4, 32], 1.0); // seq=4, d=32
        let out = moe.forward(&x);
        assert_eq!(out.shape, vec![4, 32], "MoE output shape should match input");
    }

    #[test]
    fn test_expert_utilization_tracking() {
        let mut moe = MoELayer::new(32, 64, 8, 2, false, 0.5);
        let x = Tensor::randn(&[100, 32], 1.0); // 100 tokens
        moe.forward(&x);

        assert_eq!(moe.utilization.total_tokens, 100);
        let total_count: usize = moe.utilization.counts.iter().sum();
        assert_eq!(total_count, 200, "2 active experts * 100 tokens = 200 assignments");

        // With random init, utilization should be roughly balanced
        let active = moe.utilization.active_expert_count();
        assert!(
            active >= 4,
            "At least half of 8 experts should be active, got {active}"
        );
    }

    #[test]
    fn test_utilization_reset() {
        let mut util = ExpertUtilization::new(8);
        util.record(&[0, 1]);
        util.record(&[2, 3]);
        assert_eq!(util.total_tokens, 2);
        util.reset();
        assert_eq!(util.total_tokens, 0);
        assert!(util.counts.iter().all(|&c| c == 0));
    }
}
