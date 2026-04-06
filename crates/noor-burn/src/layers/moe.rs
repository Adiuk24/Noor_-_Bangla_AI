//! Sigmoid MoE Router + per-expert dispatch.
//!
//! Routing: sigmoid(x @ gate + bias) → top-k → weighted expert outputs
//! Simplified dispatch: run all experts, mask by routing weights.
//! This is compute-wasteful but correct and GPU-friendly (no dynamic dispatch).
//! Will optimize to only-active-experts later when profiling shows it matters.

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use super::ffn::SwiGluFfn;

#[derive(Module, Debug)]
pub struct MoeLayer<B: Backend> {
    gate: Linear<B>,
    experts: Vec<SwiGluFfn<B>>,
    shared_expert: Option<SwiGluFfn<B>>,
    n_active: usize,
}

impl<B: Backend> MoeLayer<B> {
    pub fn new(
        d_model: usize,
        expert_ffn_dim: usize,
        n_experts: usize,
        n_active: usize,
        has_shared: bool,
        device: &B::Device,
    ) -> Self {
        let experts = (0..n_experts)
            .map(|_| SwiGluFfn::new(d_model, expert_ffn_dim, device))
            .collect();
        let shared_expert = if has_shared {
            Some(SwiGluFfn::new(d_model, expert_ffn_dim, device))
        } else {
            None
        };

        Self {
            gate: LinearConfig::new(d_model, n_experts).with_bias(true).init(device),
            experts,
            shared_expert,
            n_active,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, d] = x.dims();
        let device = x.device();
        let n_experts = self.experts.len();

        // Sigmoid routing scores: [batch, seq, n_experts]
        let scores = burn::tensor::activation::sigmoid(self.gate.forward(x.clone()));

        // Top-k: get the top n_active scores and zero out the rest
        let (top_values, top_indices) = scores.clone().topk_with_indices(self.n_active, 2);

        // Build sparse weight mask
        let weights = self.build_sparse_weights(scores, top_indices, batch, seq, n_experts, &device);

        // Run all experts and accumulate weighted outputs
        let mut output = Tensor::<B, 3>::zeros([batch, seq, d], &device);
        for (e, expert) in self.experts.iter().enumerate() {
            let expert_weight = weights.clone().slice([0..batch, 0..seq, e..e + 1]); // [b, s, 1]
            let expert_out = expert.forward(x.clone()); // [b, s, d]
            output = output + expert_out * expert_weight;
        }

        // Shared expert
        if let Some(ref shared) = self.shared_expert {
            output = output + shared.forward(x);
        }

        output
    }

    /// Build sparse weight tensor: only top-k expert indices get nonzero weights.
    fn build_sparse_weights(
        &self,
        scores: Tensor<B, 3>,
        top_indices: Tensor<B, 3, Int>,
        batch: usize,
        seq: usize,
        n_experts: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        // Create a mask that is 1.0 for top-k experts, 0.0 for others
        // Then multiply by original scores
        let mut mask = Tensor::<B, 3>::zeros([batch, seq, n_experts], device);
        for k in 0..self.n_active {
            let idx = top_indices.clone().slice([0..batch, 0..seq, k..k + 1]); // [b, s, 1]
            // One-hot scatter: for each (b,s), set mask[b,s,idx[b,s,k]] = 1.0
            // Burn doesn't have scatter, so we use a loop-free approximation:
            // Compare each expert index with the selected index
            for e in 0..n_experts {
                let is_selected = idx.clone().equal_elem(e as i64).float(); // [b, s, 1]
                let current = mask.clone().slice([0..batch, 0..seq, e..e + 1]);
                mask = mask.slice_assign(
                    [0..batch, 0..seq, e..e + 1],
                    current + is_selected,
                );
            }
        }

        // Multiply mask by scores to get sparse weights
        scores * mask
    }
}
