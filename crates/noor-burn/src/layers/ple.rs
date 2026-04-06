//! Per-Layer Embeddings (PLE) — gated per-layer specialization without routing.
//!
//! Each layer gets a unique learned vector. A gate modulates how much PLE
//! contributes to the hidden state:
//!   ple_out = sigmoid(W_gate @ h) * (W_up @ ple_vector[layer_idx])
//!
//! Inspired by Gemma 4 E2B. Used by Noor-Edge (2.8B) instead of MoE.

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, Embedding, EmbeddingConfig};

/// PLE module: stores all per-layer embeddings, projects + gates per layer.
#[derive(Module, Debug)]
pub struct PleLayer<B: Backend> {
    /// Learned embeddings: [n_layers, ple_dim]
    layer_embeddings: Embedding<B>,
    /// Project PLE vector up to d_model: [ple_dim] → [d_model]
    w_up: Linear<B>,
    /// Gate from hidden state: [d_model] → [d_model]
    w_gate: Linear<B>,
    layer_idx: usize,
}

impl<B: Backend> PleLayer<B> {
    pub fn new(
        d_model: usize,
        ple_dim: usize,
        n_layers: usize,
        layer_idx: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            layer_embeddings: EmbeddingConfig::new(n_layers, ple_dim).init(device),
            w_up: LinearConfig::new(ple_dim, d_model).with_bias(false).init(device),
            w_gate: LinearConfig::new(d_model, d_model).with_bias(false).init(device),
            layer_idx,
        }
    }

    /// Apply PLE gating: sigmoid(W_gate @ h) * (W_up @ ple_vector)
    pub fn forward(&self, h: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = h.device();
        let [batch, seq, _d] = h.dims();

        // Get this layer's embedding vector: [1, 1, ple_dim]
        let idx = Tensor::<B, 2, Int>::from_data([[self.layer_idx as i32]], &device);
        let ple_vec = self.layer_embeddings.forward(idx); // [1, 1, ple_dim]

        // Project up to d_model: [1, 1, d_model]
        let ple_proj = self.w_up.forward(ple_vec); // [1, 1, d_model]

        // Broadcast to [batch, seq, d_model]
        let d_model = ple_proj.dims()[2];
        let ple_expanded = ple_proj.expand([batch, seq, d_model]);

        // Gate from hidden state
        let gate = burn::tensor::activation::sigmoid(self.w_gate.forward(h));

        // Gated PLE output
        gate * ple_expanded
    }
}
