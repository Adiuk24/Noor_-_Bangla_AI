//! Noor model — full MoE transformer assembled from Burn modules.
//!
//! Supports: proxy (288M), edge (2.8B PLE), pro (12B MoE), max (28B MoE).
//! Architecture matches docs/2026-04-06-noor-architecture-design.md exactly.

use burn::prelude::*;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::nn::loss::CrossEntropyLossConfig;

use crate::config::NoorConfig;
use crate::layers::block::MoeBlock;

#[derive(Module, Debug)]
pub struct NoorModel<B: Backend> {
    embedding: Embedding<B>,
    blocks: Vec<MoeBlock<B>>,
    final_norm: RmsNorm<B>,
    output_proj: Linear<B>,
    vocab_size: usize,
    context_length: usize,
}

impl<B: Backend> NoorModel<B> {
    pub fn from_config(config: &NoorConfig, device: &B::Device) -> Self {
        let d = config.model.d_model;
        let vocab = config.model.vocab_size;

        let blocks = (0..config.model.n_layers)
            .map(|i| {
                let is_global = (i + 1) % config.attention.global_every_n == 0;
                let theta = if is_global { config.rope.prope_theta } else { config.rope.theta };

                MoeBlock::new(
                    d,
                    config.model.n_heads,
                    config.model.n_kv_heads,
                    config.model.head_dim,
                    config.moe.dense_ffn_dim,
                    config.moe.expert_ffn_dim,
                    config.moe.n_experts,
                    config.moe.n_active_experts,
                    config.moe.has_shared_expert,
                    config.attention.sliding_window,
                    is_global,
                    theta,
                    config.model.context_length,
                    config.norm.eps,
                    i,
                    device,
                )
            })
            .collect();

        Self {
            embedding: EmbeddingConfig::new(vocab, d).init(device),
            blocks,
            final_norm: RmsNormConfig::new(d)
                .with_epsilon(config.norm.eps)
                .init(device),
            output_proj: LinearConfig::new(d, vocab).with_bias(false).init(device),
            vocab_size: vocab,
            context_length: config.model.context_length,
        }
    }

    /// Forward pass: tokens → logits.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut x = self.embedding.forward(tokens);

        for block in &self.blocks {
            x = block.forward(x);
        }

        x = self.final_norm.forward(x);
        self.output_proj.forward(x)
    }

    /// Forward + cross-entropy loss for training.
    pub fn forward_loss(
        &self,
        input_ids: Tensor<B, 2, Int>,
        target_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        let logits = self.forward(input_ids); // [batch, seq, vocab]
        let [batch, seq, vocab] = logits.dims();

        let logits_flat = logits.reshape([batch * seq, vocab]);
        let targets_flat = target_ids.reshape([batch * seq]);

        CrossEntropyLossConfig::new()
            .init(&logits_flat.device())
            .forward(logits_flat, targets_flat)
    }

    pub fn param_count(&self) -> usize {
        self.num_params()
    }
}
