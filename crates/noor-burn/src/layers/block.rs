//! Transformer block — MoE variant (proxy/pro/max) and PLE variant (edge).

use burn::prelude::*;
use super::attention::GqaAttention;
use super::ffn::GeGluFfn;
use super::moe::MoeLayer;
use super::norm::SandwichNorm;

/// MoE block: sandwich_norm → attention → sandwich_norm → parallel(dense + MoE) / sqrt(2)
#[derive(Module, Debug)]
pub struct MoeBlock<B: Backend> {
    norm_attn: SandwichNorm<B>,
    attention: GqaAttention<B>,
    norm_ffn: SandwichNorm<B>,
    dense_ffn: GeGluFfn<B>,
    moe: MoeLayer<B>,
}

impl<B: Backend> MoeBlock<B> {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        dense_ffn_dim: usize,
        expert_ffn_dim: usize,
        n_experts: usize,
        n_active: usize,
        has_shared_expert: bool,
        sliding_window: usize,
        is_global: bool,
        rope_theta: f64,
        max_seq_len: usize,
        norm_eps: f64,
        layer_idx: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            norm_attn: SandwichNorm::new(d_model, norm_eps, layer_idx, device),
            attention: GqaAttention::new(
                d_model, n_heads, n_kv_heads, head_dim,
                sliding_window, is_global, rope_theta, max_seq_len, device,
            ),
            norm_ffn: SandwichNorm::new(d_model, norm_eps, layer_idx, device),
            dense_ffn: GeGluFfn::new(d_model, dense_ffn_dim, device),
            moe: MoeLayer::new(d_model, expert_ffn_dim, n_experts, n_active, has_shared_expert, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Attention sublayer with sandwich norm + residual
        let normed = self.norm_attn.pre(x.clone());
        let attn_out = self.attention.forward(normed);
        let attn_out = self.norm_attn.post(attn_out);
        let h = x + attn_out;

        // Parallel dense + MoE sublayer with sandwich norm + residual
        let normed = self.norm_ffn.pre(h.clone());
        let dense_out = self.dense_ffn.forward(normed.clone());
        let moe_out = self.moe.forward(normed);
        let ffn_out = (dense_out + moe_out) / (2.0f32).sqrt();
        let ffn_out = self.norm_ffn.post(ffn_out);

        h + ffn_out
    }
}
