//! Transformer block — MoE variant (proxy/pro/max) and PLE variant (edge).

use burn::prelude::*;
use super::attention::GqaAttention;
use super::ffn::GeGluFfn;
use super::moe::MoeLayer;
use super::norm::{SandwichNorm, PreNorm};
use super::ple::PleLayer;

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

/// Dense block with PLE: pre_norm → attention → pre_norm → GeGLU FFN → PLE gate → residual
/// Used by Noor-Edge (2.8B) — no MoE, no sandwich norm.
#[derive(Module, Debug)]
pub struct DenseBlock<B: Backend> {
    norm_attn: PreNorm<B>,
    attention: GqaAttention<B>,
    norm_ffn: PreNorm<B>,
    dense_ffn: GeGluFfn<B>,
    ple: PleLayer<B>,
}

impl<B: Backend> DenseBlock<B> {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        dense_ffn_dim: usize,
        ple_dim: usize,
        n_layers: usize,
        sliding_window: usize,
        is_global: bool,
        rope_theta: f64,
        max_seq_len: usize,
        norm_eps: f64,
        layer_idx: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            norm_attn: PreNorm::new(d_model, norm_eps, device),
            attention: GqaAttention::new(
                d_model, n_heads, n_kv_heads, head_dim,
                sliding_window, is_global, rope_theta, max_seq_len, device,
            ),
            norm_ffn: PreNorm::new(d_model, norm_eps, device),
            dense_ffn: GeGluFfn::new(d_model, dense_ffn_dim, device),
            ple: PleLayer::new(d_model, ple_dim, n_layers, layer_idx, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Attention sublayer with pre-norm + residual
        let normed = self.norm_attn.forward(x.clone());
        let attn_out = self.attention.forward(normed);
        let h = x + attn_out;

        // FFN sublayer with pre-norm + PLE + residual
        let normed = self.norm_ffn.forward(h.clone());
        let ffn_out = self.dense_ffn.forward(normed);
        let ple_out = self.ple.forward(ffn_out.clone());
        h + ffn_out + ple_out
    }
}
