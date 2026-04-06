use crate::tensor::{self, Tensor};
use crate::layers::attention::{GQAAttention, KVCache};
use crate::layers::ffn::GeGLUFFN;
use crate::layers::norm::{RMSNorm, SandwichNorm};
use crate::layers::parallel_ffn::ParallelFFN;
use crate::layers::rope::RoPE;

/// MoE transformer block (Noor-Pro / Noor-Max).
/// Sandwich norm + GQA attention + parallel dense+MoE + residual.
pub struct MoEBlock {
    pub attn_norm: SandwichNorm,
    pub attention: GQAAttention,
    pub ffn_norm: SandwichNorm,
    pub parallel_ffn: ParallelFFN,
    pub layer_idx: usize,
    pub is_global: bool,
}

/// PLE transformer block (Noor-Edge).
/// Pre-norm + GQA attention + GeGLU FFN + PLE modulation + residual.
pub struct PLEBlock {
    pub attn_norm: RMSNorm,
    pub attention: GQAAttention,
    pub ffn_norm: RMSNorm,
    pub ffn: GeGLUFFN,
    pub layer_idx: usize,
}

/// Result from a block forward pass.
pub struct BlockOutput {
    pub hidden: Tensor,
    pub kv_cache: KVCache,
    pub max_attn_logit: f32,
}

impl MoEBlock {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        dense_ffn_dim: usize,
        expert_ffn_dim: usize,
        n_experts: usize,
        n_active_experts: usize,
        has_shared_expert: bool,
        sliding_window: usize,
        layer_idx: usize,
        is_global: bool,
        eps: f64,
        init_std: f64,
    ) -> Self {
        let attn_window = if is_global { 0 } else { sliding_window };
        // Scale init by depth for stability
        let depth_init = init_std / (2.0 * (layer_idx + 1) as f64).sqrt();

        Self {
            attn_norm: SandwichNorm::new(d_model, eps, layer_idx),
            attention: GQAAttention::new(d_model, n_heads, n_kv_heads, head_dim, attn_window, depth_init),
            ffn_norm: SandwichNorm::new(d_model, eps, layer_idx),
            parallel_ffn: ParallelFFN::new(
                d_model, dense_ffn_dim, expert_ffn_dim,
                n_experts, n_active_experts, has_shared_expert, depth_init,
            ),
            layer_idx,
            is_global,
        }
    }

    /// Forward pass.
    /// x: (seq_len, d_model)
    /// rope: appropriate RoPE for this layer type (standard or p-RoPE)
    /// kv_cache: optional cache for incremental decoding
    pub fn forward(
        &mut self,
        x: &Tensor,
        rope: &RoPE,
        kv_cache: Option<KVCache>,
    ) -> BlockOutput {
        // Attention sublayer with sandwich norm + residual
        let normed = self.attn_norm.pre(x);
        let (attn_out, max_logit, new_cache) = self.attention.forward(&normed, rope, kv_cache);
        let attn_out = self.attn_norm.post(&attn_out);
        let h = tensor::add(x, &attn_out); // residual

        // FFN sublayer with sandwich norm + residual
        let normed = self.ffn_norm.pre(&h);
        let ffn_out = self.parallel_ffn.forward(&normed);
        let ffn_out = self.ffn_norm.post(&ffn_out);
        let out = tensor::add(&h, &ffn_out); // residual

        BlockOutput {
            hidden: out,
            kv_cache: new_cache,
            max_attn_logit: max_logit,
        }
    }

    pub fn param_count(&self) -> usize {
        // 4 norm weights (2 sandwich norms × 2 each)
        let norm_params = 4 * self.attention.wq.shape[0]; // 4 × d_model
        self.attention.param_count() + self.parallel_ffn.param_count() + norm_params
    }
}

impl PLEBlock {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        dense_ffn_dim: usize,
        sliding_window: usize,
        layer_idx: usize,
        eps: f64,
        init_std: f64,
    ) -> Self {
        let depth_init = init_std / (2.0 * (layer_idx + 1) as f64).sqrt();

        Self {
            attn_norm: RMSNorm::new(d_model, eps),
            attention: GQAAttention::new(d_model, n_heads, n_kv_heads, head_dim, sliding_window, depth_init),
            ffn_norm: RMSNorm::new(d_model, eps),
            ffn: GeGLUFFN::new(d_model, dense_ffn_dim, depth_init),
            layer_idx,
        }
    }

    /// Forward pass.
    /// PLE modulation is applied externally by the model (since PLE is shared across layers).
    pub fn forward(
        &self,
        x: &Tensor,
        rope: &RoPE,
        kv_cache: Option<KVCache>,
    ) -> BlockOutput {
        // Attention sublayer with pre-norm + residual
        let normed = self.attn_norm.forward(x);
        let (attn_out, max_logit, new_cache) = self.attention.forward(&normed, rope, kv_cache);
        let h = tensor::add(x, &attn_out);

        // FFN sublayer with pre-norm + residual
        let normed = self.ffn_norm.forward(&h);
        let ffn_out = self.ffn.forward(&normed);
        let out = tensor::add(&h, &ffn_out);

        BlockOutput {
            hidden: out,
            kv_cache: new_cache,
            max_attn_logit: max_logit,
        }
    }

    pub fn param_count(&self) -> usize {
        let norm_params = 2 * self.attention.wq.shape[0]; // 2 × d_model
        self.attention.param_count() + self.ffn.param_count() + norm_params
    }
}

/// Enum wrapper for either block type.
pub enum Block {
    MoE(MoEBlock),
    PLE(PLEBlock),
}

impl Block {
    pub fn forward(
        &mut self,
        x: &Tensor,
        rope: &RoPE,
        kv_cache: Option<KVCache>,
    ) -> BlockOutput {
        match self {
            Block::MoE(b) => b.forward(x, rope, kv_cache),
            Block::PLE(b) => b.forward(x, rope, kv_cache),
        }
    }

    pub fn param_count(&self) -> usize {
        match self {
            Block::MoE(b) => b.param_count(),
            Block::PLE(b) => b.param_count(),
        }
    }

    pub fn layer_idx(&self) -> usize {
        match self {
            Block::MoE(b) => b.layer_idx,
            Block::PLE(b) => b.layer_idx,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_block_shape() {
        let mut block = MoEBlock::new(
            32, 4, 2, 8,     // d_model, heads, kv_heads, head_dim
            64, 32,           // dense_ffn, expert_ffn
            4, 2, true,       // experts, active, shared
            16,               // window
            0, false,         // layer 0, sliding
            1e-6, 0.02,
        );
        let rope = RoPE::new(8, 10000.0);
        let x = Tensor::randn(&[4, 32], 1.0);
        let out = block.forward(&x, &rope, None);
        assert_eq!(out.hidden.shape, vec![4, 32]);
    }

    #[test]
    fn test_ple_block_shape() {
        let block = PLEBlock::new(
            32, 4, 2, 8,   // d_model, heads, kv_heads, head_dim
            64,             // dense_ffn
            16,             // window
            0, 1e-6, 0.02,
        );
        let rope = RoPE::new(8, 10000.0);
        let x = Tensor::randn(&[4, 32], 1.0);
        let out = block.forward(&x, &rope, None);
        assert_eq!(out.hidden.shape, vec![4, 32]);
    }

    #[test]
    fn test_residual_connection() {
        // With tiny init, block output should be close to input (residual dominates)
        let block = PLEBlock::new(16, 2, 1, 8, 32, 8, 0, 1e-6, 0.001);
        let rope = RoPE::new(8, 10000.0);
        let x = Tensor::randn(&[2, 16], 1.0);
        let out = block.forward(&x, &rope, None);

        // Measure how close output is to input
        let diff: f32 = x.data.iter().zip(out.hidden.data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>() / x.numel() as f32;
        assert!(
            diff < 0.5,
            "With tiny init, residual should dominate. Avg diff = {diff}"
        );
    }

    #[test]
    fn test_block_enum_dispatch() {
        let moe = MoEBlock::new(16, 2, 1, 8, 32, 16, 4, 2, false, 8, 0, false, 1e-6, 0.02);
        let mut block = Block::MoE(moe);
        let rope = RoPE::new(8, 10000.0);
        let x = Tensor::randn(&[2, 16], 1.0);
        let out = block.forward(&x, &rope, None);
        assert_eq!(out.hidden.shape, vec![2, 16]);
        assert_eq!(block.layer_idx(), 0);
    }

    #[test]
    fn test_kv_cache_passes_through() {
        let mut block = MoEBlock::new(16, 2, 1, 8, 32, 16, 4, 2, false, 0, 0, true, 1e-6, 0.02);
        let rope = RoPE::new(8, 10000.0);

        // First pass
        let x1 = Tensor::randn(&[3, 16], 1.0);
        let out1 = block.forward(&x1, &rope, None);
        assert_eq!(out1.kv_cache.seq_len, 3);

        // Second pass with cache
        let x2 = Tensor::randn(&[1, 16], 1.0);
        let out2 = block.forward(&x2, &rope, Some(out1.kv_cache));
        assert_eq!(out2.kv_cache.seq_len, 4);
    }

    #[test]
    fn test_max_logit_tracked() {
        let mut block = MoEBlock::new(16, 2, 1, 8, 32, 16, 4, 2, false, 0, 0, false, 1e-6, 0.5);
        let rope = RoPE::new(8, 10000.0);
        let x = Tensor::randn(&[4, 16], 1.0);
        let out = block.forward(&x, &rope, None);
        assert!(out.max_attn_logit.is_finite());
    }
}
