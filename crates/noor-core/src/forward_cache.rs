//! Forward activation cache for backward pass.
//! Stores intermediate values needed to compute gradients.

use crate::tensor::Tensor;

/// Cached activations from a single block's forward pass.
pub struct BlockCache {
    /// Input to this block (after previous block's output / embedding)
    pub input: Tensor,
    /// Input to attention sublayer (after norm)
    pub attn_norm_out: Tensor,
    /// Input to FFN sublayer (after attention + residual, then norm)
    pub ffn_norm_out: Tensor,
    /// Hidden state after attention sublayer (before residual)
    pub attn_out: Tensor,
    /// Hidden state after FFN sublayer (before residual)
    pub ffn_out: Tensor,
    /// For MoE: which experts were active per token, and their weights
    pub expert_indices: Vec<Vec<usize>>,  // per-token active expert indices
    pub expert_weights: Vec<Vec<f32>>,    // per-token expert weights
    /// Individual expert inputs/outputs for backward (only active experts)
    pub expert_inputs: Vec<Tensor>,  // one per active expert computation
}

impl BlockCache {
    pub fn empty() -> Self {
        Self {
            input: Tensor::zeros(&[0]),
            attn_norm_out: Tensor::zeros(&[0]),
            ffn_norm_out: Tensor::zeros(&[0]),
            attn_out: Tensor::zeros(&[0]),
            ffn_out: Tensor::zeros(&[0]),
            expert_indices: Vec::new(),
            expert_weights: Vec::new(),
            expert_inputs: Vec::new(),
        }
    }
}

/// Full model forward cache.
pub struct ForwardCache {
    /// Embedding output (input to first block)
    pub embedding_out: Tensor,
    /// Per-block caches
    pub block_caches: Vec<BlockCache>,
    /// Final norm input (output of last block)
    pub final_norm_input: Tensor,
    /// Final norm output (input to output projection)
    pub final_norm_out: Tensor,
    /// Token IDs (needed for embedding backward)
    pub token_ids: Vec<u32>,
}

impl ForwardCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            embedding_out: Tensor::zeros(&[0]),
            block_caches: (0..n_layers).map(|_| BlockCache::empty()).collect(),
            final_norm_input: Tensor::zeros(&[0]),
            final_norm_out: Tensor::zeros(&[0]),
            token_ids: Vec::new(),
        }
    }
}
