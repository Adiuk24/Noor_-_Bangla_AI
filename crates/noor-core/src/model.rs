use crate::config::ModelConfig;
use crate::tensor::{self, Tensor};
use crate::layers::block::{Block, MoEBlock, PLEBlock};
use crate::layers::attention::KVCache;
use crate::layers::embedding::Embedding;
use crate::layers::norm::RMSNorm;
use crate::layers::ple::PLE;
use crate::layers::rope::RoPE;

/// Full Noor model: embedding → N transformer blocks → final norm → output projection.
pub struct NoorModel {
    pub config: ModelConfig,
    pub embedding: Embedding,
    pub blocks: Vec<Block>,
    pub final_norm: RMSNorm,
    /// Output projection: (d_model, vocab_size). Can be tied with embedding.
    pub output_proj: Tensor,
    /// Standard RoPE for sliding attention layers
    pub rope: RoPE,
    /// p-RoPE for global attention layers (if used)
    pub prope: Option<RoPE>,
    /// PLE module (for edge variant only)
    pub ple: Option<PLE>,
}

/// Forward pass result.
pub struct ModelOutput {
    /// Logits: (seq_len, vocab_size)
    pub logits: Tensor,
    /// Updated KV caches per layer
    pub kv_caches: Vec<KVCache>,
    /// Max attention logit across all layers (for QK-Clip)
    pub max_attn_logit: f32,
    /// Expert utilization fractions per MoE layer
    pub expert_utilization: Vec<Vec<f32>>,
}

impl NoorModel {
    /// Create model from config with random initialization.
    pub fn from_config(config: &ModelConfig) -> Self {
        let d = config.model.d_model;
        let v = config.model.vocab_size;
        let nl = config.model.n_layers;
        let init_std = 0.02;

        // Embedding
        let embedding = Embedding::new(v, d, init_std);

        // RoPE
        let rope = RoPE::new(config.model.head_dim, config.rope.theta);
        let prope = if config.attention.global_every_n > 0 && config.rope.prope_fraction > 0.0 {
            Some(RoPE::new_proportional(
                config.model.head_dim,
                config.rope.prope_theta,
                config.rope.prope_fraction,
            ))
        } else {
            None
        };

        // Build blocks
        let mut blocks = Vec::with_capacity(nl);
        for layer_idx in 0..nl {
            if config.moe.enabled {
                let is_global = config.attention.global_every_n > 0
                    && (layer_idx + 1) % config.attention.global_every_n == 0;

                blocks.push(Block::MoE(MoEBlock::new(
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
                    layer_idx,
                    is_global,
                    config.norm.eps,
                    init_std,
                )));
            } else {
                blocks.push(Block::PLE(PLEBlock::new(
                    d,
                    config.model.n_heads,
                    config.model.n_kv_heads,
                    config.model.head_dim,
                    config.moe.dense_ffn_dim,
                    config.attention.sliding_window,
                    layer_idx,
                    config.norm.eps,
                    init_std,
                )));
            }
        }

        // Final norm
        let final_norm = RMSNorm::new(d, config.norm.eps);

        // Output projection (not tied — separate from embedding)
        let output_proj = Tensor::randn(&[d, v], init_std);

        // PLE module (edge variant)
        let ple = if config.ple.enabled {
            Some(PLE::new(nl, d, config.ple.ple_dim, init_std))
        } else {
            None
        };

        Self {
            config: config.clone(),
            embedding,
            blocks,
            final_norm,
            output_proj,
            rope,
            prope,
            ple,
        }
    }

    /// Forward pass.
    /// token_ids: (seq_len,) as slice of u32
    /// kv_caches: optional per-layer KV caches for incremental decoding
    pub fn forward(
        &mut self,
        token_ids: &[u32],
        kv_caches: Option<Vec<KVCache>>,
    ) -> ModelOutput {
        let nl = self.blocks.len();

        // Token embedding
        let mut h = self.embedding.forward(token_ids);

        // Per-layer caches
        let mut old_caches = kv_caches.unwrap_or_else(|| {
            (0..nl).map(|_| KVCache::empty(
                self.config.model.n_kv_heads,
                self.config.model.head_dim,
            )).collect()
        });

        let mut new_caches = Vec::with_capacity(nl);
        let mut max_logit = f32::NEG_INFINITY;
        let mut expert_util = Vec::new();

        for (i, block) in self.blocks.iter_mut().enumerate() {
            // Choose RoPE: global layers use p-RoPE if available
            let is_global = match block {
                Block::MoE(b) => b.is_global,
                Block::PLE(_) => false,
            };
            let rope = if is_global {
                self.prope.as_ref().unwrap_or(&self.rope)
            } else {
                &self.rope
            };

            let cache = if i < old_caches.len() {
                // Take ownership of the cache for this layer
                let mut placeholder = KVCache::empty(
                    self.config.model.n_kv_heads,
                    self.config.model.head_dim,
                );
                std::mem::swap(&mut old_caches[i], &mut placeholder);
                Some(placeholder)
            } else {
                None
            };

            let out = block.forward(&h, rope, cache);

            if out.max_attn_logit > max_logit {
                max_logit = out.max_attn_logit;
            }

            // Collect expert utilization for MoE layers
            if let Block::MoE(moe_block) = block {
                expert_util.push(moe_block.parallel_ffn.moe.utilization.fractions());
                moe_block.parallel_ffn.moe.utilization.reset();
            }

            // Apply PLE modulation (edge variant)
            if let Some(ref ple) = self.ple {
                h = ple.forward(&out.hidden, i);
            } else {
                h = out.hidden;
            }

            new_caches.push(out.kv_cache);
        }

        // Final norm
        h = self.final_norm.forward(&h);

        // Output projection → logits
        let logits = tensor::matmul(&h, &self.output_proj);

        ModelOutput {
            logits,
            kv_caches: new_caches,
            max_attn_logit: max_logit,
            expert_utilization: expert_util,
        }
    }

    /// Forward pass with activation caching for backward.
    /// Returns (ModelOutput, ForwardCache).
    pub fn forward_with_cache(
        &mut self,
        token_ids: &[u32],
    ) -> (ModelOutput, crate::forward_cache::ForwardCache) {
        let nl = self.blocks.len();
        let mut fwd_cache = crate::forward_cache::ForwardCache::new(nl);
        fwd_cache.token_ids = token_ids.to_vec();

        // Token embedding
        let mut h = self.embedding.forward(token_ids);
        fwd_cache.embedding_out = h.clone();

        let mut new_caches = Vec::with_capacity(nl);
        let mut max_logit = f32::NEG_INFINITY;
        let mut expert_util = Vec::new();

        for (i, block) in self.blocks.iter_mut().enumerate() {
            // Cache block input
            fwd_cache.block_caches[i].input = h.clone();

            let is_global = match block {
                Block::MoE(b) => b.is_global,
                Block::PLE(_) => false,
            };
            let rope = if is_global {
                self.prope.as_ref().unwrap_or(&self.rope)
            } else {
                &self.rope
            };

            let kv = KVCache::empty(self.config.model.n_kv_heads, self.config.model.head_dim);
            let out = block.forward(&h, rope, Some(kv));

            if out.max_attn_logit > max_logit {
                max_logit = out.max_attn_logit;
            }

            if let Block::MoE(moe_block) = block {
                expert_util.push(moe_block.parallel_ffn.moe.utilization.fractions());
                moe_block.parallel_ffn.moe.utilization.reset();
            }

            if let Some(ref ple) = self.ple {
                h = ple.forward(&out.hidden, i);
            } else {
                h = out.hidden;
            }

            new_caches.push(out.kv_cache);
        }

        // Cache final norm input/output
        fwd_cache.final_norm_input = h.clone();
        h = self.final_norm.forward(&h);
        fwd_cache.final_norm_out = h.clone();

        // Output projection
        let logits = tensor::matmul(&h, &self.output_proj);

        let output = ModelOutput {
            logits,
            kv_caches: new_caches,
            max_attn_logit: max_logit,
            expert_utilization: expert_util,
        };

        (output, fwd_cache)
    }

    /// Count total parameters.
    pub fn param_count_total(&self) -> usize {
        let embed = self.embedding.weight.numel();
        let blocks: usize = self.blocks.iter().map(|b| b.param_count()).sum();
        let final_norm = self.final_norm.weight.numel();
        let output = self.output_proj.numel();
        let ple = self.ple.as_ref().map_or(0, |p| p.param_count());
        embed + blocks + final_norm + output + ple
    }

    /// Generate text greedily (for testing).
    pub fn generate_greedy(&mut self, prompt_ids: &[u32], max_new_tokens: usize) -> Vec<u32> {
        let mut all_ids = prompt_ids.to_vec();

        // Prefill: process entire prompt
        let out = self.forward(prompt_ids, None);
        let mut caches = out.kv_caches;

        // Get last token's logits, pick argmax
        let vocab = self.config.model.vocab_size;
        let last_start = (prompt_ids.len() - 1) * vocab;
        let last_logits = &out.logits.data[last_start..last_start + vocab];
        let next_id = argmax(last_logits);
        all_ids.push(next_id as u32);

        // Decode: one token at a time
        for _ in 1..max_new_tokens {
            let token = [*all_ids.last().unwrap()];
            let out = self.forward(&token, Some(caches));
            caches = out.kv_caches;

            let logits = &out.logits.data[..vocab]; // single token output
            let next_id = argmax(logits);
            all_ids.push(next_id as u32);
        }

        all_ids
    }
}

fn argmax(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn load_proxy_config() -> ModelConfig {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .parent().unwrap()
            .join("config/proxy.toml");
        ModelConfig::from_toml(&path).unwrap()
    }

    #[test]
    fn test_model_forward_shape() {
        let config = load_proxy_config();
        let mut model = NoorModel::from_config(&config);

        let tokens = [1u32, 2, 3, 4, 5];
        let out = model.forward(&tokens, None);

        assert_eq!(
            out.logits.shape,
            vec![5, config.model.vocab_size],
            "Logits should be (seq_len, vocab_size)"
        );
        assert_eq!(out.kv_caches.len(), config.model.n_layers);
    }

    #[test]
    fn test_model_param_count() {
        let config = load_proxy_config();
        let model = NoorModel::from_config(&config);

        let total = model.param_count_total();
        let config_estimate = config.param_count_total();
        println!("Model actual params: {total}");
        println!("Config estimate params: {config_estimate}");

        // Should be in the same ballpark (within 20%)
        let ratio = total as f64 / config_estimate as f64;
        assert!(
            ratio > 0.8 && ratio < 1.2,
            "Param count mismatch: model={total}, config_estimate={config_estimate}, ratio={ratio:.2}"
        );
    }

    #[test]
    fn test_model_incremental_decode() {
        let config = load_proxy_config();
        let mut model = NoorModel::from_config(&config);

        // Prefill
        let tokens = [1u32, 2, 3];
        let out1 = model.forward(&tokens, None);
        assert_eq!(out1.kv_caches.len(), config.model.n_layers);
        assert_eq!(out1.kv_caches[0].seq_len, 3);

        // Decode one more token
        let next = [4u32];
        let out2 = model.forward(&next, Some(out1.kv_caches));
        assert_eq!(out2.logits.shape, vec![1, config.model.vocab_size]);
        assert_eq!(out2.kv_caches[0].seq_len, 4);
    }

    #[test]
    fn test_model_expert_utilization() {
        let config = load_proxy_config();
        let mut model = NoorModel::from_config(&config);

        let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8]; // 8 tokens
        let out = model.forward(&tokens, None);

        // Should have utilization for each MoE layer
        assert!(
            !out.expert_utilization.is_empty(),
            "Proxy model has MoE, should report utilization"
        );
        // Each util vector should have n_experts entries
        for (i, util) in out.expert_utilization.iter().enumerate() {
            assert_eq!(
                util.len(), config.moe.n_experts,
                "Layer {i} utilization should have {} entries", config.moe.n_experts
            );
            // Sum of fractions should be roughly n_active/1.0 (each token activates n_active)
            // Actually fractions are per-expert, they should sum to n_active
            let sum: f32 = util.iter().sum();
            let expected = config.moe.n_active_experts as f32;
            assert!(
                (sum - expected).abs() < 0.5,
                "Layer {i} util sum = {sum}, expected ~{expected}"
            );
        }
    }

    #[test]
    fn test_model_generate() {
        let config = load_proxy_config();
        let mut model = NoorModel::from_config(&config);

        let prompt = [1u32, 100, 500];
        let generated = model.generate_greedy(&prompt, 5);

        assert_eq!(generated.len(), 3 + 5, "Should have prompt + 5 new tokens");
        // First 3 should be the prompt
        assert_eq!(&generated[..3], &prompt);
        // Generated tokens should be valid (< vocab_size)
        for &t in &generated[3..] {
            assert!(
                (t as usize) < config.model.vocab_size,
                "Generated token {t} exceeds vocab size"
            );
        }
    }

    #[test]
    fn test_max_attn_logit_finite() {
        let config = load_proxy_config();
        let mut model = NoorModel::from_config(&config);
        let tokens = [1u32, 2, 3, 4];
        let out = model.forward(&tokens, None);
        assert!(
            out.max_attn_logit.is_finite(),
            "Max attn logit should be finite: {}",
            out.max_attn_logit
        );
    }

    #[test]
    fn test_edge_model_forward() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .parent().unwrap()
            .join("config/edge.toml");
        let config = ModelConfig::from_toml(&path).unwrap();
        let mut model = NoorModel::from_config(&config);

        let tokens = [1u32, 2, 3];
        let out = model.forward(&tokens, None);
        assert_eq!(out.logits.shape, vec![3, config.model.vocab_size]);
        // Edge has no MoE, so no expert utilization
        assert!(
            out.expert_utilization.is_empty(),
            "Edge model should have no expert utilization"
        );
    }
}
