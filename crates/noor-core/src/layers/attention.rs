use crate::tensor::{self, Tensor};
use crate::layers::rope::RoPE;

/// KV Cache for incremental decoding.
pub struct KVCache {
    /// Cached keys: (n_kv_heads, cached_len, head_dim)
    pub k: Tensor,
    /// Cached values: (n_kv_heads, cached_len, head_dim)
    pub v: Tensor,
    /// Current cached sequence length
    pub seq_len: usize,
}

impl KVCache {
    pub fn empty(n_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            k: Tensor::zeros(&[n_kv_heads, 0, head_dim]),
            v: Tensor::zeros(&[n_kv_heads, 0, head_dim]),
            seq_len: 0,
        }
    }
}

/// Grouped Query Attention with optional sliding window.
pub struct GQAAttention {
    /// Query projection: (d_model, n_heads * head_dim)
    pub wq: Tensor,
    /// Key projection: (d_model, n_kv_heads * head_dim)
    pub wk: Tensor,
    /// Value projection: (d_model, n_kv_heads * head_dim)
    pub wv: Tensor,
    /// Output projection: (n_heads * head_dim, d_model)
    pub wo: Tensor,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Sliding window size (0 = global attention)
    pub sliding_window: usize,
}

impl GQAAttention {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        init_std: f64,
    ) -> Self {
        Self {
            wq: Tensor::randn(&[d_model, n_heads * head_dim], init_std),
            wk: Tensor::randn(&[d_model, n_kv_heads * head_dim], init_std),
            wv: Tensor::randn(&[d_model, n_kv_heads * head_dim], init_std),
            wo: Tensor::randn(&[n_heads * head_dim, d_model], init_std),
            n_heads,
            n_kv_heads,
            head_dim,
            sliding_window,
        }
    }

    /// Forward pass.
    /// x: (seq_len, d_model)
    /// rope: RoPE instance for position encoding
    /// kv_cache: optional cache for incremental decoding
    /// Returns: (output, max_attn_logit, updated_kv_cache)
    pub fn forward(
        &self,
        x: &Tensor,
        rope: &RoPE,
        kv_cache: Option<KVCache>,
    ) -> (Tensor, f32, KVCache) {
        let seq_len = x.shape[0];
        let nh = self.n_heads;
        let nkv = self.n_kv_heads;
        let hd = self.head_dim;
        let group_size = nh / nkv; // how many Q heads per KV head

        // Project Q, K, V
        let q_flat = tensor::matmul(x, &self.wq); // (seq, nh*hd)
        let k_flat = tensor::matmul(x, &self.wk); // (seq, nkv*hd)
        let v_flat = tensor::matmul(x, &self.wv); // (seq, nkv*hd)

        // Reshape to (n_heads, seq, head_dim)
        let q = q_flat.reshape(&[seq_len, nh, hd]);
        let k_new = k_flat.reshape(&[seq_len, nkv, hd]);
        let v_new = v_flat.reshape(&[seq_len, nkv, hd]);

        // Transpose to (heads, seq, hd) for attention
        let q = q.transpose(0, 1); // (nh, seq, hd)
        let k_new = k_new.transpose(0, 1); // (nkv, seq, hd)
        let v_new = v_new.transpose(0, 1); // (nkv, seq, hd)

        // Get cache offset for RoPE
        let cache_len = kv_cache.as_ref().map_or(0, |c| c.seq_len);

        // Apply RoPE to Q and K
        let q = rope.apply(&q, cache_len);
        let k_new = rope.apply(&k_new, cache_len);

        // Concat with KV cache
        let (k_full, v_full, full_len) = if let Some(cache) = kv_cache {
            if cache.seq_len > 0 {
                let k = concat_seq(&cache.k, &k_new, nkv, hd);
                let v = concat_seq(&cache.v, &v_new, nkv, hd);
                let fl = cache.seq_len + seq_len;
                (k, v, fl)
            } else {
                (k_new.clone(), v_new.clone(), seq_len)
            }
        } else {
            (k_new.clone(), v_new.clone(), seq_len)
        };

        // Compute attention scores for each Q head
        let scale = 1.0 / (hd as f32).sqrt();
        let mut output_data = vec![0.0f32; nh * seq_len * hd];
        let mut max_logit: f32 = f32::NEG_INFINITY;

        for qh in 0..nh {
            let kv_idx = qh / group_size; // which KV head this Q head uses

            for sq in 0..seq_len {
                let q_pos = cache_len + sq; // absolute position of this query

                // Compute attention scores against all keys
                let mut scores = vec![f32::NEG_INFINITY; full_len];
                for sk in 0..full_len {
                    // Causal mask: can only attend to positions <= current
                    if sk > q_pos {
                        continue;
                    }
                    // Sliding window mask
                    if self.sliding_window > 0 && q_pos > sk + self.sliding_window {
                        continue;
                    }

                    // Dot product Q[qh, sq, :] · K[kv_idx, sk, :]
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        let qi = q.data[qh * seq_len * hd + sq * hd + d];
                        let ki = k_full.data[kv_idx * full_len * hd + sk * hd + d];
                        dot += qi * ki;
                    }
                    scores[sk] = dot * scale;
                    if scores[sk] > max_logit {
                        max_logit = scores[sk];
                    }
                }

                // Softmax over scores
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_scores: Vec<f32> = scores.iter().map(|&s| {
                    if s == f32::NEG_INFINITY { 0.0 } else { (s - max_s).exp() }
                }).collect();
                let sum: f32 = exp_scores.iter().sum();
                if sum > 0.0 {
                    for s in exp_scores.iter_mut() {
                        *s /= sum;
                    }
                }

                // Weighted sum of values
                for d in 0..hd {
                    let mut val = 0.0f32;
                    for sk in 0..full_len {
                        if exp_scores[sk] > 0.0 {
                            val += exp_scores[sk] * v_full.data[kv_idx * full_len * hd + sk * hd + d];
                        }
                    }
                    output_data[qh * seq_len * hd + sq * hd + d] = val;
                }
            }
        }

        // Reshape output: (nh, seq, hd) -> (seq, nh, hd) -> (seq, nh*hd)
        let attn_out = Tensor::from_slice(&output_data, &[nh, seq_len, hd]);
        let attn_out = attn_out.transpose(0, 1); // (seq, nh, hd)
        let attn_out = attn_out.reshape(&[seq_len, nh * hd]);

        // Output projection
        let output = tensor::matmul(&attn_out, &self.wo);

        // Build updated cache
        let new_cache = KVCache {
            k: k_full,
            v: v_full,
            seq_len: full_len,
        };

        (output, max_logit, new_cache)
    }

    pub fn param_count(&self) -> usize {
        self.wq.numel() + self.wk.numel() + self.wv.numel() + self.wo.numel()
    }
}

/// Concatenate new KV along the sequence dimension.
/// existing: (n_heads, old_len, hd), new_kv: (n_heads, new_len, hd)
/// -> (n_heads, old_len + new_len, hd)
fn concat_seq(existing: &Tensor, new_kv: &Tensor, n_heads: usize, hd: usize) -> Tensor {
    let old_len = existing.shape[1];
    let new_len = new_kv.shape[1];
    let total_len = old_len + new_len;
    let mut data = vec![0.0f32; n_heads * total_len * hd];

    for h in 0..n_heads {
        // Copy existing
        for s in 0..old_len {
            for d in 0..hd {
                data[h * total_len * hd + s * hd + d] =
                    existing.data[h * old_len * hd + s * hd + d];
            }
        }
        // Copy new
        for s in 0..new_len {
            for d in 0..hd {
                data[h * total_len * hd + (old_len + s) * hd + d] =
                    new_kv.data[h * new_len * hd + s * hd + d];
            }
        }
    }

    Tensor::from_slice(&data, &[n_heads, total_len, hd])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_output_shape() {
        let attn = GQAAttention::new(64, 4, 2, 16, 0, 0.02); // global attention
        let rope = RoPE::new(16, 10000.0);
        let x = Tensor::randn(&[8, 64], 1.0); // seq=8, d_model=64
        let (out, _max_logit, cache) = attn.forward(&x, &rope, None);
        assert_eq!(out.shape, vec![8, 64], "Output shape should match input");
        assert_eq!(cache.seq_len, 8);
    }

    #[test]
    fn test_causal_mask() {
        // With causal attention, output at position 0 should only depend on position 0
        let attn = GQAAttention::new(16, 2, 1, 8, 0, 0.5);
        let rope = RoPE::new(8, 10000.0);

        // Create input where position 0 is all ones, rest is all zeros
        let mut data = vec![0.0f32; 4 * 16]; // seq=4, d=16
        for i in 0..16 {
            data[i] = 1.0; // position 0
        }
        let x = Tensor::from_slice(&data, &[4, 16]);
        let (out1, _, _) = attn.forward(&x, &rope, None);

        // Change position 3 (future) — should not affect position 0's output
        let mut data2 = data.clone();
        for i in 0..16 {
            data2[3 * 16 + i] = 99.0; // position 3 changed
        }
        let x2 = Tensor::from_slice(&data2, &[4, 16]);
        let (out2, _, _) = attn.forward(&x2, &rope, None);

        // Position 0 output should be identical (causal mask)
        for i in 0..16 {
            assert!(
                (out1.data[i] - out2.data[i]).abs() < 1e-5,
                "Position 0 output changed when future changed: {} vs {}",
                out1.data[i], out2.data[i]
            );
        }
    }

    #[test]
    fn test_sliding_window() {
        // Window=2: position 5 should only attend to positions 3,4,5
        let attn = GQAAttention::new(16, 2, 1, 8, 2, 0.5);
        let rope = RoPE::new(8, 10000.0);

        // All ones input
        let x = Tensor::ones(&[8, 16]); // seq=8
        let (out1, _, _) = attn.forward(&x, &rope, None);

        // Change position 0 — should NOT affect position 5 (outside window)
        let mut data = vec![1.0f32; 8 * 16];
        for i in 0..16 {
            data[i] = 99.0; // change position 0
        }
        let x2 = Tensor::from_slice(&data, &[8, 16]);
        let (out2, _, _) = attn.forward(&x2, &rope, None);

        // Position 5's output should be the same (pos 0 is outside window of 2)
        let pos5_start = 5 * 16;
        for i in 0..16 {
            assert!(
                (out1.data[pos5_start + i] - out2.data[pos5_start + i]).abs() < 1e-4,
                "Sliding window: pos 0 change affected pos 5 output at dim {i}"
            );
        }
    }

    #[test]
    fn test_gqa_kv_sharing() {
        // 4 Q heads, 2 KV heads → each KV head serves 2 Q heads
        let attn = GQAAttention::new(32, 4, 2, 8, 0, 0.02);
        assert_eq!(attn.n_heads / attn.n_kv_heads, 2);
        let rope = RoPE::new(8, 10000.0);
        let x = Tensor::randn(&[4, 32], 1.0);
        let (out, _, _) = attn.forward(&x, &rope, None);
        assert_eq!(out.shape, vec![4, 32]);
    }

    #[test]
    fn test_kv_cache_incremental() {
        let attn = GQAAttention::new(16, 2, 1, 8, 0, 0.5);
        let rope = RoPE::new(8, 10000.0);

        // First token
        let x1 = Tensor::randn(&[1, 16], 1.0);
        let (_, _, cache1) = attn.forward(&x1, &rope, None);
        assert_eq!(cache1.seq_len, 1);

        // Second token with cache
        let x2 = Tensor::randn(&[1, 16], 1.0);
        let (_, _, cache2) = attn.forward(&x2, &rope, Some(cache1));
        assert_eq!(cache2.seq_len, 2);

        // Third token
        let x3 = Tensor::randn(&[1, 16], 1.0);
        let (_, _, cache3) = attn.forward(&x3, &rope, Some(cache2));
        assert_eq!(cache3.seq_len, 3);
    }

    #[test]
    fn test_max_logit_tracked() {
        let attn = GQAAttention::new(16, 2, 1, 8, 0, 1.0);
        let rope = RoPE::new(8, 10000.0);
        let x = Tensor::randn(&[4, 16], 1.0);
        let (_, max_logit, _) = attn.forward(&x, &rope, None);
        // Max logit should be a finite number
        assert!(max_logit.is_finite(), "Max logit should be finite, got {max_logit}");
    }
}
