//! GQA (Grouped Query Attention) with RoPE and sliding window.
//!
//! Pro: 16Q / 4KV heads, window=1024, global every 6th layer (p-RoPE)
//! Edge: 8Q / 2KV heads, window=512
//! Proxy: 12Q / 4KV heads, window=512

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, RotaryEncoding, RotaryEncodingConfig};

#[derive(Module, Debug)]
pub struct GqaAttention<B: Backend> {
    wq: Linear<B>,
    wk: Linear<B>,
    wv: Linear<B>,
    wo: Linear<B>,
    rope: RotaryEncoding<B>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    is_global: bool,
}

impl<B: Backend> GqaAttention<B> {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        is_global: bool,
        rope_theta: f64,
        max_seq_len: usize,
        device: &B::Device,
    ) -> Self {
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        Self {
            wq: LinearConfig::new(d_model, q_dim).with_bias(false).init(device),
            wk: LinearConfig::new(d_model, kv_dim).with_bias(false).init(device),
            wv: LinearConfig::new(d_model, kv_dim).with_bias(false).init(device),
            wo: LinearConfig::new(q_dim, d_model).with_bias(false).init(device),
            rope: RotaryEncodingConfig::new(max_seq_len * 2, head_dim)
                .with_theta(rope_theta as f32)
                .init(device),
            n_heads,
            n_kv_heads,
            head_dim,
            sliding_window,
            is_global,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, _d] = x.dims();
        let scale = (self.head_dim as f32).sqrt();

        // Project Q, K, V
        let q = self.wq.forward(x.clone());
        let k = self.wk.forward(x.clone());
        let v = self.wv.forward(x);

        // Reshape to [batch, seq, n_heads, head_dim]
        let q = q.reshape([batch, seq, self.n_heads, self.head_dim]);
        let k = k.reshape([batch, seq, self.n_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq, self.n_kv_heads, self.head_dim]);

        // Apply RoPE — expects [batch, seq, n_heads, head_dim]
        let q = self.rope.forward(q);
        let k = self.rope.forward(k);

        // Transpose to [batch, n_heads, seq, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Repeat KV heads for GQA: each KV head serves (n_heads / n_kv_heads) Q heads
        let groups = self.n_heads / self.n_kv_heads;
        let k = k.repeat_dim(1, groups);
        let v = v.repeat_dim(1, groups);

        // Attention scores: Q @ K^T / sqrt(d)
        let scores = q.matmul(k.swap_dims(2, 3)) / scale;

        // Causal mask + sliding window
        let mask = self.build_mask(seq, &scores.device());
        let scores = scores + mask;

        // Softmax
        let weights = burn::tensor::activation::softmax(scores, 3);

        // Weighted sum: weights @ V
        let out = weights.matmul(v); // [batch, n_heads, seq, head_dim]

        // Reshape back to [batch, seq, q_dim]
        let out = out.swap_dims(1, 2).reshape([batch, seq, self.n_heads * self.head_dim]);

        // Output projection
        self.wo.forward(out)
    }

    fn build_mask(&self, seq_len: usize, device: &B::Device) -> Tensor<B, 4> {
        // Causal mask: -inf for future positions
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        let neg_inf = -1e9f32;

        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    // Future position — mask
                    mask_data[i * seq_len + j] = neg_inf;
                } else if !self.is_global && i - j > self.sliding_window {
                    // Outside sliding window — mask (only for non-global layers)
                    mask_data[i * seq_len + j] = neg_inf;
                }
            }
        }

        Tensor::<B, 1>::from_floats(mask_data.as_slice(), device)
            .reshape([1, 1, seq_len, seq_len])
    }
}
