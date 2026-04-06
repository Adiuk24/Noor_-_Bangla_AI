use crate::tensor::Tensor;

/// Per-Layer Embeddings (PLE) — edge variant alternative to MoE.
/// Each layer gets a unique learned vector that modulates behavior
/// through a gated bottleneck. From Gemma 4 E2B.
pub struct PLE {
    /// Per-layer embedding vectors: (n_layers, ple_dim)
    pub embeddings: Tensor,
    /// Gate projection: (d_model, ple_dim) — used for sigmoid gating
    pub w_gate: Tensor,
    /// Up projection: (ple_dim, d_model) — projects PLE vector to model dim
    pub w_up: Tensor,
    pub n_layers: usize,
    pub ple_dim: usize,
    pub d_model: usize,
}

impl PLE {
    pub fn new(n_layers: usize, d_model: usize, ple_dim: usize, init_std: f64) -> Self {
        Self {
            embeddings: Tensor::randn(&[n_layers, ple_dim], init_std),
            w_gate: Tensor::randn(&[d_model, ple_dim], init_std),
            w_up: Tensor::randn(&[ple_dim, d_model], init_std),
            n_layers,
            ple_dim,
            d_model,
        }
    }

    /// Apply PLE modulation. x: (seq_len, d_model), layer_idx: which layer.
    /// Returns: x + gate(x) * up(ple_vector[layer_idx])
    pub fn forward(&self, x: &Tensor, layer_idx: usize) -> Tensor {
        assert!(layer_idx < self.n_layers, "Layer {layer_idx} out of range");
        let seq_len = x.shape[0];
        let d = x.shape[1];
        assert_eq!(d, self.d_model);

        // Get this layer's PLE vector: (ple_dim,)
        let ple_start = layer_idx * self.ple_dim;
        let ple_vec = &self.embeddings.data[ple_start..ple_start + self.ple_dim];

        // Compute up-projected PLE: ple_vec @ w_up -> (d_model,)
        let mut ple_proj = vec![0.0f32; d];
        for j in 0..d {
            let mut sum = 0.0f32;
            for k in 0..self.ple_dim {
                sum += ple_vec[k] * self.w_up.data[k * d + j];
            }
            ple_proj[j] = sum;
        }

        // Apply gated modulation per token
        let mut result = x.data.clone();
        for s in 0..seq_len {
            // Gate: sigmoid(x[s] @ w_gate) -> (ple_dim,)
            // Then reduce to scalar per-dim gate by dotting with PLE
            // Simplified: gate = sigmoid(x @ w_gate @ ple_vec^T) gives per-token scalar
            // But spec says: gate = sigmoid(W_gate @ h), element-wise with (W_up @ ple_vector)
            // So: gate is (ple_dim,) sigmoid, modulation is gate * (w_up @ ple_vec)

            // gate_scores: (ple_dim,) = sigmoid(x[s] @ w_gate)
            let mut gate_scores = vec![0.0f32; self.ple_dim];
            for k in 0..self.ple_dim {
                let mut dot = 0.0f32;
                for i in 0..d {
                    dot += x.data[s * d + i] * self.w_gate.data[i * self.ple_dim + k];
                }
                gate_scores[k] = 1.0 / (1.0 + (-dot).exp()); // sigmoid
            }

            // gated_ple: (d_model,) = sum_k(gate_scores[k] * w_up[k, :])
            // This is the per-token modulation signal
            for j in 0..d {
                let mut modulation = 0.0f32;
                for k in 0..self.ple_dim {
                    modulation += gate_scores[k] * ple_vec[k] * self.w_up.data[k * d + j];
                }
                result[s * d + j] += modulation;
            }
        }

        Tensor::from_slice(&result, &[seq_len, d])
    }

    pub fn param_count(&self) -> usize {
        self.embeddings.numel() + self.w_gate.numel() + self.w_up.numel()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ple_shape() {
        let ple = PLE::new(24, 64, 16, 0.02);
        let x = Tensor::randn(&[4, 64], 1.0);
        let out = ple.forward(&x, 0);
        assert_eq!(out.shape, vec![4, 64], "PLE output should match input shape");
    }

    #[test]
    fn test_ple_different_layers_differ() {
        let ple = PLE::new(24, 32, 8, 0.5);
        let x = Tensor::ones(&[2, 32]);
        let out0 = ple.forward(&x, 0);
        let out10 = ple.forward(&x, 10);

        // Different layer indices should produce different modulations
        let diff: f32 = out0.data.iter().zip(out10.data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.01, "Different layers should produce different PLE outputs, diff={diff}");
    }

    #[test]
    fn test_ple_is_residual() {
        // PLE adds modulation to input (residual connection built-in)
        // With zero-init gate weights, sigmoid(0) = 0.5, so there's always some modulation
        let ple = PLE::new(4, 16, 4, 0.02);
        let x = Tensor::randn(&[2, 16], 1.0);
        let out = ple.forward(&x, 0);

        // Output should be close to input (small modulation with small init)
        let diff: f32 = x.data.iter().zip(out.data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>() / x.numel() as f32;
        assert!(diff < 1.0, "PLE modulation should be small with small init, avg_diff={diff}");
    }

    #[test]
    fn test_ple_param_count() {
        let ple = PLE::new(24, 1024, 128, 0.02);
        // embeddings: 24*128 + w_gate: 1024*128 + w_up: 128*1024
        let expected = 24 * 128 + 1024 * 128 + 128 * 1024;
        assert_eq!(ple.param_count(), expected);
    }
}
