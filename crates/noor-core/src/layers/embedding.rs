use crate::tensor::Tensor;

/// Token embedding lookup table.
pub struct Embedding {
    /// Weight matrix: (vocab_size, d_model)
    pub weight: Tensor,
}

impl Embedding {
    /// Create embedding with random initialization N(0, std).
    pub fn new(vocab_size: usize, d_model: usize, init_std: f64) -> Self {
        Self {
            weight: Tensor::randn(&[vocab_size, d_model], init_std),
        }
    }

    /// Lookup token embeddings.
    /// token_ids: slice of token indices
    /// Returns: (seq_len, d_model) tensor
    pub fn forward(&self, token_ids: &[u32]) -> Tensor {
        let d = self.weight.shape[1];
        let seq_len = token_ids.len();
        let mut data = vec![0.0f32; seq_len * d];

        for (i, &tid) in token_ids.iter().enumerate() {
            let t = tid as usize;
            assert!(t < self.weight.shape[0], "Token id {t} out of vocab range");
            let src_offset = t * d;
            let dst_offset = i * d;
            data[dst_offset..dst_offset + d]
                .copy_from_slice(&self.weight.data[src_offset..src_offset + d]);
        }

        Tensor::from_slice(&data, &[seq_len, d])
    }

    pub fn vocab_size(&self) -> usize {
        self.weight.shape[0]
    }

    pub fn d_model(&self) -> usize {
        self.weight.shape[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_lookup() {
        let emb = Embedding::new(100, 64, 0.02);
        let ids = [0u32, 5, 99];
        let out = emb.forward(&ids);
        assert_eq!(out.shape, vec![3, 64]);

        // Token 0 should return the first row of the weight matrix
        for j in 0..64 {
            assert_eq!(out.data[j], emb.weight.data[j]);
        }
        // Token 5 should return row 5
        for j in 0..64 {
            assert_eq!(out.data[64 + j], emb.weight.data[5 * 64 + j]);
        }
    }

    #[test]
    fn test_embedding_dimensions() {
        let emb = Embedding::new(64000, 768, 0.02);
        assert_eq!(emb.vocab_size(), 64000);
        assert_eq!(emb.d_model(), 768);
    }
}
