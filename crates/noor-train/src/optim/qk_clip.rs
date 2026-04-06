//! QK-Clip: Monitor attention logits, rescale Q/K when they explode.
//! From Kimi K2 — prevents loss spikes with Muon optimizer at scale.
//! Post-update weight adjustment — doesn't alter current step's forward/backward.

use noor_core::tensor::Tensor;

/// QK-Clip state.
pub struct QKClip {
    /// Threshold: rescale when max attention logit exceeds this.
    pub tau: f32,
    /// Count of clips applied (for logging).
    pub clip_count: usize,
}

impl QKClip {
    pub fn new(tau: f32) -> Self {
        Self { tau, clip_count: 0 }
    }

    /// Check if clipping is needed and apply to Q/K weight matrices.
    /// max_attn_logit: maximum attention logit observed in this forward pass.
    /// wq: query weight matrix (d_model, n_heads * head_dim)
    /// wk: key weight matrix (d_model, n_kv_heads * head_dim)
    /// Returns true if clipping was applied.
    pub fn clip_if_needed(
        &mut self,
        max_attn_logit: f32,
        wq: &mut Tensor,
        wk: &mut Tensor,
    ) -> bool {
        if max_attn_logit <= self.tau {
            return false;
        }

        let scale = (self.tau / max_attn_logit).sqrt();

        // Rescale Q weights
        wq.scale_inplace(scale);

        // Rescale K weights
        wk.scale_inplace(scale);

        self.clip_count += 1;
        true
    }

    /// Apply QK-Clip across all attention layers in the model.
    /// attn_logits_per_layer: max attention logit from each layer.
    /// Returns number of layers clipped.
    pub fn clip_model(
        &mut self,
        layers: &mut [(f32, &mut Tensor, &mut Tensor)], // (max_logit, wq, wk)
    ) -> usize {
        let mut clipped = 0;
        for (max_logit, wq, wk) in layers.iter_mut() {
            if self.clip_if_needed(*max_logit, wq, wk) {
                clipped += 1;
            }
        }
        clipped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_clip_below_threshold() {
        let mut clip = QKClip::new(100.0);
        let mut wq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let mut wk = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let original_wq = wq.data.clone();

        let clipped = clip.clip_if_needed(50.0, &mut wq, &mut wk);
        assert!(!clipped);
        assert_eq!(wq.data, original_wq, "Should not modify weights below threshold");
        assert_eq!(clip.clip_count, 0);
    }

    #[test]
    fn test_clip_above_threshold() {
        let mut clip = QKClip::new(100.0);
        let mut wq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let mut wk = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        let clipped = clip.clip_if_needed(200.0, &mut wq, &mut wk);
        assert!(clipped);
        assert_eq!(clip.clip_count, 1);

        // scale = sqrt(100/200) = sqrt(0.5) ≈ 0.707
        let scale = (100.0f32 / 200.0).sqrt();
        assert!((wq.data[0] - 1.0 * scale).abs() < 1e-5);
        assert!((wk.data[0] - 5.0 * scale).abs() < 1e-5);
    }

    #[test]
    fn test_clip_reduces_magnitude() {
        let mut clip = QKClip::new(100.0);
        let mut wq = Tensor::randn(&[64, 32], 1.0);
        let mut wk = Tensor::randn(&[64, 16], 1.0);

        let before_q: f32 = wq.data.iter().map(|v| v * v).sum::<f32>();
        let before_k: f32 = wk.data.iter().map(|v| v * v).sum::<f32>();

        clip.clip_if_needed(1000.0, &mut wq, &mut wk);

        let after_q: f32 = wq.data.iter().map(|v| v * v).sum::<f32>();
        let after_k: f32 = wk.data.iter().map(|v| v * v).sum::<f32>();

        assert!(after_q < before_q, "Q magnitude should decrease after clip");
        assert!(after_k < before_k, "K magnitude should decrease after clip");
    }

    #[test]
    fn test_clip_at_exactly_threshold() {
        let mut clip = QKClip::new(100.0);
        let mut wq = Tensor::ones(&[2, 2]);
        let mut wk = Tensor::ones(&[2, 2]);

        let clipped = clip.clip_if_needed(100.0, &mut wq, &mut wk);
        assert!(!clipped, "Should not clip at exactly the threshold");
    }
}
