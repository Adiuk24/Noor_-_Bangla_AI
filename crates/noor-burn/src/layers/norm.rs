//! Sandwich RMSNorm with depth-scaled gain.
//!
//! Pre-norm AND post-norm around each sublayer:
//!   gain(layer) = 1.0 / sqrt(layer + 1)
//!   x_normed = rms_norm(x) * gain

use burn::prelude::*;
use burn::nn::{RmsNorm, RmsNormConfig};

#[derive(Module, Debug)]
pub struct SandwichNorm<B: Backend> {
    pre_norm: RmsNorm<B>,
    post_norm: RmsNorm<B>,
    depth_scale: f32,
}

impl<B: Backend> SandwichNorm<B> {
    pub fn new(d_model: usize, eps: f64, layer_idx: usize, device: &B::Device) -> Self {
        let depth_scale = 1.0 / ((layer_idx as f32 + 1.0).sqrt());
        Self {
            pre_norm: RmsNormConfig::new(d_model).with_epsilon(eps).init(device),
            post_norm: RmsNormConfig::new(d_model).with_epsilon(eps).init(device),
            depth_scale,
        }
    }

    pub fn pre(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.pre_norm.forward(x) * self.depth_scale
    }

    pub fn post(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.post_norm.forward(x) * self.depth_scale
    }
}

/// Simple pre-norm wrapper (for Edge model without sandwich).
#[derive(Module, Debug)]
pub struct PreNorm<B: Backend> {
    norm: RmsNorm<B>,
}

impl<B: Backend> PreNorm<B> {
    pub fn new(d_model: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            norm: RmsNormConfig::new(d_model).with_epsilon(eps).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.norm.forward(x)
    }
}
