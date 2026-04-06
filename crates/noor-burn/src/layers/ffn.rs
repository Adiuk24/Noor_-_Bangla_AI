//! GeGLU and SwiGLU FFN layers.

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};

/// GeGLU FFN: down(gelu(gate(x)) * up(x))
#[derive(Module, Debug)]
pub struct GeGluFfn<B: Backend> {
    w_gate: Linear<B>,
    w_up: Linear<B>,
    w_down: Linear<B>,
}

impl<B: Backend> GeGluFfn<B> {
    pub fn new(d_model: usize, d_ffn: usize, device: &B::Device) -> Self {
        Self {
            w_gate: LinearConfig::new(d_model, d_ffn).with_bias(false).init(device),
            w_up: LinearConfig::new(d_model, d_ffn).with_bias(false).init(device),
            w_down: LinearConfig::new(d_ffn, d_model).with_bias(false).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = burn::tensor::activation::gelu(self.w_gate.forward(x.clone()));
        let up = self.w_up.forward(x);
        self.w_down.forward(gate * up)
    }
}

/// SwiGLU FFN: down(silu(gate(x)) * up(x))
#[derive(Module, Debug)]
pub struct SwiGluFfn<B: Backend> {
    w_gate: Linear<B>,
    w_up: Linear<B>,
    w_down: Linear<B>,
}

impl<B: Backend> SwiGluFfn<B> {
    pub fn new(d_model: usize, d_ffn: usize, device: &B::Device) -> Self {
        Self {
            w_gate: LinearConfig::new(d_model, d_ffn).with_bias(false).init(device),
            w_up: LinearConfig::new(d_model, d_ffn).with_bias(false).init(device),
            w_down: LinearConfig::new(d_ffn, d_model).with_bias(false).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = burn::tensor::activation::silu(self.w_gate.forward(x.clone()));
        let up = self.w_up.forward(x);
        self.w_down.forward(gate * up)
    }
}
