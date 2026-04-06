//! Manual gradient computation for all layer types.
//! No autograd graph — hand-derived backward passes for each op.

use crate::tensor::Tensor;
use std::collections::HashMap;

/// Gradient accumulator: maps param name → gradient tensor.
pub type Gradients = HashMap<String, Tensor>;

/// Cross-entropy loss backward: dL/d_logits = softmax(logits) - one_hot(targets)
pub fn cross_entropy_backward(logits: &Tensor, targets: &[u32]) -> Tensor {
    let batch = logits.shape[0];
    let vocab = logits.shape[1];

    // Compute softmax
    let mut grad = vec![0.0f32; batch * vocab];
    for b in 0..batch {
        let offset = b * vocab;
        let max_val = logits.data[offset..offset + vocab]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0f32;
        for v in 0..vocab {
            grad[offset + v] = (logits.data[offset + v] - max_val).exp();
            sum += grad[offset + v];
        }
        for v in 0..vocab {
            grad[offset + v] /= sum;
        }
        // Subtract one-hot
        grad[offset + targets[b] as usize] -= 1.0;
        // Average over batch
        for v in 0..vocab {
            grad[offset + v] /= batch as f32;
        }
    }

    Tensor::from_slice(&grad, &[batch, vocab])
}

/// Linear layer backward: y = x @ W
/// Given dL/dy, compute dL/dx and dL/dW.
pub fn linear_backward(
    grad_output: &Tensor, // (seq, out_dim)
    input: &Tensor,       // (seq, in_dim)
    weight: &Tensor,      // (in_dim, out_dim)
) -> (Tensor, Tensor) {
    // dL/dx = grad_output @ W^T
    let wt = weight.transpose(0, 1);
    let grad_input = crate::tensor::matmul(grad_output, &wt);

    // dL/dW = x^T @ grad_output
    let xt = input.transpose(0, 1);
    let grad_weight = crate::tensor::matmul(&xt, grad_output);

    (grad_input, grad_weight)
}

/// RMSNorm backward.
/// Given dL/dy and forward inputs (x, weight, normalized output), compute dL/dx and dL/dw.
pub fn rms_norm_backward(
    grad_output: &Tensor, // (*, d)
    x: &Tensor,           // (*, d)
    weight: &Tensor,      // (d,)
    eps: f64,
) -> (Tensor, Tensor) {
    let d = *x.shape.last().unwrap();
    let n = x.numel() / d;

    let mut grad_input = vec![0.0f32; x.numel()];
    let mut grad_weight = vec![0.0f32; d];

    for i in 0..n {
        let offset = i * d;
        let x_slice = &x.data[offset..offset + d];
        let go_slice = &grad_output.data[offset..offset + d];

        // Compute RMS
        let mut sum_sq = 0.0f64;
        for &v in x_slice {
            sum_sq += (v as f64) * (v as f64);
        }
        let rms = ((sum_sq / d as f64) + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Normalized values
        let mut normed = vec![0.0f64; d];
        for j in 0..d {
            normed[j] = x_slice[j] as f64 * inv_rms;
        }

        // Gradient w.r.t weight
        for j in 0..d {
            grad_weight[j] += (go_slice[j] as f64 * normed[j]) as f32;
        }

        // Gradient w.r.t input
        // d(rms_norm)/dx_j = (w_j / rms) * (1 - x_j^2 / (d * rms^2))
        // Simplified via chain rule:
        let mut dot = 0.0f64;
        for j in 0..d {
            dot += go_slice[j] as f64 * weight.data[j] as f64 * normed[j];
        }

        for j in 0..d {
            let g = go_slice[j] as f64 * weight.data[j] as f64;
            grad_input[offset + j] = ((g * inv_rms - normed[j] * dot * inv_rms / d as f64)) as f32;
        }
    }

    (
        Tensor::from_slice(&grad_input, &x.shape),
        Tensor::from_slice(&grad_weight, &[d]),
    )
}

/// Element-wise GELU backward.
pub fn gelu_backward(grad_output: &Tensor, input: &Tensor) -> Tensor {
    let sqrt_2_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let data: Vec<f32> = grad_output.data.iter().zip(input.data.iter()).map(|(&go, &x)| {
        let inner = sqrt_2_pi * (x + 0.044715 * x * x * x);
        let tanh_val = inner.tanh();
        let sech2 = 1.0 - tanh_val * tanh_val;
        let d_inner = sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x * x);
        go * (0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_inner)
    }).collect();
    Tensor::from_slice(&data, &input.shape)
}

/// Element-wise SiLU backward: d(x * sigmoid(x))/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
pub fn silu_backward(grad_output: &Tensor, input: &Tensor) -> Tensor {
    let data: Vec<f32> = grad_output.data.iter().zip(input.data.iter()).map(|(&go, &x)| {
        let sig = 1.0 / (1.0 + (-x).exp());
        go * (sig + x * sig * (1.0 - sig))
    }).collect();
    Tensor::from_slice(&data, &input.shape)
}

/// Sigmoid backward: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
pub fn sigmoid_backward(grad_output: &Tensor, sigmoid_output: &Tensor) -> Tensor {
    let data: Vec<f32> = grad_output.data.iter().zip(sigmoid_output.data.iter()).map(|(&go, &s)| {
        go * s * (1.0 - s)
    }).collect();
    Tensor::from_slice(&data, &sigmoid_output.shape)
}

/// Compute global gradient norm.
pub fn global_grad_norm(grads: &Gradients) -> f32 {
    let mut total = 0.0f64;
    for tensor in grads.values() {
        for &v in &tensor.data {
            total += (v as f64) * (v as f64);
        }
    }
    (total as f32).sqrt()
}

/// Clip gradients by global norm. Returns the actual norm.
pub fn clip_grad_norm(grads: &mut Gradients, max_norm: f32) -> f32 {
    let norm = global_grad_norm(grads);
    if norm > max_norm {
        let scale = max_norm / norm;
        for tensor in grads.values_mut() {
            tensor.scale_inplace(scale);
        }
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_backward_shape() {
        let logits = Tensor::randn(&[4, 100], 1.0);
        let targets = vec![5u32, 10, 20, 50];
        let grad = cross_entropy_backward(&logits, &targets);
        assert_eq!(grad.shape, vec![4, 100]);
    }

    #[test]
    fn test_cross_entropy_backward_sums_to_zero() {
        // Softmax - one_hot sums to 0 per row (before batch averaging)
        let logits = Tensor::randn(&[2, 50], 1.0);
        let targets = vec![5u32, 10];
        let grad = cross_entropy_backward(&logits, &targets);

        for b in 0..2 {
            let sum: f32 = grad.data[b * 50..(b + 1) * 50].iter().sum();
            assert!(
                sum.abs() < 1e-5,
                "CE backward per-row should sum to ~0, got {sum}"
            );
        }
    }

    #[test]
    fn test_linear_backward_shapes() {
        let go = Tensor::randn(&[4, 32], 0.1);
        let x = Tensor::randn(&[4, 16], 1.0);
        let w = Tensor::randn(&[16, 32], 0.1);
        let (grad_x, grad_w) = linear_backward(&go, &x, &w);
        assert_eq!(grad_x.shape, vec![4, 16]);
        assert_eq!(grad_w.shape, vec![16, 32]);
    }

    #[test]
    fn test_numerical_gradient_linear() {
        // Numerical gradient check for linear: y = x @ W, loss = sum(y)
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let w = Tensor::from_slice(&[0.5, 0.3, -0.1, 0.7], &[2, 2]);
        let go = Tensor::ones(&[2, 2]); // dL/dy = 1 (loss = sum(y))

        let (_, grad_w) = linear_backward(&go, &x, &w);

        // Numerical check: perturb w[0,0] by eps, measure loss change
        let eps = 1e-4f32;
        let y1 = crate::tensor::matmul(&x, &w);
        let loss1: f32 = y1.data.iter().sum();

        let mut w2 = w.clone();
        w2.data[0] += eps;
        let y2 = crate::tensor::matmul(&x, &w2);
        let loss2: f32 = y2.data.iter().sum();

        let numerical = (loss2 - loss1) / eps;
        let analytical = grad_w.data[0];

        assert!(
            (numerical - analytical).abs() < 0.01,
            "Gradient mismatch: numerical={numerical}, analytical={analytical}"
        );
    }

    #[test]
    fn test_gelu_backward_shape() {
        let go = Tensor::randn(&[4, 32], 0.1);
        let x = Tensor::randn(&[4, 32], 1.0);
        let grad = gelu_backward(&go, &x);
        assert_eq!(grad.shape, x.shape);
    }

    #[test]
    fn test_silu_backward_shape() {
        let go = Tensor::randn(&[4, 32], 0.1);
        let x = Tensor::randn(&[4, 32], 1.0);
        let grad = silu_backward(&go, &x);
        assert_eq!(grad.shape, x.shape);
    }

    #[test]
    fn test_grad_clipping() {
        let mut grads = Gradients::new();
        // Create a gradient with norm = 10
        grads.insert("w".to_string(), Tensor::from_slice(&[6.0, 8.0], &[2]));
        let norm = global_grad_norm(&grads);
        assert!((norm - 10.0).abs() < 0.01);

        let clipped_norm = clip_grad_norm(&mut grads, 1.0);
        assert!((clipped_norm - 10.0).abs() < 0.01); // returns pre-clip norm

        let new_norm = global_grad_norm(&grads);
        assert!((new_norm - 1.0).abs() < 0.01, "After clipping, norm should be ~1.0, got {new_norm}");
    }

    #[test]
    fn test_rms_norm_backward_shape() {
        let go = Tensor::randn(&[4, 32], 0.1);
        let x = Tensor::randn(&[4, 32], 1.0);
        let w = Tensor::ones(&[32]);
        let (grad_x, grad_w) = rms_norm_backward(&go, &x, &w, 1e-6);
        assert_eq!(grad_x.shape, vec![4, 32]);
        assert_eq!(grad_w.shape, vec![32]);
    }
}
