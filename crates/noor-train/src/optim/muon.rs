//! Muon optimizer: Newton-Schulz orthogonalization of momentum.
//! ~2x more token-efficient than AdamW with only 1 momentum state.

use noor_core::tensor::Tensor;
use std::collections::HashMap;

/// Muon optimizer state.
pub struct Muon {
    /// Per-parameter momentum: name → tensor
    pub momentum: HashMap<String, Tensor>,
    /// Momentum coefficient
    pub beta: f32,
    /// Learning rate (updated by schedule)
    pub lr: f32,
    /// Newton-Schulz iterations
    pub ns_iters: usize,
}

impl Muon {
    pub fn new(beta: f32, lr: f32) -> Self {
        Self {
            momentum: HashMap::new(),
            beta,
            lr,
            ns_iters: 5,
        }
    }

    /// Perform one optimization step.
    pub fn step(&mut self, params: &mut HashMap<String, Tensor>, grads: &HashMap<String, Tensor>) {
        for (name, grad) in grads {
            let param = match params.get_mut(name) {
                Some(p) => p,
                None => continue,
            };

            // Update momentum: m = beta * m + grad
            let m = self.momentum.entry(name.clone()).or_insert_with(|| {
                Tensor::zeros(&grad.shape)
            });
            for i in 0..m.numel() {
                m.data[i] = self.beta * m.data[i] + grad.data[i];
            }

            // Newton-Schulz orthogonalization (for 2D weight matrices)
            if m.ndim() == 2 {
                let ortho = newton_schulz_orthogonalize(m, self.ns_iters);
                // Update: param -= lr * ortho
                for i in 0..param.numel() {
                    param.data[i] -= self.lr * ortho.data[i];
                }
            } else {
                // For 1D params (biases, norms): simple momentum update
                for i in 0..param.numel() {
                    param.data[i] -= self.lr * m.data[i];
                }
            }
        }
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Newton-Schulz orthogonalization.
/// Iteratively computes the polar factor of M: the closest orthogonal matrix.
/// X = M / ||M||_F
/// for i in 0..iters: A = X @ X^T; X = 1.5*X - 0.5*A@X
fn newton_schulz_orthogonalize(m: &Tensor, iters: usize) -> Tensor {
    assert_eq!(m.ndim(), 2);
    let rows = m.shape[0];
    let cols = m.shape[1];

    // Compute Frobenius norm
    let mut frob_sq = 0.0f64;
    for &v in &m.data {
        frob_sq += (v as f64) * (v as f64);
    }
    let frob = frob_sq.sqrt();
    if frob < 1e-12 {
        return m.clone();
    }

    // X = M / ||M||_F
    let inv_frob = 1.0 / frob as f32;
    let mut x_data: Vec<f32> = m.data.iter().map(|&v| v * inv_frob).collect();

    for _ in 0..iters {
        // A = X @ X^T  (rows × rows)
        let mut a = vec![0.0f32; rows * rows];
        for i in 0..rows {
            for j in 0..rows {
                let mut dot = 0.0f32;
                for k in 0..cols {
                    dot += x_data[i * cols + k] * x_data[j * cols + k];
                }
                a[i * rows + j] = dot;
            }
        }

        // X_new = 1.5 * X - 0.5 * A @ X
        let mut x_new = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let mut ax = 0.0f32;
                for k in 0..rows {
                    ax += a[i * rows + k] * x_data[k * cols + j];
                }
                x_new[i * cols + j] = 1.5 * x_data[i * cols + j] - 0.5 * ax;
            }
        }
        x_data = x_new;
    }

    Tensor::from_slice(&x_data, &[rows, cols])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_muon_step_decreases_loss() {
        // Simple quadratic: loss = sum((W @ x - target)^2)
        let mut params = HashMap::new();
        params.insert("w".to_string(), Tensor::randn(&[4, 4], 0.5));
        let mut optim = Muon::new(0.9, 0.01);

        // Run 50 steps with random gradients pointing toward origin
        for _ in 0..50 {
            let mut grads = HashMap::new();
            // Gradient = param (points toward zero)
            grads.insert("w".to_string(), params["w"].clone());
            optim.step(&mut params, &grads);
        }

        // Params should have decreased in magnitude
        let norm: f32 = params["w"].data.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm < 2.0, "After 50 steps toward origin, norm should decrease, got {norm}");
    }

    #[test]
    fn test_newton_schulz_produces_orthogonal() {
        let m = Tensor::randn(&[4, 4], 1.0);
        let ortho = newton_schulz_orthogonalize(&m, 30); // more iterations for convergence

        // Check O @ O^T ≈ I
        let ot = ortho.transpose(0, 1);
        let prod = noor_core::tensor::matmul(&ortho, &ot);

        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let actual = prod.data[i * 4 + j];
                assert!(
                    (actual - expected).abs() < 0.15,
                    "O@O^T[{i},{j}] = {actual}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_muon_1d_params() {
        // 1D params (bias) should work without NS orthogonalization
        let mut params = HashMap::new();
        params.insert("bias".to_string(), Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]));
        let mut optim = Muon::new(0.9, 0.1);

        let mut grads = HashMap::new();
        grads.insert("bias".to_string(), Tensor::from_slice(&[0.1, 0.1, 0.1], &[3]));
        optim.step(&mut params, &grads);

        // Bias should decrease
        assert!(params["bias"].data[0] < 1.0);
    }

    #[test]
    fn test_lr_update() {
        let mut optim = Muon::new(0.9, 0.01);
        optim.set_lr(0.001);
        assert!((optim.lr - 0.001).abs() < 1e-7);
    }
}
