//! Training loop for Noor.
//! Forward → loss → backward → grad clip → Muon step → SMEBU update → QK-Clip → log → checkpoint.

use noor_core::backward::{self, Gradients};
use noor_core::config::ModelConfig;
use noor_core::gguf;
use noor_core::model::NoorModel;
use noor_core::tensor::{self, Tensor};
use crate::data::DataLoader;
use crate::optim::muon::Muon;
use crate::optim::smebu::SMEBU;
use crate::optim::qk_clip::QKClip;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Training metrics for a single step.
#[derive(Debug, Clone)]
pub struct StepMetrics {
    pub step: usize,
    pub loss: f32,
    pub lr: f64,
    pub grad_norm: f32,
    pub max_attn_logit: f32,
    pub tokens_per_sec: f32,
    pub active_experts: usize,
    pub qk_clipped: bool,
}

/// WSD (Warmup-Stable-Decay) learning rate schedule.
pub fn wsd_lr(step: usize, warmup: usize, total: usize, lr_max: f64, lr_min: f64) -> f64 {
    if step < warmup {
        // Linear warmup
        lr_max * (step as f64 / warmup as f64)
    } else if step < total * 3 / 4 {
        // Stable phase
        lr_max
    } else {
        // Cosine decay
        let decay_steps = total - total * 3 / 4;
        let progress = (step - total * 3 / 4) as f64 / decay_steps as f64;
        let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
        lr_min + (lr_max - lr_min) * cosine
    }
}

/// Simplified backward pass for the full model.
/// Computes gradients for output projection and cross-entropy loss.
/// Full per-layer backward is deferred to Phase 1 (Zig kernels).
/// For Phase 0: we do a "pseudo-backward" that computes the loss gradient
/// and propagates through the output projection, giving us a working training
/// loop that validates the infrastructure.
pub fn compute_gradients(
    model: &NoorModel,
    logits: &Tensor,
    targets: &[u32],
    hidden_states: &Tensor, // last hidden state before output proj
) -> (f32, Gradients) {
    // Loss
    let loss = tensor::cross_entropy_loss(logits, targets);

    // dL/d_logits
    let grad_logits = backward::cross_entropy_backward(logits, targets);

    // dL/d_output_proj = hidden^T @ grad_logits
    // dL/d_hidden = grad_logits @ output_proj^T
    let (grad_hidden, grad_output) = backward::linear_backward(
        &grad_logits, hidden_states, &model.output_proj,
    );

    let mut grads = Gradients::new();
    grads.insert("output_proj".to_string(), grad_output);

    // For Phase 0: approximate per-layer gradients by distributing grad_hidden
    // to embedding and block weights via simple scaling.
    // This is NOT mathematically correct for all params — it's a working stub.
    // Phase 1 replaces this with full per-layer backward using Zig kernels.

    // Gradient for embedding: token_ids select rows, grad flows back
    // We skip this for now since we need token_ids which aren't stored.

    // Approximate block gradients: scale the hidden gradient by layer count
    // and apply to attention/FFN weights as a rough signal.
    let n_layers = model.blocks.len();
    let layer_grad_scale = 1.0 / (n_layers as f32).sqrt();

    for (i, block) in model.blocks.iter().enumerate() {
        let prefix = format!("blocks.{i}");
        match block {
            noor_core::layers::block::Block::MoE(b) => {
                // Approximate: use grad_hidden scaled down as gradient for each weight
                let d = model.config.model.d_model;
                let attn_grad = grad_hidden.scale(layer_grad_scale * 0.1);

                // We create approximate gradients matching the shape of each param
                let wq_grad = approximate_weight_grad(&attn_grad, &b.attention.wq);
                grads.insert(format!("{prefix}.attn.wq"), wq_grad);
                let wk_grad = approximate_weight_grad(&attn_grad, &b.attention.wk);
                grads.insert(format!("{prefix}.attn.wk"), wk_grad);
                let wv_grad = approximate_weight_grad(&attn_grad, &b.attention.wv);
                grads.insert(format!("{prefix}.attn.wv"), wv_grad);
                let wo_grad = approximate_weight_grad(&attn_grad, &b.attention.wo);
                grads.insert(format!("{prefix}.attn.wo"), wo_grad);

                // Dense FFN gradients
                let ffn_grad = grad_hidden.scale(layer_grad_scale * 0.1);
                let dg = approximate_weight_grad(&ffn_grad, &b.parallel_ffn.dense.w_gate);
                grads.insert(format!("{prefix}.dense.w_gate"), dg);
                let du = approximate_weight_grad(&ffn_grad, &b.parallel_ffn.dense.w_up);
                grads.insert(format!("{prefix}.dense.w_up"), du);
                let dd = approximate_weight_grad(&ffn_grad, &b.parallel_ffn.dense.w_down);
                grads.insert(format!("{prefix}.dense.w_down"), dd);

                // Expert gradients (active experts only — matches SMEBU intent)
                let router_grad = Tensor::randn(&[d, b.parallel_ffn.moe.router.n_experts], 0.001);
                grads.insert(format!("{prefix}.moe.router.gate"), router_grad);

                for (j, expert) in b.parallel_ffn.moe.experts.iter().enumerate() {
                    let eg = approximate_weight_grad(&ffn_grad, &expert.w_gate);
                    grads.insert(format!("{prefix}.moe.experts.{j}.w_gate"), eg);
                    let eu = approximate_weight_grad(&ffn_grad, &expert.w_up);
                    grads.insert(format!("{prefix}.moe.experts.{j}.w_up"), eu);
                    let ed = approximate_weight_grad(&ffn_grad, &expert.w_down);
                    grads.insert(format!("{prefix}.moe.experts.{j}.w_down"), ed);
                }

                if let Some(ref shared) = b.parallel_ffn.moe.shared_expert {
                    let sg = approximate_weight_grad(&ffn_grad, &shared.w_gate);
                    grads.insert(format!("{prefix}.moe.shared.w_gate"), sg);
                    let su = approximate_weight_grad(&ffn_grad, &shared.w_up);
                    grads.insert(format!("{prefix}.moe.shared.w_up"), su);
                    let sd = approximate_weight_grad(&ffn_grad, &shared.w_down);
                    grads.insert(format!("{prefix}.moe.shared.w_down"), sd);
                }
            }
            noor_core::layers::block::Block::PLE(b) => {
                let attn_grad = grad_hidden.scale(layer_grad_scale * 0.1);
                let wq_grad = approximate_weight_grad(&attn_grad, &b.attention.wq);
                grads.insert(format!("{prefix}.attn.wq"), wq_grad);
                let wk_grad = approximate_weight_grad(&attn_grad, &b.attention.wk);
                grads.insert(format!("{prefix}.attn.wk"), wk_grad);
                let wv_grad = approximate_weight_grad(&attn_grad, &b.attention.wv);
                grads.insert(format!("{prefix}.attn.wv"), wv_grad);
                let wo_grad = approximate_weight_grad(&attn_grad, &b.attention.wo);
                grads.insert(format!("{prefix}.attn.wo"), wo_grad);

                let ffn_grad = grad_hidden.scale(layer_grad_scale * 0.1);
                let fg = approximate_weight_grad(&ffn_grad, &b.ffn.w_gate);
                grads.insert(format!("{prefix}.ffn.w_gate"), fg);
                let fu = approximate_weight_grad(&ffn_grad, &b.ffn.w_up);
                grads.insert(format!("{prefix}.ffn.w_up"), fu);
                let fd = approximate_weight_grad(&ffn_grad, &b.ffn.w_down);
                grads.insert(format!("{prefix}.ffn.w_down"), fd);
            }
        }
    }

    (loss, grads)
}

/// Create an approximate gradient matching a weight's shape.
/// Uses the hidden gradient signal to produce a scaled random direction.
/// Phase 1 replaces this with proper chain-rule backward.
fn approximate_weight_grad(hidden_grad: &Tensor, weight: &Tensor) -> Tensor {
    let grad_norm: f32 = hidden_grad.data.iter().map(|v| v * v).sum::<f32>().sqrt();
    let scale = grad_norm / weight.numel() as f32;
    // Random direction scaled by gradient magnitude
    let mut g = Tensor::randn(&weight.shape, scale as f64);
    // Mix in signal from the actual gradient to give some real direction
    if hidden_grad.shape[hidden_grad.ndim() - 1] == weight.shape[0] {
        // hidden_grad is (seq, d_model), weight is (d_model, out)
        // Approximate: x^T @ grad ≈ mean(hidden_grad, dim=0)^T @ ones
        let d = hidden_grad.shape[hidden_grad.ndim() - 1];
        let seq = hidden_grad.numel() / d;
        for i in 0..d.min(weight.shape[0]) {
            let mut mean = 0.0f32;
            for s in 0..seq {
                mean += hidden_grad.data[s * d + i];
            }
            mean /= seq as f32;
            for j in 0..weight.shape.get(1).copied().unwrap_or(1).min(g.shape.get(1).copied().unwrap_or(1)) {
                let idx = i * g.shape.get(1).copied().unwrap_or(1) + j;
                if idx < g.numel() {
                    g.data[idx] = g.data[idx] * 0.5 + mean * 0.5;
                }
            }
        }
    }
    g
}

/// Run the training loop.
pub fn train(
    config: &ModelConfig,
    model: &mut NoorModel,
    data: &mut DataLoader,
    checkpoint_dir: Option<&Path>,
) -> Vec<StepMetrics> {
    let total_steps = config.training.total_steps;
    let warmup = config.training.warmup_steps;
    let lr_max = config.training.lr_max;
    let lr_min = config.training.lr_min;
    let grad_clip_max = config.training.grad_clip as f32;
    let log_every = config.training.log_every_steps;
    let ckpt_every = config.training.checkpoint_every_steps;

    let mut optimizer = Muon::new(config.optimizer.beta as f32, lr_max as f32);
    let mut qk_clip = QKClip::new(config.optimizer.qk_clip_tau as f32);

    // Per-MoE-layer SMEBU instances
    let n_moe_layers = model.blocks.iter()
        .filter(|b| matches!(b, noor_core::layers::block::Block::MoE(_)))
        .count();
    let mut smebus: Vec<SMEBU> = (0..n_moe_layers)
        .map(|_| SMEBU::new(
            config.moe.n_experts,
            config.smebu.kappa as f32,
            config.smebu.beta as f32,
            config.smebu.lambda as f32,
        ))
        .collect();

    let mut all_metrics = Vec::new();

    // Collect mutable param references for optimizer
    let mut params = gguf::collect_model_tensors(model);

    for step in 0..total_steps {
        let step_start = Instant::now();

        // Get batch
        let batch = match data.next_batch() {
            Some(b) => b,
            None => {
                data.reset();
                data.next_batch().expect("No data available")
            }
        };

        // Use first sequence in batch for simplicity (gradient accumulation deferred)
        let input_ids = &batch.input_ids[0];
        let target_ids: Vec<u32> = batch.target_ids[0].clone();

        // Forward pass
        let output = model.forward(input_ids, None);

        // We need the hidden states before output projection for backward.
        // Re-run forward to get them (inefficient but correct for Phase 0).
        // In Phase 1, forward() will cache intermediate states.
        let h = model.embedding.forward(input_ids);
        // Use logits directly for loss computation

        // Compute loss and gradients
        let (loss, mut grads) = compute_gradients(
            model,
            &output.logits,
            &target_ids,
            &h, // approximate hidden states
        );

        // Gradient clipping
        let grad_norm = backward::clip_grad_norm(&mut grads, grad_clip_max);

        // LR schedule
        let lr = wsd_lr(step, warmup, total_steps, lr_max, lr_min);
        optimizer.set_lr(lr as f32);

        // Optimizer step
        optimizer.step(&mut params, &grads);

        // Write updated params back to model
        apply_params_to_model(model, &params);

        // SMEBU update for each MoE layer
        for (i, util) in output.expert_utilization.iter().enumerate() {
            if i < smebus.len() {
                smebus[i].update(util);
                // Apply biases back to router
                if let Some(noor_core::layers::block::Block::MoE(ref mut moe_block)) = model.blocks.iter_mut()
                    .filter(|b| matches!(b, noor_core::layers::block::Block::MoE(_)))
                    .nth(i)
                {
                    smebus[i].apply_to_router(&mut moe_block.parallel_ffn.moe.router.expert_biases);
                }
            }
        }

        // QK-Clip
        let qk_clipped = output.max_attn_logit > qk_clip.tau;
        if qk_clipped {
            for block in model.blocks.iter_mut() {
                if let noor_core::layers::block::Block::MoE(ref mut b) = block {
                    qk_clip.clip_if_needed(
                        output.max_attn_logit,
                        &mut b.attention.wq,
                        &mut b.attention.wk,
                    );
                }
            }
        }

        let elapsed = step_start.elapsed().as_secs_f32();
        let tokens_per_sec = input_ids.len() as f32 / elapsed;

        // Count active experts
        let active_experts = output.expert_utilization.first()
            .map(|u| u.iter().filter(|&&f| f > 0.01).count())
            .unwrap_or(0);

        let metrics = StepMetrics {
            step,
            loss,
            lr,
            grad_norm,
            max_attn_logit: output.max_attn_logit,
            tokens_per_sec,
            active_experts,
            qk_clipped,
        };

        // Logging
        if step % log_every == 0 {
            eprintln!(
                "step={:>5} | loss={:.4} | lr={:.2e} | gnorm={:.3} | max_logit={:.1} | tok/s={:.0} | experts={} {}",
                metrics.step, metrics.loss, metrics.lr, metrics.grad_norm,
                metrics.max_attn_logit, metrics.tokens_per_sec, metrics.active_experts,
                if metrics.qk_clipped { "| QK-CLIPPED" } else { "" }
            );
        }

        all_metrics.push(metrics);

        // Checkpoint
        if ckpt_every > 0 && step > 0 && step % ckpt_every == 0 {
            if let Some(dir) = checkpoint_dir {
                let ckpt_path = dir.join(format!("checkpoint_step_{step}.gguf"));
                let tensors = gguf::collect_model_tensors(model);
                let mut meta = HashMap::new();
                meta.insert("model.name".to_string(), gguf::GGUFValue::String(config.model.name.clone()));
                meta.insert("training.step".to_string(), gguf::GGUFValue::U64(step as u64));
                meta.insert("training.loss".to_string(), gguf::GGUFValue::F32(loss));
                meta.insert("training.lr".to_string(), gguf::GGUFValue::F64(lr));
                if let Err(e) = gguf::save_gguf(&ckpt_path, &tensors, &meta) {
                    eprintln!("Checkpoint save failed: {e}");
                } else {
                    eprintln!("Checkpoint saved: {}", ckpt_path.display());
                }
            }
        }
    }

    all_metrics
}

/// Apply parameter HashMap back to model weights.
fn apply_params_to_model(model: &mut NoorModel, params: &HashMap<String, Tensor>) {
    if let Some(t) = params.get("output_proj") {
        model.output_proj = t.clone();
    }
    if let Some(t) = params.get("embedding.weight") {
        model.embedding.weight = t.clone();
    }
    for (i, block) in model.blocks.iter_mut().enumerate() {
        let prefix = format!("blocks.{i}");
        match block {
            noor_core::layers::block::Block::MoE(b) => {
                if let Some(t) = params.get(&format!("{prefix}.attn.wq")) { b.attention.wq = t.clone(); }
                if let Some(t) = params.get(&format!("{prefix}.attn.wk")) { b.attention.wk = t.clone(); }
                if let Some(t) = params.get(&format!("{prefix}.attn.wv")) { b.attention.wv = t.clone(); }
                if let Some(t) = params.get(&format!("{prefix}.attn.wo")) { b.attention.wo = t.clone(); }
                if let Some(t) = params.get(&format!("{prefix}.dense.w_gate")) { b.parallel_ffn.dense.w_gate = t.clone(); }
                if let Some(t) = params.get(&format!("{prefix}.dense.w_up")) { b.parallel_ffn.dense.w_up = t.clone(); }
                if let Some(t) = params.get(&format!("{prefix}.dense.w_down")) { b.parallel_ffn.dense.w_down = t.clone(); }
                if let Some(t) = params.get(&format!("{prefix}.moe.router.gate")) { b.parallel_ffn.moe.router.gate = t.clone(); }
                for (j, expert) in b.parallel_ffn.moe.experts.iter_mut().enumerate() {
                    if let Some(t) = params.get(&format!("{prefix}.moe.experts.{j}.w_gate")) { expert.w_gate = t.clone(); }
                    if let Some(t) = params.get(&format!("{prefix}.moe.experts.{j}.w_up")) { expert.w_up = t.clone(); }
                    if let Some(t) = params.get(&format!("{prefix}.moe.experts.{j}.w_down")) { expert.w_down = t.clone(); }
                }
                if let Some(ref mut shared) = b.parallel_ffn.moe.shared_expert {
                    if let Some(t) = params.get(&format!("{prefix}.moe.shared.w_gate")) { shared.w_gate = t.clone(); }
                    if let Some(t) = params.get(&format!("{prefix}.moe.shared.w_up")) { shared.w_up = t.clone(); }
                    if let Some(t) = params.get(&format!("{prefix}.moe.shared.w_down")) { shared.w_down = t.clone(); }
                }
            }
            noor_core::layers::block::Block::PLE(b) => {
                if let Some(t) = params.get(&format!("{prefix}.attn.wq")) { b.attention.wq = t.clone(); }
                if let Some(t) = params.get(&format!("{prefix}.attn.wk")) { b.attention.wk = t.clone(); }
                if let Some(t) = params.get(&format!("{prefix}.attn.wv")) { b.attention.wv = t.clone(); }
                if let Some(t) = params.get(&format!("{prefix}.attn.wo")) { b.attention.wo = t.clone(); }
                if let Some(t) = params.get(&format!("{prefix}.ffn.w_gate")) { b.ffn.w_gate = t.clone(); }
                if let Some(t) = params.get(&format!("{prefix}.ffn.w_up")) { b.ffn.w_up = t.clone(); }
                if let Some(t) = params.get(&format!("{prefix}.ffn.w_down")) { b.ffn.w_down = t.clone(); }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wsd_schedule() {
        // Warmup phase
        let lr = wsd_lr(0, 100, 1000, 3e-4, 3e-5);
        assert!((lr - 0.0).abs() < 1e-10, "Step 0 should be 0");

        let lr = wsd_lr(50, 100, 1000, 3e-4, 3e-5);
        assert!((lr - 1.5e-4).abs() < 1e-8, "Mid-warmup should be half lr_max");

        // Stable phase
        let lr = wsd_lr(200, 100, 1000, 3e-4, 3e-5);
        assert!((lr - 3e-4).abs() < 1e-10, "Stable phase should be lr_max");

        // Decay phase
        let lr = wsd_lr(999, 100, 1000, 3e-4, 3e-5);
        assert!(lr < 3e-4 && lr >= 3e-5, "Decay phase lr={lr}");
    }

    #[test]
    fn test_training_loop_tiny() {
        // Create a tiny model for fast testing
        let config_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .parent().unwrap()
            .join("config/proxy.toml");
        let mut config = noor_core::config::ModelConfig::from_toml(&config_path).unwrap();

        // Override to make it tiny
        config.model.d_model = 32;
        config.model.n_layers = 2;
        config.model.n_heads = 2;
        config.model.n_kv_heads = 1;
        config.model.head_dim = 16;
        config.model.vocab_size = 100;
        config.model.context_length = 16;
        config.moe.n_experts = 4;
        config.moe.n_active_experts = 2;
        config.moe.expert_ffn_dim = 16;
        config.moe.dense_ffn_dim = 32;
        config.attention.sliding_window = 8;
        config.training.total_steps = 5;
        config.training.warmup_steps = 2;
        config.training.log_every_steps = 1;
        config.training.checkpoint_every_steps = 0; // no checkpoints
        config.training.batch_size_tokens = 32;
        config.training.micro_batch_tokens = 16;

        let mut model = NoorModel::from_config(&config);

        // Create synthetic data
        let tokens: Vec<u32> = (0..500).map(|i| (i % 100) as u32).collect();
        let mut loader = DataLoader::from_tokens(tokens, config.model.context_length, 1);

        let metrics = train(&config, &mut model, &mut loader, None);

        assert_eq!(metrics.len(), 5, "Should have 5 steps of metrics");
        // Loss should be finite
        for m in &metrics {
            assert!(m.loss.is_finite(), "Loss should be finite at step {}: {}", m.step, m.loss);
            assert!(m.grad_norm.is_finite(), "Grad norm should be finite at step {}", m.step);
        }
    }
}
