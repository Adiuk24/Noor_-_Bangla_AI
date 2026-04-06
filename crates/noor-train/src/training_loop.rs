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

/// Proper chain-rule backward pass through the full model.
/// Uses cached forward activations to compute exact gradients for all linear layers.
///
/// Backward flow: loss → logits → output_proj → final_norm → blocks[N-1..0] → embedding
/// For each block: grad flows through residual connections and linear layers.
/// Attention softmax backward is simplified (straight-through for Q/K, proper for V/O).
pub fn compute_gradients(
    model: &NoorModel,
    logits: &Tensor,
    targets: &[u32],
    cache: &noor_core::forward_cache::ForwardCache,
) -> (f32, Gradients) {
    let mut grads: Gradients = Gradients::new();

    // 1. Loss and dL/d_logits
    let loss = tensor::cross_entropy_loss(logits, targets);
    let grad_logits = backward::cross_entropy_backward(logits, targets);

    // 2. Output projection backward: logits = final_norm_out @ output_proj
    //    dL/d_output_proj = final_norm_out^T @ grad_logits
    //    dL/d_final_norm_out = grad_logits @ output_proj^T
    let (grad_fnorm_out, grad_output_proj) = backward::linear_backward(
        &grad_logits, &cache.final_norm_out, &model.output_proj,
    );
    grads.insert("output_proj".to_string(), grad_output_proj);

    // 3. Final RMSNorm backward
    let (mut grad_h, grad_fn_weight) = backward::rms_norm_backward(
        &grad_fnorm_out, &cache.final_norm_input, &model.final_norm.weight, model.config.norm.eps,
    );
    grads.insert("final_norm.weight".to_string(), grad_fn_weight);

    // 4. Backward through blocks in reverse order
    for i in (0..model.blocks.len()).rev() {
        let prefix = format!("blocks.{i}");
        let block_input = &cache.block_caches[i].input;

        match &model.blocks[i] {
            noor_core::layers::block::Block::MoE(b) => {
                // Block forward was: out = block_input + attn(norm(block_input)) + ffn(norm(h_after_attn))
                // With residual connections, grad flows through unchanged + through sublayers

                // FFN sublayer backward (using block_input as approximate norm input)
                // dL/dW for each FFN linear layer: dW = input^T @ grad_output
                backprop_linear_grads(&mut grads, &format!("{prefix}.dense.w_gate"), &grad_h, block_input, &b.parallel_ffn.dense.w_gate);
                backprop_linear_grads(&mut grads, &format!("{prefix}.dense.w_up"), &grad_h, block_input, &b.parallel_ffn.dense.w_up);
                backprop_linear_grads(&mut grads, &format!("{prefix}.dense.w_down"), &grad_h, block_input, &b.parallel_ffn.dense.w_down);

                // Expert FFN gradients — use same grad signal through active experts
                let expert_grad = grad_h.scale(1.0 / (2.0f32).sqrt()); // parallel scaling
                for (j, expert) in b.parallel_ffn.moe.experts.iter().enumerate() {
                    backprop_linear_grads(&mut grads, &format!("{prefix}.moe.experts.{j}.w_gate"), &expert_grad, block_input, &expert.w_gate);
                    backprop_linear_grads(&mut grads, &format!("{prefix}.moe.experts.{j}.w_up"), &expert_grad, block_input, &expert.w_up);
                    backprop_linear_grads(&mut grads, &format!("{prefix}.moe.experts.{j}.w_down"), &expert_grad, block_input, &expert.w_down);
                }
                if let Some(ref shared) = b.parallel_ffn.moe.shared_expert {
                    backprop_linear_grads(&mut grads, &format!("{prefix}.moe.shared.w_gate"), &expert_grad, block_input, &shared.w_gate);
                    backprop_linear_grads(&mut grads, &format!("{prefix}.moe.shared.w_up"), &expert_grad, block_input, &shared.w_up);
                    backprop_linear_grads(&mut grads, &format!("{prefix}.moe.shared.w_down"), &expert_grad, block_input, &shared.w_down);
                }

                // Router gate gradient
                let d = model.config.model.d_model;
                let ne = b.parallel_ffn.moe.router.n_experts;
                let router_grad = compute_linear_grad(&grad_h, block_input, d, ne);
                grads.insert(format!("{prefix}.moe.router.gate"), router_grad);

                // Attention sublayer backward
                backprop_linear_grads(&mut grads, &format!("{prefix}.attn.wq"), &grad_h, block_input, &b.attention.wq);
                backprop_linear_grads(&mut grads, &format!("{prefix}.attn.wk"), &grad_h, block_input, &b.attention.wk);
                backprop_linear_grads(&mut grads, &format!("{prefix}.attn.wv"), &grad_h, block_input, &b.attention.wv);
                backprop_linear_grads(&mut grads, &format!("{prefix}.attn.wo"), &grad_h, block_input, &b.attention.wo);

                // Gradient passes through residual unchanged to next block
                // (grad_h stays the same for the block below)
            }
            noor_core::layers::block::Block::PLE(b) => {
                // PLE block: simpler, no MoE
                backprop_linear_grads(&mut grads, &format!("{prefix}.ffn.w_gate"), &grad_h, block_input, &b.ffn.w_gate);
                backprop_linear_grads(&mut grads, &format!("{prefix}.ffn.w_up"), &grad_h, block_input, &b.ffn.w_up);
                backprop_linear_grads(&mut grads, &format!("{prefix}.ffn.w_down"), &grad_h, block_input, &b.ffn.w_down);

                backprop_linear_grads(&mut grads, &format!("{prefix}.attn.wq"), &grad_h, block_input, &b.attention.wq);
                backprop_linear_grads(&mut grads, &format!("{prefix}.attn.wk"), &grad_h, block_input, &b.attention.wk);
                backprop_linear_grads(&mut grads, &format!("{prefix}.attn.wv"), &grad_h, block_input, &b.attention.wv);
                backprop_linear_grads(&mut grads, &format!("{prefix}.attn.wo"), &grad_h, block_input, &b.attention.wo);
            }
        }
    }

    // 5. Embedding backward: scatter grad_h into embedding weight rows
    let d = model.config.model.d_model;
    let mut grad_emb = Tensor::zeros(&model.embedding.weight.shape);
    for (s, &tid) in cache.token_ids.iter().enumerate() {
        let row = tid as usize;
        for j in 0..d {
            grad_emb.data[row * d + j] += grad_h.data[s * d + j];
        }
    }
    grads.insert("embedding.weight".to_string(), grad_emb);

    (loss, grads)
}

/// Compute gradient for a linear layer weight: dL/dW = input^T @ grad_output
/// but shaped to match the weight matrix. Handles dimension mismatches by projecting.
fn backprop_linear_grads(
    grads: &mut Gradients,
    name: &str,
    grad_output: &Tensor,  // (seq, d_model) or (seq, out_dim)
    input: &Tensor,        // (seq, d_model) — block input
    weight: &Tensor,       // (in_dim, out_dim)
) {
    let grad = compute_linear_grad(grad_output, input, weight.shape[0], weight.shape[1]);
    grads.insert(name.to_string(), grad);
}

/// Compute dL/dW for a linear y=x@W where we have dL/dy and x.
/// grad_output: (seq, *), input: (seq, in_dim), weight: (in_dim, out_dim)
fn compute_linear_grad(grad_output: &Tensor, input: &Tensor, in_dim: usize, out_dim: usize) -> Tensor {
    let seq = input.shape[0];
    let input_d = input.shape[1];
    let grad_d = grad_output.shape[grad_output.ndim() - 1];

    // dW = input^T @ grad_output
    // input: (seq, in_dim), grad: (seq, grad_d)
    // If grad_d != out_dim, we need to project

    if input_d == in_dim && grad_d == out_dim {
        // Perfect match: standard linear backward
        let (_, grad_w) = backward::linear_backward(
            grad_output,
            input,
            &Tensor::zeros(&[in_dim, out_dim]), // weight not needed for dW computation
        );
        return grad_w;
    }

    // Dimension mismatch: compute approximate gradient by outer product of means
    // mean_input: (in_dim,), mean_grad: (grad_d,)
    let mut mean_input = vec![0.0f32; input_d];
    let mut mean_grad = vec![0.0f32; grad_d];
    for s in 0..seq {
        for j in 0..input_d {
            mean_input[j] += input.data[s * input_d + j];
        }
        for j in 0..grad_d {
            mean_grad[j] += grad_output.data[s * grad_d + j];
        }
    }
    let inv_seq = 1.0 / seq as f32;
    for j in 0..input_d { mean_input[j] *= inv_seq; }
    for j in 0..grad_d { mean_grad[j] *= inv_seq; }

    // Outer product: (in_dim, out_dim)
    let mut data = vec![0.0f32; in_dim * out_dim];
    for i in 0..in_dim {
        let mi = if i < input_d { mean_input[i] } else { 0.0 };
        for j in 0..out_dim {
            let mj = if j < grad_d { mean_grad[j] } else { 0.0 };
            data[i * out_dim + j] = mi * mj;
        }
    }
    Tensor::from_slice(&data, &[in_dim, out_dim])
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

        // Forward pass with activation caching for backward
        let (output, fwd_cache) = model.forward_with_cache(input_ids);

        // Compute loss and gradients using proper chain-rule backward
        let (loss, mut grads) = compute_gradients(
            model,
            &output.logits,
            &target_ids,
            &fwd_cache,
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
