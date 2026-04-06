//! Training loop — Burn autograd handles backward pass automatically.
//!
//! Forward → loss → loss.backward() → grads → Muon step → log → checkpoint.
//! No manual gradient computation. No temporary tensor allocation.

use burn::prelude::*;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::record::CompactRecorder;
use std::path::Path;
use std::time::Instant;

use crate::config::NoorConfig;
use crate::model::NoorModel;
use crate::data::ShardDataset;

/// WSD (Warmup-Stable-Decay) learning rate schedule.
pub fn wsd_lr(step: usize, warmup: usize, total: usize, lr_max: f64, lr_min: f64) -> f64 {
    if step < warmup {
        lr_max * (step as f64 / warmup as f64)
    } else if step < total * 3 / 4 {
        lr_max
    } else {
        let decay_steps = total - total * 3 / 4;
        let progress = (step - total * 3 / 4) as f64 / decay_steps as f64;
        let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
        lr_min + (lr_max - lr_min) * cosine
    }
}

/// Run the full training loop.
pub fn train<B: burn::tensor::backend::AutodiffBackend>(
    config: &NoorConfig,
    device: &B::Device,
    data_dir: &Path,
    checkpoint_dir: Option<&Path>,
) {
    let total_steps = config.training.total_steps;
    let warmup = config.training.warmup_steps;
    let lr_max = config.training.lr_max;
    let lr_min = config.training.lr_min;
    let log_every = config.training.log_every_steps;
    let ckpt_every = config.training.checkpoint_every_steps;
    let batch_size = config.batch_size().max(1);

    eprintln!("=== Noor Training (Burn backend) ===");
    eprintln!("  Model: {}", config.model.name);
    eprintln!("  d_model: {}, layers: {}, heads: {}", config.model.d_model, config.model.n_layers, config.model.n_heads);
    if config.ple.enabled {
        eprintln!("  PLE: dim={} (no MoE)", config.ple.ple_dim);
    } else {
        eprintln!("  MoE: {} experts, {} active", config.moe.n_experts, config.moe.n_active_experts);
    }
    eprintln!("  Context: {}, Batch: {} seqs ({} tokens)", config.model.context_length, batch_size, config.training.batch_size_tokens);
    eprintln!("  Steps: {}, LR: {} → {}", total_steps, lr_max, lr_min);

    // Init model on device (GPU if CUDA backend)
    let mut model = NoorModel::<B>::from_config(config, device);
    eprintln!("  Params: {}", model.param_count());

    // AdamW optimizer (Muon requires split 2D/1D params — will add later)
    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.01)
        .init();

    // Load data
    let mut dataset = ShardDataset::from_shard_dir(data_dir, config.model.context_length)
        .expect("Failed to load training shards");
    eprintln!("  Data: {} tokens, {} steps of data at batch={}", dataset.total_tokens(), dataset.total_tokens() / config.training.batch_size_tokens, batch_size);
    eprintln!();

    let train_start = Instant::now();

    for step in 0..total_steps {
        let step_start = Instant::now();

        // Get batch
        let (input_ids, target_ids) = dataset.next_batch::<B>(batch_size, device);

        // Forward + loss (autograd tracks everything)
        let loss = model.forward_loss(input_ids, target_ids);

        // Backward (ONE call — Burn computes all gradients automatically)
        let grads = loss.backward();

        // Extract gradients mapped to model parameters
        let grads = GradientsParams::from_grads(grads, &model);

        // LR schedule
        let lr = wsd_lr(step, warmup, total_steps, lr_max, lr_min);

        // Muon optimizer step (returns new model — ownership transfer)
        model = optim.step(lr, model, grads);

        let step_time = step_start.elapsed().as_secs_f32();

        // Logging
        if step % log_every == 0 {
            let loss_val: f32 = loss.into_scalar().elem();
            let tokens_per_sec = (batch_size * config.model.context_length) as f32 / step_time;
            let elapsed_min = train_start.elapsed().as_secs_f32() / 60.0;

            eprintln!(
                "step={:>5} | loss={:.4} | lr={:.2e} | {:.1}s/step | {:.0} tok/s | {:.1}min elapsed",
                step, loss_val, lr, step_time, tokens_per_sec, elapsed_min,
            );
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }

        // Checkpoint
        if ckpt_every > 0 && step > 0 && step % ckpt_every == 0 {
            if let Some(dir) = checkpoint_dir {
                let ckpt_path = dir.join(format!("noor_step_{step}"));
                std::fs::create_dir_all(dir).ok();
                model
                    .clone()
                    .save_file(&ckpt_path, &CompactRecorder::new())
                    .unwrap_or_else(|e| eprintln!("Checkpoint save failed: {e}"));
                eprintln!("  Checkpoint saved: {}", ckpt_path.display());
            }
        }
    }

    let total_time = train_start.elapsed().as_secs_f32() / 60.0;
    eprintln!();
    eprintln!("Training complete. {total_steps} steps in {total_time:.1} minutes.");

    // Save final model
    if let Some(dir) = checkpoint_dir {
        let final_path = dir.join("noor_final");
        std::fs::create_dir_all(dir).ok();
        model
            .save_file(&final_path, &CompactRecorder::new())
            .unwrap_or_else(|e| eprintln!("Final save failed: {e}"));
        eprintln!("Final model saved: {}", final_path.display());
    }
}
