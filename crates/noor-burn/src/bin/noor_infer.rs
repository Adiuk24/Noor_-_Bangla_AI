//! Noor inference CLI — load checkpoint, generate tokens.
//!
//! Usage:
//!   cargo run -p noor-burn --release --features cuda --bin noor-infer -- \
//!     --config config/edge_runpod.toml --checkpoint checkpoints/noor_step_1000 \
//!     --prompt "আমি বাংলায়" --max-tokens 50

use clap::Parser;
use std::path::PathBuf;

use burn::record::CompactRecorder;
use burn::prelude::*;

#[derive(Parser)]
#[command(name = "noor-infer", about = "Noor inference (Burn backend)")]
struct Args {
    #[arg(long)]
    config: PathBuf,

    #[arg(long)]
    checkpoint: PathBuf,

    /// Prompt token IDs (comma-separated) or raw text (needs tokenizer)
    #[arg(long, default_value = "1,100,200,300,400")]
    prompt: String,

    #[arg(long, default_value = "50")]
    max_tokens: usize,

    /// Temperature for sampling (0 = greedy)
    #[arg(long, default_value = "0.0")]
    temperature: f32,
}

/// Greedy or temperature-sampled generation.
fn generate<B: burn::tensor::backend::Backend>(
    model: &noor_burn::model::NoorModel<B>,
    prompt_ids: Vec<i32>,
    max_tokens: usize,
    temperature: f32,
    device: &B::Device,
) -> Vec<i32> {
    let mut tokens = prompt_ids.clone();

    for _ in 0..max_tokens {
        let input = Tensor::<B, 2, Int>::from_data(
            TensorData::new(tokens.clone(), [1, tokens.len()]),
            device,
        );

        let logits = model.forward(input); // [1, seq, vocab]
        let seq_len = tokens.len();
        let vocab_size = logits.dims()[2];
        let last_logits = logits.slice([0..1, (seq_len - 1)..seq_len]); // [1, 1, vocab]
        let last_logits = last_logits.reshape([1, vocab_size]); // [1, vocab]

        let next_token = if temperature <= 0.0 {
            // Greedy: argmax
            last_logits.argmax(1).into_scalar().elem::<i32>()
        } else {
            // Temperature sampling
            let scaled = last_logits / temperature;
            let probs = burn::tensor::activation::softmax(scaled, 1);
            // Simple top-1 from probs (proper sampling needs rand)
            probs.argmax(1).into_scalar().elem::<i32>()
        };

        tokens.push(next_token);

        // Stop on EOS (token 2)
        if next_token == 2 {
            break;
        }
    }

    tokens
}

fn run_inference<B: burn::tensor::backend::Backend>(
    config: &noor_burn::config::NoorConfig,
    device: &B::Device,
    args: &Args,
) {
    eprintln!("=== Noor Inference ===");
    eprintln!("  Model: {}", config.model.name);
    eprintln!("  Checkpoint: {:?}", args.checkpoint);

    // Init model structure
    let model = noor_burn::model::NoorModel::<B>::from_config(config, device);
    eprintln!("  Params: {}", model.param_count());

    // Load checkpoint weights
    let model = model
        .load_file(&args.checkpoint, &CompactRecorder::new(), device)
        .expect("Failed to load checkpoint");
    eprintln!("  Checkpoint loaded!");

    // Parse prompt tokens
    let prompt_ids: Vec<i32> = args
        .prompt
        .split(',')
        .filter_map(|s| s.trim().parse::<i32>().ok())
        .collect();

    if prompt_ids.is_empty() {
        eprintln!("  No valid token IDs in prompt. Using default [1, 100, 200]");
    }

    let ids = if prompt_ids.is_empty() {
        vec![1, 100, 200]
    } else {
        prompt_ids
    };

    eprintln!("  Prompt tokens: {:?}", ids);
    eprintln!("  Generating {} tokens...", args.max_tokens);
    eprintln!();

    let output = generate(&model, ids, args.max_tokens, args.temperature, device);

    println!("Generated token IDs:");
    println!("{:?}", output);

    // Check for patterns (not just random noise)
    let unique: std::collections::HashSet<i32> = output.iter().copied().collect();
    let total = output.len();
    let unique_count = unique.len();
    let repeat_ratio = 1.0 - (unique_count as f32 / total as f32);

    eprintln!();
    eprintln!("=== Generation Stats ===");
    eprintln!("  Total tokens: {}", total);
    eprintln!("  Unique tokens: {}", unique_count);
    eprintln!("  Repeat ratio: {:.1}%", repeat_ratio * 100.0);

    if unique_count < 5 && total > 10 {
        eprintln!("  WARNING: Very repetitive output — model may be undertrained");
    } else if unique_count > total / 2 {
        eprintln!("  OK: Diverse output — model is generating varied tokens");
    }
}

fn main() {
    let args = Args::parse();

    let config = noor_burn::config::NoorConfig::from_toml(&args.config)
        .expect("Failed to load config");

    #[cfg(feature = "cuda")]
    {
        use burn::backend::{Cuda, Autodiff};
        type B = Autodiff<Cuda>;
        let device = burn::backend::cuda::CudaDevice::default();
        run_inference::<B>(&config, &device, &args);
    }

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::{Wgpu, Autodiff};
        type B = Autodiff<Wgpu>;
        let device = burn::backend::wgpu::WgpuDevice::default();
        run_inference::<B>(&config, &device, &args);
    }

    #[cfg(feature = "ndarray")]
    {
        use burn::backend::{NdArray, Autodiff};
        type B = Autodiff<NdArray>;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        run_inference::<B>(&config, &device, &args);
    }
}
