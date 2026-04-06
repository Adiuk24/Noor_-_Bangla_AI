use clap::{Parser, Subcommand};
use std::path::Path;

#[derive(Parser)]
#[command(name = "noor", about = "Noor — sparse MoE language model")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a model
    Train {
        /// Path to model config TOML
        #[arg(short, long)]
        config: String,
        /// Path to data shard directory
        #[arg(short, long)]
        data: String,
        /// Resume from GGUF checkpoint
        #[arg(short, long)]
        resume: Option<String>,
        /// Number of training steps (overrides config)
        #[arg(short, long)]
        steps: Option<usize>,
        /// Directory for checkpoints
        #[arg(long)]
        checkpoint_dir: Option<String>,
    },
    /// Run inference
    Run {
        /// Path to GGUF model
        #[arg(short, long)]
        model: String,
        /// Input prompt
        #[arg(short, long)]
        prompt: String,
        /// Max tokens to generate
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        /// Temperature (0 = greedy)
        #[arg(long, default_value = "0.0")]
        temperature: f32,
    },
    /// Evaluate on held-out data
    Eval {
        /// Path to model config TOML
        #[arg(short, long)]
        config: String,
        /// Path to eval data (binary shard or raw tokens)
        #[arg(short, long)]
        data: String,
    },
    /// Benchmark speed
    Bench {
        /// Path to model config TOML
        #[arg(short, long)]
        config: String,
    },
    /// Preprocess raw text into tokenized binary shards
    Preprocess {
        /// Input text file
        #[arg(short, long)]
        input: String,
        /// Output directory for shards
        #[arg(short, long)]
        output: String,
        /// Vocab size for byte-level tokenizer
        #[arg(long, default_value = "64000")]
        vocab_size: usize,
        /// Tokens per shard
        #[arg(long, default_value = "100000")]
        shard_size: usize,
        /// Context length for sequence splitting
        #[arg(long, default_value = "2048")]
        context_length: usize,
    },
    /// Generate synthetic training data (pre-tokenized shards)
    GenData {
        /// Output directory for shards
        #[arg(short, long)]
        output: String,
        /// Vocab size
        #[arg(long, default_value = "64000")]
        vocab_size: usize,
        /// Total number of tokens to generate
        #[arg(long, default_value = "100000")]
        num_tokens: usize,
        /// Tokens per shard
        #[arg(long, default_value = "50000")]
        shard_size: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train { config, data, resume, steps, checkpoint_dir } => {
            cmd_train(&config, &data, resume.as_deref(), steps, checkpoint_dir.as_deref());
        }
        Commands::Run { model, prompt, max_tokens, temperature } => {
            cmd_run(&model, &prompt, max_tokens, temperature);
        }
        Commands::Eval { config, data } => {
            cmd_eval(&config, &data);
        }
        Commands::Preprocess { input, output, vocab_size, shard_size, context_length } => {
            cmd_preprocess(&input, &output, vocab_size, shard_size, context_length);
        }
        Commands::Bench { config } => {
            cmd_bench(&config);
        }
        Commands::GenData { output, vocab_size, num_tokens, shard_size } => {
            cmd_gen_data(&output, vocab_size, num_tokens, shard_size);
        }
    }
}

fn cmd_train(config_path: &str, data_dir: &str, _resume: Option<&str>, steps_override: Option<usize>, checkpoint_dir: Option<&str>) {
    eprintln!("Loading config: {config_path}");
    let mut config = noor_core::config::ModelConfig::from_toml(Path::new(config_path))
        .expect("Failed to load config");

    if let Some(s) = steps_override {
        config.training.total_steps = s;
    }

    eprintln!("Creating model: {} (d={}, layers={}, experts={})",
        config.model.name, config.model.d_model, config.model.n_layers, config.moe.n_experts);
    let mut model = noor_core::model::NoorModel::from_config(&config);
    eprintln!("Total params: {}", model.param_count_total());

    eprintln!("Loading data from: {data_dir}");
    let mut loader = noor_train::data::DataLoader::from_shard_dir(
        Path::new(data_dir),
        config.model.context_length,
        1, // batch_size = 1 for Phase 0 (gradient accumulation later)
    ).expect("Failed to load data shards");
    eprintln!("Total tokens: {}", loader.total_tokens());

    let ckpt_dir = checkpoint_dir.map(Path::new);
    if let Some(dir) = ckpt_dir {
        std::fs::create_dir_all(dir).expect("Failed to create checkpoint dir");
    }

    eprintln!("Starting training for {} steps...", config.training.total_steps);
    let metrics = noor_train::training_loop::train(&config, &mut model, &mut loader, ckpt_dir);

    // Summary
    if let Some(last) = metrics.last() {
        eprintln!("\nTraining complete.");
        eprintln!("  Final loss: {:.4}", last.loss);
        eprintln!("  Final LR: {:.2e}", last.lr);
        eprintln!("  Active experts: {}", last.active_experts);
    }

    // Save final checkpoint
    let final_path = checkpoint_dir
        .map(|d| Path::new(d).join("final.gguf"))
        .unwrap_or_else(|| Path::new("noor_final.gguf").to_path_buf());
    let tensors = noor_core::gguf::collect_model_tensors(&model);
    let mut meta = std::collections::HashMap::new();
    meta.insert("model.name".to_string(), noor_core::gguf::GGUFValue::String(config.model.name.clone()));
    meta.insert("training.steps".to_string(), noor_core::gguf::GGUFValue::U64(config.training.total_steps as u64));
    if let Some(last) = metrics.last() {
        meta.insert("training.final_loss".to_string(), noor_core::gguf::GGUFValue::F32(last.loss));
    }
    noor_core::gguf::save_gguf(&final_path, &tensors, &meta).expect("Failed to save final checkpoint");
    eprintln!("Final checkpoint saved: {}", final_path.display());
}

fn cmd_run(config_path: &str, prompt: &str, max_tokens: usize, temperature: f32) {
    eprintln!("Loading config: {config_path}");
    let config = noor_core::config::ModelConfig::from_toml(Path::new(config_path))
        .expect("Failed to load config");
    let mut model = noor_core::model::NoorModel::from_config(&config);

    // Encode prompt as bytes
    let tokenizer = noor_core::tokenizer::NoorTokenizer::byte_level(config.model.vocab_size);
    let prompt_ids = tokenizer.encode(prompt);

    eprintln!("Generating {} tokens...", max_tokens);
    let generated = noor_train::eval::generate(&mut model, &prompt_ids, max_tokens, temperature);
    let text = tokenizer.decode(&generated);
    println!("{text}");
}

fn cmd_eval(config_path: &str, data_path: &str) {
    eprintln!("Loading config: {config_path}");
    let config = noor_core::config::ModelConfig::from_toml(Path::new(config_path))
        .expect("Failed to load config");
    let mut model = noor_core::model::NoorModel::from_config(&config);

    // Load eval data
    let shard = noor_train::data::DataShard::open(Path::new(data_path))
        .expect("Failed to open eval data");
    let tokens = shard.read_all_tokens();
    eprintln!("Eval tokens: {}", tokens.len());

    let ppl = noor_train::eval::eval_perplexity(&mut model, &tokens);
    eprintln!("Perplexity: {ppl:.2}");
}

fn cmd_bench(config_path: &str) {
    eprintln!("Loading config: {config_path}");
    let config = noor_core::config::ModelConfig::from_toml(Path::new(config_path))
        .expect("Failed to load config");
    let mut model = noor_core::model::NoorModel::from_config(&config);

    let prompt: Vec<u32> = (0..32).map(|i| (i % config.model.vocab_size) as u32).collect();

    eprintln!("Benchmarking forward pass...");
    let start = std::time::Instant::now();
    let n_runs = 5;
    for _ in 0..n_runs {
        let _ = model.forward(&prompt, None);
    }
    let elapsed = start.elapsed().as_secs_f32();
    let avg_ms = elapsed * 1000.0 / n_runs as f32;
    let tok_per_sec = (prompt.len() * n_runs) as f32 / elapsed;

    eprintln!("  Avg forward: {avg_ms:.1} ms");
    eprintln!("  Throughput: {tok_per_sec:.0} tok/s");
    eprintln!("  Model params: {}", model.param_count_total());
}

fn cmd_gen_data(output_dir: &str, vocab_size: usize, num_tokens: usize, shard_size: usize) {
    std::fs::create_dir_all(output_dir).expect("Failed to create output dir");

    eprintln!("Generating {} tokens with vocab_size={} into {}", num_tokens, vocab_size, output_dir);

    let mut remaining = num_tokens;
    let mut shard_idx = 0;

    while remaining > 0 {
        let chunk = remaining.min(shard_size);
        // Generate pseudo-random tokens with some structure (repeated patterns)
        let tokens: Vec<u32> = (0..chunk).map(|i| {
            // Mix of patterns: sequential, repeated, random-ish
            let pattern = i / 100;
            match pattern % 4 {
                0 => (i % vocab_size) as u32,                    // sequential
                1 => ((i * 7 + 13) % vocab_size) as u32,        // strided
                2 => ((i / 10) % vocab_size) as u32,             // repeated groups
                _ => ((i.wrapping_mul(2654435761)) % vocab_size) as u32, // hash
            }
        }).collect();

        let sequences = vec![tokens];
        let path = Path::new(output_dir).join(format!("shard_{shard_idx:04}.bin"));
        noor_train::data::write_shard(&path, &sequences).expect("Failed to write shard");
        eprintln!("  Written: {} ({chunk} tokens)", path.display());

        remaining -= chunk;
        shard_idx += 1;
    }

    eprintln!("Done. {} shards, {} total tokens.", shard_idx, num_tokens);
}

fn cmd_preprocess(input_path: &str, output_dir: &str, vocab_size: usize, shard_size: usize, context_length: usize) {
    std::fs::create_dir_all(output_dir).expect("Failed to create output dir");

    eprintln!("Reading: {input_path}");
    let text = std::fs::read_to_string(input_path).expect("Failed to read input file");
    eprintln!("Text size: {} bytes", text.len());

    // Byte-level tokenization
    let tokenizer = noor_core::tokenizer::NoorTokenizer::byte_level(vocab_size);
    let all_tokens = tokenizer.encode(&text);
    eprintln!("Total tokens: {} (byte-level)", all_tokens.len());

    // Split into sequences of context_length + 1 (for input/target pairs)
    let seq_len = context_length + 1;
    let n_sequences = all_tokens.len() / seq_len;
    eprintln!("Sequences (ctx={}): {}", context_length, n_sequences);

    let mut shard_idx = 0;
    let mut seq_start = 0;
    let seqs_per_shard = shard_size / seq_len;

    while seq_start < n_sequences {
        let seq_end = (seq_start + seqs_per_shard).min(n_sequences);
        let mut sequences = Vec::new();

        for s in seq_start..seq_end {
            let offset = s * seq_len;
            let seq = all_tokens[offset..offset + seq_len].to_vec();
            sequences.push(seq);
        }

        let shard_tokens: usize = sequences.iter().map(|s| s.len()).sum();
        let path = Path::new(output_dir).join(format!("shard_{shard_idx:04}.bin"));
        noor_train::data::write_shard(&path, &sequences).expect("Failed to write shard");
        eprintln!("  {} — {} sequences, {} tokens", path.display(), sequences.len(), shard_tokens);

        seq_start = seq_end;
        shard_idx += 1;
    }

    // Save vocab for reference
    let vocab_path = Path::new(output_dir).join("vocab.txt");
    tokenizer.save_vocab(&vocab_path).expect("Failed to save vocab");

    eprintln!("Done. {} shards from {} sequences.", shard_idx, n_sequences);
}
