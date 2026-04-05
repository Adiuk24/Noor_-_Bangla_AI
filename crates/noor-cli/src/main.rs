use clap::{Parser, Subcommand};

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
    },
    /// Evaluate on benchmarks
    Eval {
        /// Path to GGUF model
        #[arg(short, long)]
        model: String,
        /// Comma-separated benchmark names
        #[arg(short, long)]
        bench: String,
    },
    /// Benchmark speed
    Bench {
        /// Path to GGUF model
        #[arg(short, long)]
        model: String,
    },
    /// Convert weights to GGUF
    Convert {
        /// Input directory (safetensors)
        #[arg(short, long)]
        input: String,
        /// Output GGUF path
        #[arg(short, long)]
        output: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train { config, data, resume, steps } => {
            println!("Training with config={config}, data={data}");
            if let Some(r) = resume {
                println!("  Resuming from {r}");
            }
            if let Some(s) = steps {
                println!("  Steps: {s}");
            }
            // TODO: Step 0.19 — implement training loop
        }
        Commands::Run { model, prompt, max_tokens } => {
            println!("Running {model} with prompt: {prompt} (max {max_tokens} tokens)");
            // TODO: Step 0.21 — implement inference
        }
        Commands::Eval { model, bench } => {
            println!("Evaluating {model} on: {bench}");
            // TODO: Step 0.20 — implement eval harness
        }
        Commands::Bench { model } => {
            println!("Benchmarking {model}");
            // TODO: implement speed benchmark
        }
        Commands::Convert { input, output } => {
            println!("Converting {input} → {output}");
            // TODO: Step 0.13 — implement GGUF conversion
        }
    }
}
