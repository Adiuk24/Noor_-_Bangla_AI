//! Convert Burn .mpk checkpoint → half-precision .bin for lighter inference.
//!
//! Usage:
//!   cargo run -p noor-burn --release --features ndarray --bin noor-convert -- \
//!     --config config/edge_kaggle.toml \
//!     --checkpoint checkpoints/kaggle_output/noor/checkpoints/noor_final \
//!     --output checkpoints/noor_final_f16

use clap::Parser;
use std::path::PathBuf;

use burn::record::{CompactRecorder, HalfPrecisionSettings, BinFileRecorder};
use burn::prelude::*;

#[derive(Parser)]
#[command(name = "noor-convert", about = "Convert Burn .mpk → f16 .bin")]
struct Args {
    #[arg(long)]
    config: PathBuf,

    #[arg(long)]
    checkpoint: PathBuf,

    #[arg(long)]
    output: PathBuf,
}

fn convert<B: Backend>(config: &noor_burn::config::NoorConfig, device: &B::Device, args: &Args) {
    eprintln!("=== Noor Convert: f32 .mpk → f16 .bin ===");

    let model = noor_burn::model::NoorModel::<B>::from_config(config, device);
    eprintln!("  Params: {}", model.param_count());

    let model = model
        .load_file(&args.checkpoint, &CompactRecorder::new(), device)
        .expect("Failed to load checkpoint");
    eprintln!("  Checkpoint loaded!");

    let recorder = BinFileRecorder::<HalfPrecisionSettings>::new();
    model
        .save_file(&args.output, &recorder)
        .expect("Failed to save");

    let out_path = format!("{}.bin", args.output.display());
    let out_size = std::fs::metadata(&out_path)
        .or_else(|_| std::fs::metadata(&args.output))
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!("  Saved: {} ({:.0} MB)", args.output.display(), out_size as f64 / 1_048_576.0);
    eprintln!("  Done!");
}

fn main() {
    let args = Args::parse();
    let config = noor_burn::config::NoorConfig::from_toml(&args.config)
        .expect("Failed to load config");

    use burn::backend::NdArray;
    type B = NdArray;
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    convert::<B>(&config, &device, &args);
}
