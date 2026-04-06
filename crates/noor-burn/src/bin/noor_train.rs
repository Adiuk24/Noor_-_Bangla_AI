//! Noor training CLI — Burn backend.
//!
//! Usage:
//!   cargo run -p noor-burn --release --features ndarray --bin noor-train -- \
//!     --config config/proxy.toml --data data/noor_training/shards/
//!
//! For CUDA GPU:
//!   cargo run -p noor-burn --release --features cuda --bin noor-train -- \
//!     --config config/proxy.toml --data data/noor_training/shards/

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "noor-train", about = "Train Noor model (Burn backend)")]
struct Args {
    /// Path to config TOML (proxy.toml, edge.toml, pro.toml)
    #[arg(long)]
    config: PathBuf,

    /// Path to data shard directory
    #[arg(long)]
    data: PathBuf,

    /// Checkpoint output directory
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: PathBuf,
}

fn main() {
    let args = Args::parse();

    let config = noor_burn::config::NoorConfig::from_toml(&args.config)
        .expect("Failed to load config");

    eprintln!("Noor Training — Burn Backend");
    eprintln!("  Config: {:?}", args.config);
    eprintln!("  Data: {:?}", args.data);
    eprintln!("  Checkpoints: {:?}", args.checkpoint_dir);

    #[cfg(feature = "cuda")]
    {
        use burn::backend::{Cuda, Autodiff};
        type MyBackend = Autodiff<Cuda>;
        let device = burn::backend::cuda::CudaDevice::default();
        eprintln!("  Backend: CUDA GPU");
        noor_burn::training::train::<MyBackend>(
            &config, &device, &args.data, Some(&args.checkpoint_dir),
        );
    }

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::{Wgpu, Autodiff};
        type MyBackend = Autodiff<Wgpu>;
        let device = burn::backend::wgpu::WgpuDevice::default();
        eprintln!("  Backend: WGPU (Vulkan/Metal)");
        noor_burn::training::train::<MyBackend>(
            &config, &device, &args.data, Some(&args.checkpoint_dir),
        );
    }

    #[cfg(feature = "ndarray")]
    {
        use burn::backend::{NdArray, Autodiff};
        type MyBackend = Autodiff<NdArray>;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        eprintln!("  Backend: NdArray (CPU)");
        noor_burn::training::train::<MyBackend>(
            &config, &device, &args.data, Some(&args.checkpoint_dir),
        );
    }
}
