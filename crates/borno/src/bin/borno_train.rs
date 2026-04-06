//! Borno tokenizer training CLI.
//! Usage: borno-train --corpus-dir <path> --output-dir <path> [--merges 61000]

#[cfg(feature = "train")]
fn main() {
    use clap::Parser;
    use std::path::PathBuf;

    #[derive(Parser)]
    #[command(name = "borno-train", about = "Train the Borno 64K BPE tokenizer")]
    struct Args {
        /// Directory containing plain text corpus files (.txt)
        #[arg(long)]
        corpus_dir: PathBuf,

        /// Output directory for trained tokenizer files
        #[arg(long)]
        output_dir: PathBuf,

        /// Number of BPE merges (default: 61000 for 64K total vocab)
        #[arg(long, default_value = "61000")]
        merges: usize,
    }

    let args = Args::parse();

    // Collect .txt files from corpus directory
    let files: Vec<String> = std::fs::read_dir(&args.corpus_dir)
        .expect("Cannot read corpus directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "txt"))
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();

    if files.is_empty() {
        eprintln!("No .txt files found in {:?}", args.corpus_dir);
        std::process::exit(1);
    }

    println!("Borno BPE Trainer");
    println!("  Corpus files: {}", files.len());
    for f in &files {
        println!("    - {f}");
    }
    println!("  Merges: {}", args.merges);
    println!("  Output: {:?}", args.output_dir);
    println!();

    std::fs::create_dir_all(&args.output_dir).expect("Cannot create output directory");

    // Train
    borno::trainer::train_bpe(&files, &args.output_dir, args.merges)
        .expect("Training failed");

    println!("\nTraining complete!");
    println!("  Vocab: {:?}", args.output_dir.join("borno-vocab.json"));
    println!("  Merges: {:?}", args.output_dir.join("borno-merges.txt"));

    // Build rs-bpe encoder from trained vocab and save as bincode
    let tokenizer_json = args.output_dir.join("tokenizer.json");
    if tokenizer_json.exists() {
        println!("\nBuilding rs-bpe encoder...");
        match borno::trainer::load_trained_vocab(&tokenizer_json) {
            Ok(tokens) => {
                let encoder = borno::encoder::BornoEncoder::from_tokens(tokens);
                let encoder_path = args.output_dir.join("borno_encoder.bin");
                encoder.save(&encoder_path).expect("Failed to save encoder");
                println!("  Encoder saved: {:?}", encoder_path);

                // Quick fertility test
                let test_en = "The quick brown fox jumps over the lazy dog.";
                let test_bn = "আমি বাংলায় কথা বলি।";
                let borno = borno::Borno::from_tokens(
                    borno::trainer::load_trained_vocab(&tokenizer_json).unwrap(),
                );
                let en_tokens = borno.encode(test_en);
                let bn_tokens = borno.encode(test_bn);
                let en_words = test_en.split_whitespace().count();
                let bn_words = test_bn.split_whitespace().count();
                println!("\nFertility check:");
                println!(
                    "  English: {} tokens / {} words = {:.2}",
                    en_tokens.len(),
                    en_words,
                    en_tokens.len() as f64 / en_words as f64
                );
                println!(
                    "  Bangla:  {} tokens / {} words = {:.2}",
                    bn_tokens.len(),
                    bn_words,
                    bn_tokens.len() as f64 / bn_words as f64
                );
            }
            Err(e) => eprintln!("Warning: could not build encoder: {e}"),
        }
    }
}

#[cfg(not(feature = "train"))]
fn main() {
    eprintln!("borno-train requires the 'train' feature:");
    eprintln!("  cargo run -p borno --features train --bin borno-train -- --help");
    std::process::exit(1);
}
