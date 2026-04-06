//! Streaming JSONL-to-shard tokenizer with deduplication.
//!
//! Reads JSONL from stdin, deduplicates by content hash, tokenizes with trained
//! Borno encoder, and writes binary shards for noor-train's DataLoader.
//!
//! Shard format: [seq_len: u32][token_ids: u32 * seq_len] repeated.
//!
//! Usage:
//!   cat data.jsonl | borno-shard --encoder path/to/borno_encoder.bin --output-dir shards/
//!
//! Dedup: Uses a hash of (text_length, first_200_bytes) to skip duplicates.
//!        Pass --dedup-file seen.bin to persist the hash set across runs.

#[cfg(feature = "shard")]
fn main() {
    use clap::Parser;
    use std::collections::HashSet;
    use std::io::{self, BufRead, Write};
    use std::path::PathBuf;

    #[derive(Parser)]
    #[command(name = "borno-shard", about = "Stream JSONL → deduplicated Borno-tokenized shards")]
    struct Args {
        /// Path to trained borno_encoder.bin
        #[arg(long)]
        encoder: PathBuf,

        /// Output directory for shard files
        #[arg(long)]
        output_dir: PathBuf,

        /// Starting shard index (to append to existing shards)
        #[arg(long, default_value = "0")]
        start_index: usize,

        /// Context length (sequence length for training)
        #[arg(long, default_value = "2048")]
        context_length: usize,

        /// Max tokens per shard (~500K default)
        #[arg(long, default_value = "500000")]
        shard_size: usize,

        /// JSONL format: "text" (extracts .text), "instruction" (.instruction+.input+.output), "raw" (line is text)
        #[arg(long, default_value = "text")]
        format: String,

        /// Path to dedup hash file. Loads existing hashes at start, saves on finish.
        /// This lets you run multiple batches without re-processing duplicates.
        #[arg(long)]
        dedup_file: Option<PathBuf>,

        /// Minimum text length to include (skip very short entries)
        #[arg(long, default_value = "50")]
        min_length: usize,
    }

    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir).expect("Cannot create output dir");

    let borno = borno::Borno::load(&args.encoder)
        .expect("Failed to load Borno encoder");

    let bos = borno.bos_id();
    let eos = borno.eos_id();
    let seq_len = args.context_length + 1;

    // Load existing dedup hashes if file exists
    let mut seen: HashSet<u64> = HashSet::new();
    if let Some(ref dedup_path) = args.dedup_file {
        if dedup_path.exists() {
            let data = std::fs::read(dedup_path).expect("Failed to read dedup file");
            let count = data.len() / 8;
            for i in 0..count {
                let offset = i * 8;
                let hash = u64::from_le_bytes([
                    data[offset], data[offset+1], data[offset+2], data[offset+3],
                    data[offset+4], data[offset+5], data[offset+6], data[offset+7],
                ]);
                seen.insert(hash);
            }
            eprintln!("Loaded {} existing dedup hashes from {:?}", seen.len(), dedup_path);
        }
    }

    let stdin = io::stdin();
    let reader = stdin.lock();

    let mut token_buf: Vec<u32> = Vec::new();
    let mut shard_idx = args.start_index;
    let mut total_tokens: u64 = 0;
    let mut total_sequences: u64 = 0;
    let mut lines_read: u64 = 0;
    let mut lines_skipped: u64 = 0;
    let mut lines_deduped: u64 = 0;

    let seqs_per_shard = args.shard_size / seq_len;

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => continue,
        };

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        lines_read += 1;

        // Extract text based on format
        let text = match args.format.as_str() {
            "raw" => line.to_string(),
            "text" => {
                match serde_json::from_str::<serde_json::Value>(line) {
                    Ok(v) => {
                        if let Some(t) = v.get("text").and_then(|t| t.as_str()) {
                            t.to_string()
                        } else {
                            lines_skipped += 1;
                            continue;
                        }
                    }
                    Err(_) => {
                        lines_skipped += 1;
                        continue;
                    }
                }
            }
            "instruction" => {
                match serde_json::from_str::<serde_json::Value>(line) {
                    Ok(v) => {
                        let instr = v.get("instruction").and_then(|t| t.as_str()).unwrap_or("");
                        let input = v.get("input").and_then(|t| t.as_str()).unwrap_or("");
                        let output = v.get("output").and_then(|t| t.as_str()).unwrap_or("");
                        if instr.is_empty() && output.is_empty() {
                            lines_skipped += 1;
                            continue;
                        }
                        if input.is_empty() {
                            format!("{}\n{}", instr, output)
                        } else {
                            format!("{}\n{}\n{}", instr, input, output)
                        }
                    }
                    Err(_) => {
                        lines_skipped += 1;
                        continue;
                    }
                }
            }
            _ => {
                eprintln!("Unknown format: {}", args.format);
                std::process::exit(1);
            }
        };

        // Skip too-short entries
        if text.len() < args.min_length {
            lines_skipped += 1;
            continue;
        }

        // Dedup: hash (length, first 200 bytes)
        let hash = content_hash(&text);
        if !seen.insert(hash) {
            lines_deduped += 1;
            continue;
        }

        // Tokenize: BOS + content + EOS
        let mut tokens = Vec::new();
        tokens.push(bos);
        tokens.extend(borno.encode(&text));
        tokens.push(eos);

        token_buf.extend(tokens);

        // Flush full shards
        while token_buf.len() >= seqs_per_shard * seq_len {
            let shard_path = args.output_dir.join(format!("shard_{shard_idx:04}.bin"));
            write_shard(&shard_path, &token_buf, seq_len, seqs_per_shard);

            let shard_tokens = seqs_per_shard * seq_len;
            total_tokens += shard_tokens as u64;
            total_sequences += seqs_per_shard as u64;

            eprintln!(
                "  {} — {} sequences, {} tokens",
                shard_path.display(),
                seqs_per_shard,
                shard_tokens
            );

            token_buf.drain(..seqs_per_shard * seq_len);
            shard_idx += 1;
        }

        // Progress every 10K lines
        if lines_read % 10000 == 0 {
            eprint!(
                "\r  [{} lines, {} deduped, {} shards, {}M tokens]",
                lines_read, lines_deduped, shard_idx - args.start_index, total_tokens / 1_000_000
            );
            io::stderr().flush().ok();
        }
    }

    // Flush remaining tokens
    if token_buf.len() >= seq_len {
        let remaining_seqs = token_buf.len() / seq_len;
        let shard_path = args.output_dir.join(format!("shard_{shard_idx:04}.bin"));
        write_shard(&shard_path, &token_buf, seq_len, remaining_seqs);

        let shard_tokens = remaining_seqs * seq_len;
        total_tokens += shard_tokens as u64;
        total_sequences += remaining_seqs as u64;

        eprintln!(
            "  {} — {} sequences, {} tokens",
            shard_path.display(),
            remaining_seqs,
            shard_tokens
        );
        shard_idx += 1;
    }

    // Save dedup hashes for next run
    if let Some(ref dedup_path) = args.dedup_file {
        let mut data = Vec::with_capacity(seen.len() * 8);
        for hash in &seen {
            data.extend_from_slice(&hash.to_le_bytes());
        }
        std::fs::write(dedup_path, &data).expect("Failed to write dedup file");
        eprintln!("Saved {} dedup hashes to {:?}", seen.len(), dedup_path);
    }

    eprintln!();
    eprintln!(
        "Done. {} shards, {} sequences, {}M tokens total.",
        shard_idx - args.start_index, total_sequences, total_tokens / 1_000_000
    );
    eprintln!(
        "  {} lines read, {} skipped, {} duplicates removed, {} unique hashes",
        lines_read, lines_skipped, lines_deduped, seen.len()
    );
}

/// FNV-1a inspired hash of (text_length, first 200 bytes, last 100 bytes).
/// Fast, low-collision for text dedup.
#[cfg(feature = "shard")]
fn content_hash(text: &str) -> u64 {
    let bytes = text.as_bytes();
    let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
    let prime: u64 = 0x100000001b3;

    // Mix in length
    let len = bytes.len() as u64;
    h ^= len;
    h = h.wrapping_mul(prime);

    // Hash first 200 bytes
    let prefix_len = bytes.len().min(200);
    for &b in &bytes[..prefix_len] {
        h ^= b as u64;
        h = h.wrapping_mul(prime);
    }

    // Hash last 100 bytes (catches entries with same prefix but different endings)
    if bytes.len() > 200 {
        let suffix_start = bytes.len().saturating_sub(100);
        for &b in &bytes[suffix_start..] {
            h ^= b as u64;
            h = h.wrapping_mul(prime);
        }
    }

    h
}

#[cfg(feature = "shard")]
fn write_shard(path: &std::path::Path, tokens: &[u32], seq_len: usize, num_seqs: usize) {
    use std::io::Write;
    let mut file = std::fs::File::create(path).expect("Failed to create shard file");
    for i in 0..num_seqs {
        let start = i * seq_len;
        let seq = &tokens[start..start + seq_len];
        let len = seq.len() as u32;
        file.write_all(&len.to_le_bytes()).unwrap();
        for &token in seq {
            file.write_all(&token.to_le_bytes()).unwrap();
        }
    }
}

#[cfg(not(feature = "shard"))]
fn main() {
    eprintln!("borno-shard requires the 'shard' feature:");
    eprintln!("  cargo run -p borno --features shard --bin borno-shard -- --help");
    std::process::exit(1);
}
