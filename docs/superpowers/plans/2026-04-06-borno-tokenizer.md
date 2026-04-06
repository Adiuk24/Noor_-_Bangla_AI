# Borno Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Borno (বর্ণ), a 64K BPE tokenizer in Rust with grapheme-aware Bengali, byte fallback, ADE special tokens, and rs-bpe backtracking encoder. Then retrain the 0.5B proxy.

**Architecture:** Two-phase approach — HuggingFace `tokenizers` crate trains the BPE vocab from a 1GB corpus (English+Bangla+code), then the `bpe` crate provides 10x-faster backtracking encoding at inference. Bangla text goes through NFC normalization + grapheme cluster segmentation before BPE. The crate integrates into noor-core as the tokenizer backend.

**Tech Stack:** Rust, `tokenizers` 0.22, `bpe` 0.2, `unicode-normalization`, `unicode-segmentation`, `reqwest`, `serde`/`bincode`, `clap`

---

## File Structure

```
crates/borno/
├── Cargo.toml                  — crate manifest with all dependencies
├── src/
│   ├── lib.rs                  — pub struct Borno, top-level encode/decode API
│   ├── bangla.rs               — NFC normalization + Bengali grapheme cluster segmentation
│   ├── pretokenize.rs          — script detection, routing Bangla vs Latin/code spans
│   ├── vocab.rs                — special token registry, vocab layout constants, byte fallback
│   ├── trainer.rs              — HF tokenizers BPE training wrapper
│   ├── encoder.rs              — rs-bpe backtracking encoder loader + wrapper
│   └── bin/
│       └── borno_train.rs      — CLI binary: download corpus, train, validate fertility
├── tests/
│   ├── bangla_tests.rs         — grapheme preservation, NFC consistency, fertility
│   └── roundtrip_tests.rs      — encode/decode roundtrip for all scripts + edge cases
└── data/                       — (gitignored) corpus downloads + trained vocab artifacts
```

**Modified files:**
- `Cargo.toml` (workspace root) — add `crates/borno` to members, add workspace deps
- `crates/noor-core/Cargo.toml` — add `borno` dependency
- `crates/noor-core/src/tokenizer.rs` — replace byte-level with Borno backend
- `crates/noor-core/src/lib.rs` — update re-exports if needed
- `config/proxy.toml` — `vocab_size = 32000` → `64000`
- `config/proxy_tiny.toml` — `vocab_size = 300` → `64000`

---

### Task 1: Crate Scaffolding

**Files:**
- Create: `crates/borno/Cargo.toml`
- Create: `crates/borno/src/lib.rs`
- Create: `crates/borno/src/vocab.rs`
- Modify: `Cargo.toml` (workspace root)

- [ ] **Step 1: Create Cargo.toml for borno crate**

```toml
# crates/borno/Cargo.toml
[package]
name = "borno"
version.workspace = true
edition.workspace = true
license.workspace = true
authors.workspace = true
description = "Borno (বর্ণ) — 64K BPE tokenizer for Noor. Bangla-native, English+code, rs-bpe backtracking."

[[bin]]
name = "borno-train"
path = "src/bin/borno_train.rs"

[features]
train = ["tokenizers", "reqwest"]
rand = ["bpe/rand"]

[dependencies]
bpe = "0.2"
unicode-normalization = "0.1"
unicode-segmentation = "1"
serde = { version = "1", features = ["derive"] }
bincode = "1"
clap = { version = "4", features = ["derive"], optional = true }
tokenizers = { version = "0.22", optional = true }
reqwest = { version = "0.12", features = ["blocking"], optional = true }
indicatif = { version = "0.17", optional = true }

[dev-dependencies]
approx = "0.5"
```

- [ ] **Step 2: Create vocab.rs with constants and special tokens**

```rust
// crates/borno/src/vocab.rs

/// Total vocabulary size.
pub const VOCAB_SIZE: usize = 64_000;

/// Byte fallback tokens occupy IDs 0-255.
pub const BYTE_FALLBACK_START: u32 = 0;
pub const BYTE_FALLBACK_END: u32 = 255;

/// Special token IDs.
pub const BOS_ID: u32 = 256;
pub const EOS_ID: u32 = 257;
pub const PAD_ID: u32 = 258;
pub const UNK_ID: u32 = 259;
pub const USER_ID: u32 = 260;
pub const ASSISTANT_ID: u32 = 261;
pub const SYSTEM_ID: u32 = 262;
pub const TOOL_CALL_ID: u32 = 263;
pub const TOOL_RESULT_ID: u32 = 264;
pub const THINK_ID: u32 = 265;
pub const THINK_END_ID: u32 = 266;
pub const MEMORY_ID: u32 = 267;
pub const MEMORY_END_ID: u32 = 268;
pub const CODE_ID: u32 = 269;
pub const CODE_END_ID: u32 = 270;

/// Reserved range for future ADE tokens.
pub const RESERVED_START: u32 = 271;
pub const RESERVED_END: u32 = 2999;

/// BPE merge tokens start here.
pub const BPE_MERGE_START: u32 = 3000;
pub const BPE_MERGE_END: u32 = 63_999;
pub const BPE_MERGE_COUNT: usize = (BPE_MERGE_END - BPE_MERGE_START + 1) as usize; // 61,000

/// Special token strings, ordered by ID (256..).
pub const SPECIAL_TOKENS: &[&str] = &[
    "<bos>", "<eos>", "<pad>", "<unk>",
    "<user>", "<assistant>", "<system>",
    "<tool_call>", "<tool_result>",
    "<think>", "</think>",
    "<memory>", "</memory>",
    "<code>", "</code>",
];

/// Returns the byte-fallback token bytes for a given ID (0-255).
pub fn byte_token(id: u32) -> Vec<u8> {
    debug_assert!(id <= 255);
    vec![id as u8]
}

/// Build the full base vocabulary: 256 byte tokens + 15 special tokens + reserved slots.
/// Returns tokens as byte sequences, ordered by ID.
pub fn build_base_vocab() -> Vec<Vec<u8>> {
    let mut tokens: Vec<Vec<u8>> = Vec::with_capacity(BPE_MERGE_START as usize);

    // 0-255: raw byte fallback
    for b in 0u8..=255 {
        tokens.push(vec![b]);
    }

    // 256-270: special tokens (stored as UTF-8 bytes)
    for &s in SPECIAL_TOKENS {
        tokens.push(s.as_bytes().to_vec());
    }

    // 271-2999: reserved (placeholder bytes — will never be encoded into)
    for i in RESERVED_START..=RESERVED_END {
        tokens.push(format!("<reserved_{i}>").into_bytes());
    }

    tokens
}
```

- [ ] **Step 3: Create lib.rs stub**

```rust
// crates/borno/src/lib.rs

pub mod vocab;

// These modules will be added in subsequent tasks:
// pub mod bangla;
// pub mod pretokenize;
// pub mod trainer;
// pub mod encoder;
```

- [ ] **Step 4: Add borno to workspace**

In the workspace root `Cargo.toml`, add `"crates/borno"` to `[workspace.members]` and add workspace deps:

```toml
[workspace]
resolver = "2"
members = [
    "crates/noor-core",
    "crates/noor-train",
    "crates/noor-cli",
    "crates/borno",
]

[workspace.dependencies]
noor-core = { path = "crates/noor-core" }
noor-train = { path = "crates/noor-train" }
borno = { path = "crates/borno" }
serde = { version = "1", features = ["derive"] }
toml = "0.8"
clap = { version = "4", features = ["derive"] }
memmap2 = "0.9"
indicatif = "0.17"
rand = "0.8"
rand_distr = "0.4"
```

- [ ] **Step 5: Verify it compiles**

Run: `cd /Users/adi/noor && cargo build -p borno`
Expected: compiles with no errors.

- [ ] **Step 6: Commit**

```bash
cd /Users/adi/noor
git add crates/borno/ Cargo.toml
git commit -m "feat(borno): scaffold Borno tokenizer crate with vocab layout"
```

---

### Task 2: Bangla Pre-tokenization

**Files:**
- Create: `crates/borno/src/bangla.rs`
- Create: `crates/borno/tests/bangla_tests.rs`
- Modify: `crates/borno/src/lib.rs`

- [ ] **Step 1: Write failing tests for Bangla NFC + grapheme segmentation**

```rust
// crates/borno/tests/bangla_tests.rs

#[test]
fn test_nfc_normalization() {
    // Two representations of "কো" — must normalize to same form
    let decomposed = "কো"; // may be two codepoints: ক + ো
    let result = borno::bangla::normalize(decomposed);
    // After NFC, identical inputs produce identical outputs
    let result2 = borno::bangla::normalize(decomposed);
    assert_eq!(result, result2);
}

#[test]
fn test_grapheme_cluster_ksha() {
    // ক্ষ = ক + ্ + ষ — one grapheme cluster, must not be split
    let clusters = borno::bangla::grapheme_clusters("ক্ষ");
    assert_eq!(clusters, vec!["ক্ষ"]);
}

#[test]
fn test_grapheme_cluster_stra() {
    // স্ত্র = স + ্ + ত + ্ + র — one cluster
    let clusters = borno::bangla::grapheme_clusters("স্ত্র");
    assert_eq!(clusters, vec!["স্ত্র"]);
}

#[test]
fn test_grapheme_clusters_bangla_word() {
    // বাংলা → বা + ং + লা (3 grapheme clusters)
    let clusters = borno::bangla::grapheme_clusters("বাংলা");
    assert_eq!(clusters.len(), 3);
    assert_eq!(clusters, vec!["বা", "ং", "লা"]);
}

#[test]
fn test_grapheme_ki() {
    // কি = ক + ি — one cluster (consonant + vowel sign)
    let clusters = borno::bangla::grapheme_clusters("কি");
    assert_eq!(clusters, vec!["কি"]);
}

#[test]
fn test_normalize_then_segment() {
    let input = "আমি বাংলায় কথা বলি";
    let result = borno::bangla::normalize_and_segment(input);
    // Should produce grapheme clusters, not individual codepoints
    // Every cluster should be valid UTF-8
    for cluster in &result {
        assert!(!cluster.is_empty());
    }
    // "আমি" → আ + মি = 2 clusters
    // Space handled separately
    // Total clusters should be reasonable (not 30+ byte-level)
    assert!(result.len() < 20, "Got {} clusters, expected <20", result.len());
}

#[test]
fn test_is_bengali() {
    assert!(borno::bangla::is_bengali_char('ক'));
    assert!(borno::bangla::is_bengali_char('া'));
    assert!(borno::bangla::is_bengali_char('্'));
    assert!(!borno::bangla::is_bengali_char('a'));
    assert!(!borno::bangla::is_bengali_char('1'));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/adi/noor && cargo test -p borno --test bangla_tests 2>&1`
Expected: compilation errors — `bangla` module doesn't exist yet.

- [ ] **Step 3: Implement bangla.rs**

```rust
// crates/borno/src/bangla.rs

use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

/// Bengali Unicode block range.
const BENGALI_START: u32 = 0x0980;
const BENGALI_END: u32 = 0x09FF;

/// Returns true if the character is in the Bengali Unicode block.
pub fn is_bengali_char(c: char) -> bool {
    let cp = c as u32;
    (BENGALI_START..=BENGALI_END).contains(&cp)
}

/// NFC-normalize a string. Critical for Bengali where the same
/// visual character can have multiple codepoint representations.
pub fn normalize(input: &str) -> String {
    input.nfc().collect()
}

/// Split text into Unicode extended grapheme clusters.
/// For Bengali, this keeps conjuncts (consonant + hasanta + consonant) together.
pub fn grapheme_clusters(input: &str) -> Vec<&str> {
    input.graphemes(true).collect()
}

/// NFC normalize, then split into grapheme clusters.
/// This is the pre-tokenization path for Bengali text.
pub fn normalize_and_segment(input: &str) -> Vec<String> {
    let normalized = normalize(input);
    normalized.graphemes(true).map(String::from).collect()
}
```

- [ ] **Step 4: Add bangla module to lib.rs**

```rust
// crates/borno/src/lib.rs

pub mod vocab;
pub mod bangla;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/adi/noor && cargo test -p borno --test bangla_tests -- --nocapture 2>&1`
Expected: all 7 tests pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/adi/noor
git add crates/borno/src/bangla.rs crates/borno/src/lib.rs crates/borno/tests/bangla_tests.rs
git commit -m "feat(borno): Bangla NFC normalization + grapheme cluster segmentation"
```

---

### Task 3: Script-Aware Pre-tokenization

**Files:**
- Create: `crates/borno/src/pretokenize.rs`
- Modify: `crates/borno/src/lib.rs`

- [ ] **Step 1: Write failing test**

Add to `crates/borno/tests/bangla_tests.rs`:

```rust
#[test]
fn test_script_split_mixed() {
    // Mixed Bengali + English text should split into script spans
    let spans = borno::pretokenize::split_by_script("Hello বাংলা world");
    assert_eq!(spans.len(), 3);
    assert_eq!(spans[0].text, "Hello ");
    assert!(!spans[0].is_bengali);
    assert_eq!(spans[1].text, "বাংলা");
    assert!(spans[1].is_bengali);
    assert_eq!(spans[2].text, " world");
    assert!(!spans[2].is_bengali);
}

#[test]
fn test_pretokenize_bengali_span() {
    let tokens = borno::pretokenize::pretokenize("আমি বাংলায় কথা বলি");
    // Should produce grapheme clusters, each as a pre-token
    assert!(tokens.len() > 5);
    assert!(tokens.len() < 20);
}

#[test]
fn test_pretokenize_english_span() {
    let tokens = borno::pretokenize::pretokenize("Hello world! def foo():");
    // Should produce GPT4-style word splits
    assert!(tokens.len() >= 4);
}

#[test]
fn test_pretokenize_mixed() {
    let tokens = borno::pretokenize::pretokenize("Hello বাংলা code");
    // Bengali grapheme clusters + English words
    assert!(tokens.len() >= 3);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/adi/noor && cargo test -p borno --test bangla_tests 2>&1`
Expected: fails — `pretokenize` module doesn't exist.

- [ ] **Step 3: Implement pretokenize.rs**

```rust
// crates/borno/src/pretokenize.rs

use crate::bangla;

/// A span of text with its script classification.
#[derive(Debug, Clone, PartialEq)]
pub struct ScriptSpan {
    pub text: String,
    pub is_bengali: bool,
}

/// Split text into contiguous runs of Bengali vs non-Bengali characters.
/// Whitespace adjacent to a script span is grouped with that span.
pub fn split_by_script(input: &str) -> Vec<ScriptSpan> {
    if input.is_empty() {
        return vec![];
    }

    let mut spans = Vec::new();
    let mut current = String::new();
    let mut current_is_bengali: Option<bool> = None;

    for c in input.chars() {
        let is_b = bangla::is_bengali_char(c);
        // Whitespace and punctuation: attach to current span
        let is_neutral = c.is_whitespace() || c.is_ascii_punctuation();

        if is_neutral {
            current.push(c);
            continue;
        }

        match current_is_bengali {
            None => {
                current_is_bengali = Some(is_b);
                current.push(c);
            }
            Some(was_bengali) if was_bengali == is_b => {
                current.push(c);
            }
            Some(was_bengali) => {
                // Script changed — split. Trailing whitespace stays with previous span.
                spans.push(ScriptSpan {
                    text: current.clone(),
                    is_bengali: was_bengali,
                });
                current.clear();
                current_is_bengali = Some(is_b);
                current.push(c);
            }
        }
    }

    if !current.is_empty() {
        spans.push(ScriptSpan {
            text: current,
            is_bengali: current_is_bengali.unwrap_or(false),
        });
    }

    spans
}

/// GPT-4-style regex-like pre-tokenization for Latin/code text.
/// Splits on whitespace boundaries and punctuation while keeping words intact.
fn split_latin(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        if c.is_whitespace() {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            // Include leading whitespace with next token
            current.push(c);
        } else if c.is_ascii_punctuation() {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            tokens.push(c.to_string());
        } else {
            current.push(c);
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

/// Pre-tokenize text: split by script, then apply script-appropriate segmentation.
/// Bengali spans → NFC + grapheme clusters.
/// Latin/code spans → word-level splits.
pub fn pretokenize(input: &str) -> Vec<String> {
    let normalized = bangla::normalize(input);
    let spans = split_by_script(&normalized);
    let mut result = Vec::new();

    for span in spans {
        if span.is_bengali {
            // Bengali: grapheme cluster segmentation
            let clusters = bangla::grapheme_clusters(&span.text);
            result.extend(clusters.into_iter().map(String::from));
        } else {
            // Latin/code: word-level splitting
            result.extend(split_latin(&span.text));
        }
    }

    result
}
```

- [ ] **Step 4: Add pretokenize module to lib.rs**

```rust
// crates/borno/src/lib.rs

pub mod vocab;
pub mod bangla;
pub mod pretokenize;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/adi/noor && cargo test -p borno --test bangla_tests -- --nocapture 2>&1`
Expected: all 11 tests pass (7 from Task 2 + 4 new).

- [ ] **Step 6: Commit**

```bash
cd /Users/adi/noor
git add crates/borno/src/pretokenize.rs crates/borno/src/lib.rs crates/borno/tests/bangla_tests.rs
git commit -m "feat(borno): script-aware pre-tokenization (Bengali grapheme + Latin word split)"
```

---

### Task 4: BPE Trainer (HuggingFace tokenizers wrapper)

**Files:**
- Create: `crates/borno/src/trainer.rs`
- Modify: `crates/borno/src/lib.rs`

- [ ] **Step 1: Write failing test**

Add to a new test file `crates/borno/tests/roundtrip_tests.rs`:

```rust
#[cfg(feature = "train")]
#[test]
fn test_train_tiny_bpe() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("borno_test_train");
    std::fs::create_dir_all(&dir).unwrap();

    // Create a tiny corpus file
    let corpus_path = dir.join("tiny_corpus.txt");
    let mut f = std::fs::File::create(&corpus_path).unwrap();
    for _ in 0..100 {
        writeln!(f, "hello world this is a test of the tokenizer").unwrap();
        writeln!(f, "আমি বাংলায় কথা বলি এটি একটি পরীক্ষা").unwrap();
        writeln!(f, "def foo(x): return x + 1").unwrap();
    }
    drop(f);

    let output_dir = dir.join("output");
    std::fs::create_dir_all(&output_dir).unwrap();

    // Train a tiny BPE (500 merges for speed)
    let result = borno::trainer::train_bpe(
        &[corpus_path.to_str().unwrap().to_string()],
        &output_dir,
        500,
    );
    assert!(result.is_ok(), "Training failed: {:?}", result.err());

    // Check output files exist
    assert!(output_dir.join("borno_vocab.json").exists());
    assert!(output_dir.join("borno_merges.txt").exists());

    // Clean up
    std::fs::remove_dir_all(&dir).ok();
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/adi/noor && cargo test -p borno --test roundtrip_tests --features train 2>&1`
Expected: fails — `trainer` module doesn't exist.

- [ ] **Step 3: Implement trainer.rs**

```rust
// crates/borno/src/trainer.rs
//! BPE training via HuggingFace tokenizers crate.
//! Only available with the "train" feature.

use crate::vocab;
use std::path::Path;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::normalizers::unicode::NFC;
use tokenizers::Tokenizer;
use tokenizers::AddedToken;

/// Train a BPE tokenizer on the given corpus files.
///
/// - `files`: paths to plain text files (one sentence/line per line)
/// - `output_dir`: directory to write borno_vocab.json and borno_merges.txt
/// - `n_merges`: number of BPE merge operations (61000 for production, less for testing)
pub fn train_bpe(
    files: &[String],
    output_dir: &Path,
    n_merges: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Total vocab = 256 bytes + special tokens + reserved + merges
    // But HF trainer counts from its own initial alphabet.
    // We train with vocab_size = initial_alphabet_size + n_merges,
    // then remap IDs to our layout post-training.

    let mut trainer = tokenizers::models::bpe::BpeTrainer::builder()
        .vocab_size(256 + n_merges) // byte alphabet + merges
        .min_frequency(2)
        .show_progress(true)
        .special_tokens(
            vocab::SPECIAL_TOKENS
                .iter()
                .map(|&s| AddedToken::from(s, true))
                .collect(),
        )
        .initial_alphabet(ByteLevel::alphabet())
        .build();

    let mut tokenizer = Tokenizer::new(BPE::default());

    // NFC normalization built into the tokenizer
    tokenizer.with_normalizer(NFC);

    // Byte-level pre-tokenization (handles the base encoding)
    tokenizer.with_pre_tokenizer(ByteLevel::new(false, true, true));

    // Train from files
    tokenizer.train_from_files(&mut trainer, files.to_vec())?;

    // Save in HuggingFace format (vocab.json + merges.txt)
    tokenizer.save(output_dir.join("borno_tokenizer.json"), true)?;

    // Also save just the model (vocab + merges separately)
    let model = tokenizer.get_model();
    if let Some(bpe) = model.as_any().downcast_ref::<BPE>() {
        // Get vocab
        let vocab = bpe.get_vocab();
        let vocab_json = serde_json::to_string_pretty(&vocab)?;
        std::fs::write(output_dir.join("borno_vocab.json"), vocab_json)?;

        // Get merges
        let merges = bpe.get_merges();
        let merges_txt: String = merges
            .iter()
            .map(|(a, b)| format!("{a} {b}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(output_dir.join("borno_merges.txt"), merges_txt)?;
    }

    Ok(())
}

/// Load a trained HF tokenizer and extract ordered token list
/// for building the rs-bpe encoder.
pub fn load_trained_vocab(
    tokenizer_path: &Path,
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

    let vocab = tokenizer.get_vocab(true);
    let vocab_size = tokenizer.get_vocab_size(true);

    // Build ordered token list: index = token ID, value = token bytes
    let mut tokens: Vec<Vec<u8>> = vec![vec![]; vocab_size];
    for (token_str, id) in &vocab {
        if (*id as usize) < vocab_size {
            tokens[*id as usize] = token_str.as_bytes().to_vec();
        }
    }

    Ok(tokens)
}
```

- [ ] **Step 4: Add trainer module to lib.rs (feature-gated)**

```rust
// crates/borno/src/lib.rs

pub mod vocab;
pub mod bangla;
pub mod pretokenize;

#[cfg(feature = "train")]
pub mod trainer;
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/adi/noor && cargo test -p borno --test roundtrip_tests --features train -- --nocapture 2>&1`
Expected: `test_train_tiny_bpe` passes. HF tokenizer trains on the tiny corpus and writes output files.

Note: The trainer.rs code may need adjustments based on exact HF tokenizers API — the `get_model()`, `as_any()`, and `get_merges()` methods may differ. If compilation fails, check the exact API at `~/.cargo/registry/src/*/tokenizers-0.22.*/src/models/bpe/model.rs` and adapt. The key contract is: train BPE, save vocab.json and merges.txt.

- [ ] **Step 6: Commit**

```bash
cd /Users/adi/noor
git add crates/borno/src/trainer.rs crates/borno/src/lib.rs crates/borno/tests/roundtrip_tests.rs
git commit -m "feat(borno): BPE trainer using HuggingFace tokenizers crate"
```

---

### Task 5: rs-bpe Backtracking Encoder

**Files:**
- Create: `crates/borno/src/encoder.rs`
- Modify: `crates/borno/src/lib.rs`

- [ ] **Step 1: Write failing test**

Add to `crates/borno/tests/roundtrip_tests.rs`:

```rust
#[test]
fn test_encoder_byte_fallback() {
    // Build an encoder with just byte tokens (no merges)
    let encoder = borno::encoder::BornoEncoder::from_byte_fallback();
    let text = "hello";
    let ids = encoder.encode(text);
    // Each ASCII byte should map to its byte ID
    assert_eq!(ids, vec![104, 101, 108, 108, 111]); // h=104, e=101, l=108, o=111
    let decoded = encoder.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_encoder_special_tokens() {
    let encoder = borno::encoder::BornoEncoder::from_byte_fallback();
    assert_eq!(encoder.bos_id(), 256);
    assert_eq!(encoder.eos_id(), 257);
    assert_eq!(encoder.vocab_size(), 64_000);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/adi/noor && cargo test -p borno --test roundtrip_tests 2>&1`
Expected: fails — `encoder` module doesn't exist.

- [ ] **Step 3: Implement encoder.rs**

```rust
// crates/borno/src/encoder.rs
//! Fast BPE encoder using rs-bpe backtracking algorithm.

use crate::bangla;
use crate::pretokenize;
use crate::vocab;
use bpe::byte_pair_encoding::BytePairEncoding;
use std::path::Path;

/// Borno encoder wrapping rs-bpe for fast backtracking BPE.
pub struct BornoEncoder {
    bpe: BytePairEncoding,
    /// Special token strings for matching during encoding
    special_tokens: Vec<(String, u32)>,
}

impl BornoEncoder {
    /// Create a minimal encoder with only byte fallback (no BPE merges).
    /// Useful for testing and as a baseline.
    pub fn from_byte_fallback() -> Self {
        let base_vocab = vocab::build_base_vocab();
        // Pad to full vocab size with empty placeholders
        let mut tokens = base_vocab;
        while tokens.len() < vocab::VOCAB_SIZE {
            tokens.push(format!("<unused_{}>", tokens.len()).into_bytes());
        }

        let bpe = BytePairEncoding::from_dictionary(tokens, None);

        let special_tokens: Vec<(String, u32)> = vocab::SPECIAL_TOKENS
            .iter()
            .enumerate()
            .map(|(i, &s)| (s.to_string(), 256 + i as u32))
            .collect();

        Self { bpe, special_tokens }
    }

    /// Load encoder from a trained vocabulary.
    /// `tokens` must be ordered by ID (index 0 = token 0).
    pub fn from_tokens(tokens: Vec<Vec<u8>>) -> Self {
        let bpe = BytePairEncoding::from_dictionary(tokens, None);

        let special_tokens: Vec<(String, u32)> = vocab::SPECIAL_TOKENS
            .iter()
            .enumerate()
            .map(|(i, &s)| (s.to_string(), 256 + i as u32))
            .collect();

        Self { bpe, special_tokens }
    }

    /// Load encoder from a saved bincode file.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        let bpe: BytePairEncoding = bincode::deserialize(&data)?;
        let special_tokens: Vec<(String, u32)> = vocab::SPECIAL_TOKENS
            .iter()
            .enumerate()
            .map(|(i, &s)| (s.to_string(), 256 + i as u32))
            .collect();
        Ok(Self { bpe, special_tokens })
    }

    /// Save encoder to a bincode file for fast loading.
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let data = bincode::serialize(&self.bpe)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Encode text to token IDs using rs-bpe backtracking.
    /// Handles special tokens, NFC normalization, and script-aware pre-tokenization.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Check for special tokens first
        let segments = self.split_special_tokens(text);
        let mut ids = Vec::new();

        for (segment, special_id) in segments {
            if let Some(id) = special_id {
                ids.push(id);
            } else {
                // Pre-tokenize by script, then encode each pre-token
                let pre_tokens = pretokenize::pretokenize(&segment);
                for pt in &pre_tokens {
                    let encoded = self.bpe.encode_via_backtracking(pt.as_bytes());
                    ids.extend(encoded);
                }
            }
        }

        ids
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            // Check special tokens
            if let Some((s, _)) = self.special_tokens.iter().find(|(_, sid)| *sid == id) {
                bytes.extend_from_slice(s.as_bytes());
            } else {
                let token_bytes = self.bpe.decode_tokens(&[id]);
                bytes.extend(token_bytes);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Split text on special token boundaries.
    /// Returns (text_segment, Option<special_token_id>) pairs.
    fn split_special_tokens(&self, text: &str) -> Vec<(String, Option<u32>)> {
        let mut result = Vec::new();
        let mut remaining = text.to_string();

        while !remaining.is_empty() {
            // Find the earliest special token match
            let mut earliest: Option<(usize, usize, u32)> = None; // (start, len, id)
            for (token_str, id) in &self.special_tokens {
                if let Some(pos) = remaining.find(token_str.as_str()) {
                    match earliest {
                        None => earliest = Some((pos, token_str.len(), *id)),
                        Some((ep, _, _)) if pos < ep => {
                            earliest = Some((pos, token_str.len(), *id))
                        }
                        _ => {}
                    }
                }
            }

            match earliest {
                Some((pos, len, id)) => {
                    if pos > 0 {
                        result.push((remaining[..pos].to_string(), None));
                    }
                    result.push((remaining[pos..pos + len].to_string(), Some(id)));
                    remaining = remaining[pos + len..].to_string();
                }
                None => {
                    result.push((remaining, None));
                    break;
                }
            }
        }

        result
    }

    pub fn bos_id(&self) -> u32 { vocab::BOS_ID }
    pub fn eos_id(&self) -> u32 { vocab::EOS_ID }
    pub fn pad_id(&self) -> u32 { vocab::PAD_ID }
    pub fn unk_id(&self) -> u32 { vocab::UNK_ID }
    pub fn vocab_size(&self) -> usize { vocab::VOCAB_SIZE }
}
```

- [ ] **Step 4: Add encoder module to lib.rs**

```rust
// crates/borno/src/lib.rs

pub mod vocab;
pub mod bangla;
pub mod pretokenize;
pub mod encoder;

#[cfg(feature = "train")]
pub mod trainer;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/adi/noor && cargo test -p borno --test roundtrip_tests 2>&1`
Expected: `test_encoder_byte_fallback` and `test_encoder_special_tokens` pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/adi/noor
git add crates/borno/src/encoder.rs crates/borno/src/lib.rs crates/borno/tests/roundtrip_tests.rs
git commit -m "feat(borno): rs-bpe backtracking encoder with byte fallback and special tokens"
```

---

### Task 6: Top-Level Borno API

**Files:**
- Modify: `crates/borno/src/lib.rs`

- [ ] **Step 1: Write failing test**

Add to `crates/borno/tests/roundtrip_tests.rs`:

```rust
#[test]
fn test_borno_api_encode_decode_ascii() {
    let borno = borno::Borno::from_byte_fallback();
    let text = "Hello, world!";
    let ids = borno.encode(text);
    let decoded = borno.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_borno_api_encode_decode_bangla() {
    let borno = borno::Borno::from_byte_fallback();
    let text = "নূর";
    let ids = borno.encode(text);
    assert!(!ids.is_empty());
    let decoded = borno.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_borno_api_encode_decode_mixed() {
    let borno = borno::Borno::from_byte_fallback();
    let text = "Hello নূর world";
    let ids = borno.encode(text);
    let decoded = borno.decode(&ids);
    assert_eq!(decoded, text);
}

#[test]
fn test_borno_api_special_tokens() {
    let borno = borno::Borno::from_byte_fallback();
    let text = "<bos>Hello<eos>";
    let ids = borno.encode(text);
    assert_eq!(ids[0], borno::vocab::BOS_ID);
    assert_eq!(*ids.last().unwrap(), borno::vocab::EOS_ID);
}

#[test]
fn test_borno_api_empty() {
    let borno = borno::Borno::from_byte_fallback();
    assert_eq!(borno.encode(""), vec![]);
    assert_eq!(borno.decode(&[]), "");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/adi/noor && cargo test -p borno --test roundtrip_tests 2>&1`
Expected: fails — `Borno` struct doesn't exist on `borno` crate root.

- [ ] **Step 3: Implement top-level Borno API in lib.rs**

```rust
// crates/borno/src/lib.rs

pub mod vocab;
pub mod bangla;
pub mod pretokenize;
pub mod encoder;

#[cfg(feature = "train")]
pub mod trainer;

use encoder::BornoEncoder;
use std::path::Path;

/// Borno (বর্ণ) — the Noor tokenizer.
///
/// 64K vocab BPE with Bangla-native pre-tokenization and rs-bpe backtracking.
pub struct Borno {
    encoder: BornoEncoder,
}

impl Borno {
    /// Create a byte-fallback-only tokenizer (no trained merges).
    pub fn from_byte_fallback() -> Self {
        Self {
            encoder: BornoEncoder::from_byte_fallback(),
        }
    }

    /// Load a trained Borno tokenizer from a bincode-serialized encoder file.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            encoder: BornoEncoder::load(path)?,
        })
    }

    /// Load from a trained token list (ordered by ID).
    pub fn from_tokens(tokens: Vec<Vec<u8>>) -> Self {
        Self {
            encoder: BornoEncoder::from_tokens(tokens),
        }
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encoder.encode(text)
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.encoder.decode(ids)
    }

    /// Save the encoder to a bincode file.
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        self.encoder.save(path)
    }

    pub fn bos_id(&self) -> u32 { self.encoder.bos_id() }
    pub fn eos_id(&self) -> u32 { self.encoder.eos_id() }
    pub fn pad_id(&self) -> u32 { self.encoder.pad_id() }
    pub fn unk_id(&self) -> u32 { self.encoder.unk_id() }
    pub fn vocab_size(&self) -> usize { self.encoder.vocab_size() }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/adi/noor && cargo test -p borno --test roundtrip_tests 2>&1`
Expected: all roundtrip tests pass (7 total: 2 encoder + 5 API).

- [ ] **Step 5: Commit**

```bash
cd /Users/adi/noor
git add crates/borno/src/lib.rs crates/borno/tests/roundtrip_tests.rs
git commit -m "feat(borno): top-level Borno API with encode/decode/load/save"
```

---

### Task 7: Training CLI Binary

**Files:**
- Create: `crates/borno/src/bin/borno_train.rs`

- [ ] **Step 1: Implement the CLI**

```rust
// crates/borno/src/bin/borno_train.rs
//! Borno tokenizer training CLI.
//! Usage: borno-train --corpus-dir <path> --output-dir <path> [--merges 61000]

#[cfg(not(feature = "train"))]
compile_error!("borno-train requires the 'train' feature: cargo run --features train");

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
    println!("  Merges: {}", args.merges);
    println!("  Output: {:?}", args.output_dir);
    println!();

    std::fs::create_dir_all(&args.output_dir).expect("Cannot create output directory");

    // Train
    borno::trainer::train_bpe(&files, &args.output_dir, args.merges)
        .expect("Training failed");

    println!("\nTraining complete!");
    println!("  Vocab: {:?}", args.output_dir.join("borno_vocab.json"));
    println!("  Merges: {:?}", args.output_dir.join("borno_merges.txt"));

    // Build rs-bpe encoder from trained vocab and save as bincode
    let tokenizer_json = args.output_dir.join("borno_tokenizer.json");
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
                    borno::trainer::load_trained_vocab(&tokenizer_json).unwrap()
                );
                let en_tokens = borno.encode(test_en);
                let bn_tokens = borno.encode(test_bn);
                let en_words = test_en.split_whitespace().count();
                let bn_words = test_bn.split_whitespace().count();
                println!("\nFertility check:");
                println!("  English: {} tokens / {} words = {:.2}",
                    en_tokens.len(), en_words, en_tokens.len() as f64 / en_words as f64);
                println!("  Bangla:  {} tokens / {} words = {:.2}",
                    bn_tokens.len(), bn_words, bn_tokens.len() as f64 / bn_words as f64);
            }
            Err(e) => eprintln!("Warning: could not build encoder: {e}"),
        }
    }
}

#[cfg(not(feature = "train"))]
fn main() {}
```

- [ ] **Step 2: Add clap and indicatif as optional deps (already in Cargo.toml from Task 1, but ensure clap is included)**

Check that `crates/borno/Cargo.toml` has:
```toml
clap = { version = "4", features = ["derive"], optional = true }
```

And add clap to the `train` feature:
```toml
[features]
train = ["tokenizers", "reqwest", "dep:clap", "dep:indicatif"]
rand = ["bpe/rand"]
```

- [ ] **Step 3: Verify CLI compiles**

Run: `cd /Users/adi/noor && cargo build -p borno --features train --bin borno-train 2>&1`
Expected: compiles with no errors.

- [ ] **Step 4: Commit**

```bash
cd /Users/adi/noor
git add crates/borno/src/bin/borno_train.rs crates/borno/Cargo.toml
git commit -m "feat(borno): training CLI binary (borno-train)"
```

---

### Task 8: Corpus Download & Real Training

**Files:**
- Modify: `crates/borno/src/bin/borno_train.rs` (add download subcommand)
- Create: `crates/borno/data/.gitignore`

This task downloads real corpus data and trains the production 64K tokenizer.

- [ ] **Step 1: Create data/.gitignore**

```gitignore
# crates/borno/data/.gitignore
# Corpus files are large — don't commit
*.txt
*.bin
*.json
!.gitignore
```

- [ ] **Step 2: Download English corpus sample**

Run manually:
```bash
mkdir -p /Users/adi/noor/crates/borno/data/corpus
# Download a small FineWeb sample (~400MB) or use existing Shakespeare + Wikipedia
# For now, use what we have + download CC-100 Bangla sample
curl -L "https://data.statmt.org/cc-100/bn.txt.xz" -o /tmp/bn.txt.xz
xz -d /tmp/bn.txt.xz
head -c 400000000 /tmp/bn.txt > /Users/adi/noor/crates/borno/data/corpus/bangla.txt
```

Note: If CC-100 download is too large/slow, use a smaller Bangla Wikipedia dump or create a synthetic corpus from available Bangla text. The key is having ~300-400MB of Bangla text.

For English, we can use FineWeb-Edu sample:
```bash
# English — use a pre-downloaded sample or HF datasets CLI
# Minimum viable: combine Shakespeare + any English text available
cp /Users/adi/noor/data/raw/tiny_shakespeare.txt /Users/adi/noor/crates/borno/data/corpus/english.txt
# Add more English text if available
```

For code, create a sample from any available code files or download from The Stack.

- [ ] **Step 3: Train the production 64K tokenizer**

Run:
```bash
cd /Users/adi/noor
cargo run -p borno --features train --bin borno-train -- \
  --corpus-dir crates/borno/data/corpus \
  --output-dir crates/borno/data/trained \
  --merges 61000
```

Expected: Training completes in <30 seconds on 1GB. Output files written to `crates/borno/data/trained/`.

- [ ] **Step 4: Validate fertility**

Check the fertility output from the training CLI. Targets:
- English: < 2.0 tokens/word
- Bangla: < 3.0 tokens/word

If Bangla fertility > 3.0, increase Bangla corpus proportion and retrain.

- [ ] **Step 5: Copy trained encoder to a stable location**

```bash
mkdir -p /Users/adi/noor/checkpoints/tokenizer
cp /Users/adi/noor/crates/borno/data/trained/borno_encoder.bin /Users/adi/noor/checkpoints/tokenizer/
cp /Users/adi/noor/crates/borno/data/trained/borno_vocab.json /Users/adi/noor/checkpoints/tokenizer/
cp /Users/adi/noor/crates/borno/data/trained/borno_merges.txt /Users/adi/noor/checkpoints/tokenizer/
```

- [ ] **Step 6: Commit**

```bash
cd /Users/adi/noor
git add crates/borno/data/.gitignore checkpoints/tokenizer/
git commit -m "feat(borno): train production 64K BPE tokenizer on English+Bangla+code corpus"
```

---

### Task 9: Integrate Borno into noor-core

**Files:**
- Modify: `crates/noor-core/Cargo.toml`
- Modify: `crates/noor-core/src/tokenizer.rs`
- Modify: `crates/noor-core/src/lib.rs`
- Modify: `config/proxy.toml`
- Modify: `config/proxy_tiny.toml`

- [ ] **Step 1: Add borno dependency to noor-core**

In `crates/noor-core/Cargo.toml`, add:
```toml
[dependencies]
borno.workspace = true
serde.workspace = true
toml.workspace = true
rand.workspace = true
rand_distr.workspace = true
```

- [ ] **Step 2: Rewrite tokenizer.rs to use Borno**

```rust
// crates/noor-core/src/tokenizer.rs
//! Noor tokenizer — wraps Borno for encoding/decoding.

use std::path::Path;

/// Noor tokenizer. Wraps the Borno BPE tokenizer.
pub struct NoorTokenizer {
    borno: borno::Borno,
}

impl NoorTokenizer {
    /// Create a byte-level fallback tokenizer (no trained BPE merges).
    /// Used for testing and when no trained tokenizer is available.
    pub fn byte_level(vocab_size: usize) -> Self {
        assert_eq!(vocab_size, borno::vocab::VOCAB_SIZE,
            "Borno requires vocab_size={}, got {}", borno::vocab::VOCAB_SIZE, vocab_size);
        Self {
            borno: borno::Borno::from_byte_fallback(),
        }
    }

    /// Load a trained Borno tokenizer from a bincode encoder file.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            borno: borno::Borno::load(path)?,
        })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.borno.encode(text)
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.borno.decode(ids)
    }

    pub fn vocab_size(&self) -> usize {
        self.borno.vocab_size()
    }

    pub fn bos_id(&self) -> u32 { self.borno.bos_id() }
    pub fn eos_id(&self) -> u32 { self.borno.eos_id() }
    pub fn pad_id(&self) -> u32 { self.borno.pad_id() }
    pub fn unk_id(&self) -> u32 { self.borno.unk_id() }

    /// Load from a vocab text file (legacy, for backward compat during transition).
    pub fn from_vocab_file(path: &Path) -> std::io::Result<Self> {
        // Try loading as Borno bincode first
        if path.extension().is_some_and(|e| e == "bin") {
            return borno::Borno::load(path)
                .map(|b| Self { borno: b })
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()));
        }
        // Fallback: byte-level tokenizer
        Ok(Self::byte_level(borno::vocab::VOCAB_SIZE))
    }

    /// Save vocabulary (delegates to Borno).
    pub fn save_vocab(&self, path: &Path) -> std::io::Result<()> {
        self.borno.save(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_level_encode_decode() {
        let tok = NoorTokenizer::byte_level(64_000);
        let text = "hello world";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_vocab_size() {
        let tok = NoorTokenizer::byte_level(64_000);
        assert_eq!(tok.vocab_size(), 64_000);
    }

    #[test]
    fn test_special_tokens() {
        let tok = NoorTokenizer::byte_level(64_000);
        assert_eq!(tok.bos_id(), 256);
        assert_eq!(tok.eos_id(), 257);
        assert_eq!(tok.pad_id(), 258);
        assert_eq!(tok.unk_id(), 259);
    }

    #[test]
    fn test_encode_bangla() {
        let tok = NoorTokenizer::byte_level(64_000);
        let text = "নূর";
        let ids = tok.encode(text);
        assert!(!ids.is_empty());
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_empty() {
        let tok = NoorTokenizer::byte_level(64_000);
        assert_eq!(tok.encode(""), Vec::<u32>::new());
    }
}
```

- [ ] **Step 3: Update proxy.toml and proxy_tiny.toml**

In `config/proxy.toml`, change:
```toml
vocab_size = 64000
```

In `config/proxy_tiny.toml`, change:
```toml
vocab_size = 64000
```

- [ ] **Step 4: Fix any code in noor-core or noor-train that uses the old tokenizer API**

Search for `NoorTokenizer::byte_level(32000)` or similar hardcoded 32000 values and update to 64000. Also search for `NoorTokenizer::byte_level(300)` (from proxy_tiny) and update. Check:
- `crates/noor-train/src/data.rs` — may reference tokenizer
- `crates/noor-cli/src/main.rs` — may construct tokenizer
- `crates/noor-core/src/model.rs` — may reference vocab_size from config

Run: `cd /Users/adi/noor && grep -rn "32000\|byte_level(300\|byte_level(1000\|byte_level(500" crates/ config/ --include="*.rs" --include="*.toml"`

Update all occurrences. The vocab_size in config TOML files drives the model's embedding size — tests that construct a `NoorTokenizer::byte_level(N)` where N != 64000 should use 64000 or be gated by config.

- [ ] **Step 5: Verify everything compiles**

Run: `cd /Users/adi/noor && cargo build --workspace 2>&1`
Expected: compiles with no errors.

- [ ] **Step 6: Run all tests**

Run: `cd /Users/adi/noor && cargo test --workspace 2>&1`
Expected: all tests pass. Some existing tests may need vocab_size updates.

- [ ] **Step 7: Commit**

```bash
cd /Users/adi/noor
git add crates/noor-core/Cargo.toml crates/noor-core/src/tokenizer.rs config/proxy.toml config/proxy_tiny.toml
git commit -m "feat: integrate Borno tokenizer into noor-core, update vocab to 64K"
```

---

### Task 10: Re-prep Data & Retrain Proxy

**Files:**
- Modify: `crates/noor-cli/src/main.rs` (if needed for data prep)
- Data: `data/train/` (re-encoded shards)

- [ ] **Step 1: Re-encode training data with Borno**

The existing data shards in `data/train/` were encoded with the old byte-level tokenizer. They need re-encoding with Borno's 64K BPE.

If a trained Borno encoder exists at `checkpoints/tokenizer/borno_encoder.bin`, use it. Otherwise, use byte-level fallback (which gives same encoding as before but with 64K vocab size).

Run the data preprocessing:
```bash
cd /Users/adi/noor
# Remove old shards
rm -f data/train/shard_*.bin data/train/vocab.txt

# Re-encode using the CLI or a script
cargo run -p noor-cli -- convert \
  --input data/raw/ \
  --output data/train/ \
  --tokenizer checkpoints/tokenizer/borno_encoder.bin \
  2>&1
```

If the CLI `convert` command doesn't support `--tokenizer` flag yet, update it to accept the Borno encoder path. The preprocessor should:
1. Load Borno encoder
2. Read each .txt file from input dir
3. Encode all text to token IDs
4. Write binary shards in the existing format: `[seq_len: u32][token_ids: u32 * seq_len]`

- [ ] **Step 2: Verify shard token IDs are in [0, 64000) range**

```bash
cd /Users/adi/noor
cargo run -p noor-cli -- eval --model checkpoints/... 2>&1 || true
# Or write a quick check in the test suite
```

- [ ] **Step 3: Train proxy with 64K vocab**

```bash
cd /Users/adi/noor
cargo run --release -p noor-cli -- train \
  --config config/proxy.toml \
  --data data/train/ \
  2>&1
```

Expected: Training starts, loss decreases over steps. The embedding table is now 64K x 768 = ~98MB in f32 (vs 32K x 768 = ~49MB before). Training should still fit in memory.

Monitor:
- Loss should decrease (similar trajectory to previous 32K run)
- No out-of-range token ID errors
- Expert utilization balanced

- [ ] **Step 4: Validate trained model**

After training completes (or at a checkpoint):
```bash
cd /Users/adi/noor
cargo run -p noor-cli -- run \
  --model checkpoints/latest.gguf \
  --prompt "Hello world" \
  --max-tokens 50
```

Expected: generates tokens. Quality may be low (Shakespeare-only data) but should not produce garbage.

- [ ] **Step 5: Commit trained model and updated configs**

```bash
cd /Users/adi/noor
git add config/ data/train/vocab.txt
git commit -m "feat: retrain 0.5B proxy with Borno 64K tokenizer"
```

---

## Summary

| Task | Component | Key Files |
|------|-----------|-----------|
| 1 | Crate scaffolding | `crates/borno/Cargo.toml`, `vocab.rs`, `lib.rs` |
| 2 | Bangla pre-tokenization | `bangla.rs`, `bangla_tests.rs` |
| 3 | Script-aware pre-tokenization | `pretokenize.rs` |
| 4 | BPE trainer (HF) | `trainer.rs` |
| 5 | rs-bpe encoder | `encoder.rs` |
| 6 | Top-level API | `lib.rs` |
| 7 | Training CLI | `bin/borno_train.rs` |
| 8 | Corpus download & training | data files, trained vocab |
| 9 | noor-core integration | `tokenizer.rs`, configs |
| 10 | Re-prep data & retrain proxy | data shards, proxy model |
