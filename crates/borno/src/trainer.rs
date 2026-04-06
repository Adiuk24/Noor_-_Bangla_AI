//! BPE trainer wrapping the HuggingFace `tokenizers` crate.
//!
//! Feature-gated behind `train`.

use std::path::Path;

use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::unicode::NFC;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::{AddedToken, Model, TokenizerBuilder};

use crate::vocab;

/// Train a BPE tokenizer from corpus files and save the results.
///
/// Creates three output files in `output_dir`:
/// - `tokenizer.json` — full HuggingFace tokenizer (for reloading)
/// - `borno_vocab.json` — vocab mapping (token -> id)
/// - `borno_merges.txt` — merge rules
///
/// `n_merges` controls how many BPE merges to learn (total vocab = 256 byte tokens + n_merges).
pub fn train_bpe(
    files: &[String],
    output_dir: &Path,
    n_merges: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let vocab_size = 256 + n_merges;

    // Build special tokens list from our vocab constants.
    let special_tokens: Vec<AddedToken> = vocab::SPECIAL_TOKENS
        .iter()
        .map(|&s| AddedToken::from(s.to_string(), true))
        .collect();

    // Build the BPE trainer.
    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(false)
        .vocab_size(vocab_size)
        .min_frequency(2)
        .special_tokens(special_tokens)
        .build();

    // Build the tokenizer pipeline: NFC normalizer + ByteLevel pre-tokenizer.
    // We use wrapper types so that TokenizerImpl can serialize/deserialize properly.
    let nfc_normalizer: NormalizerWrapper = NFC.into();
    let byte_level_pre: PreTokenizerWrapper = ByteLevel::default().into();
    let byte_level_post: PostProcessorWrapper = ByteLevel::default().into();
    let byte_level_dec: DecoderWrapper = ByteLevel::default().into();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(nfc_normalizer))
        .with_pre_tokenizer(Some(byte_level_pre))
        .with_post_processor(Some(byte_level_post))
        .with_decoder(Some(byte_level_dec))
        .build()
        .map_err(|e| format!("Failed to build tokenizer: {e}"))?;

    // Train from files.
    let file_list: Vec<String> = files.to_vec();
    tokenizer
        .train_from_files(&mut trainer, file_list)
        .map_err(|e| format!("Training failed: {e}"))?;

    // Save full tokenizer JSON.
    tokenizer
        .save(output_dir.join("tokenizer.json"), true)
        .map_err(|e| format!("Failed to save tokenizer.json: {e}"))?;

    // Save vocab and merges via the model's own save method.
    tokenizer
        .get_model()
        .save(output_dir, Some("borno"))
        .map_err(|e| format!("Failed to save vocab/merges: {e}"))?;

    Ok(())
}

/// Build the GPT-2 ByteLevel char-to-byte reverse mapping.
/// The HF ByteLevel pre-tokenizer maps each byte (0-255) to a unicode char.
/// This function builds the reverse: char -> byte.
fn build_char_to_byte_map() -> std::collections::HashMap<char, u8> {
    let mut bs: Vec<u8> = Vec::new();
    bs.extend(b'!'..=b'~');
    bs.extend(0xA1u8..=0xACu8);
    bs.extend(0xAEu8..=0xFFu8);

    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n: u32 = 0;

    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    cs.into_iter()
        .zip(bs)
        .map(|(c, b)| (char::from_u32(c).unwrap(), b))
        .collect()
}

/// Convert a HuggingFace ByteLevel token string back to raw bytes.
fn hf_token_to_bytes(token: &str, char_to_byte: &std::collections::HashMap<char, u8>) -> Vec<u8> {
    token.chars().map(|c| char_to_byte.get(&c).copied().unwrap_or(c as u8)).collect()
}

/// Load a trained HuggingFace tokenizer and extract an ordered token list
/// suitable for constructing a `BytePairEncoding` via `from_dictionary`.
///
/// Returns only BPE-encodable tokens (byte fallback + learned merges) as raw
/// byte sequences, ordered by merge priority. Special tokens are excluded
/// since they're handled externally by `BornoEncoder`.
pub fn load_trained_vocab(
    tokenizer_path: &Path,
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

    let char_to_byte = build_char_to_byte_map();
    let vocab = tokenizer.get_vocab(true);

    // Collect special token strings to filter them out.
    let special_set: std::collections::HashSet<&str> =
        vocab::SPECIAL_TOKENS.iter().copied().collect();

    // Sort tokens by ID to get merge-priority order.
    let mut pairs: Vec<(String, u32)> = vocab.into_iter().collect();
    pairs.sort_by_key(|(_tok, id)| *id);

    // Convert HF ByteLevel unicode tokens to raw bytes, skipping specials.
    let mut tokens: Vec<Vec<u8>> = Vec::new();
    for (tok_str, _id) in &pairs {
        if special_set.contains(tok_str.as_str()) {
            continue;
        }
        tokens.push(hf_token_to_bytes(tok_str, &char_to_byte));
    }

    Ok(tokens)
}
