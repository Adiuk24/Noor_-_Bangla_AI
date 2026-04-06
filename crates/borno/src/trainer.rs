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

/// Load a trained HuggingFace tokenizer and extract an ordered token list
/// suitable for constructing a `BytePairEncoding` via `from_dictionary`.
///
/// Returns tokens ordered by their ID (ascending).
pub fn load_trained_vocab(
    tokenizer_path: &Path,
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

    let vocab = tokenizer.get_vocab(true);
    let vocab_size = tokenizer.get_vocab_size(true);

    // Sort tokens by ID to get an ordered list.
    let mut pairs: Vec<(String, u32)> = vocab.into_iter().collect();
    pairs.sort_by_key(|(_tok, id)| *id);

    let mut tokens: Vec<Vec<u8>> = Vec::with_capacity(vocab_size);
    for (tok_str, _id) in pairs {
        tokens.push(tok_str.into_bytes());
    }

    Ok(tokens)
}
