//! rs-bpe backtracking encoder for Borno.
//!
//! Wraps the `bpe` crate's `BytePairEncoding` for fast, optimal BPE encoding
//! with special token handling and pretokenization.
//!
//! The BPE dictionary contains only tokens that the BPE algorithm can actually
//! produce (byte fallback tokens + learned merges). Special tokens, reserved
//! tokens, and placeholders are handled externally by `BornoEncoder`.

use std::path::Path;

use bpe::byte_pair_encoding::BytePairEncoding;
use serde::{Deserialize, Serialize};

use crate::pretokenize;
use crate::vocab;

/// A special token entry: the string form and its token ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpecialToken {
    text: String,
    id: u32,
}

/// The Borno encoder: wraps `BytePairEncoding` with special token support
/// and Bangla-aware pretokenization.
///
/// The BPE engine only knows about byte tokens (0-255) and learned merges.
/// Special tokens (BOS, EOS, etc.) are handled by splitting them out before
/// BPE encoding and mapping them to fixed IDs.
#[derive(Serialize, Deserialize)]
pub struct BornoEncoder {
    bpe: BytePairEncoding,
    special_tokens: Vec<SpecialToken>,
    total_vocab_size: usize,
}

impl BornoEncoder {
    /// Create an encoder with only byte-fallback tokens (no BPE merges).
    ///
    /// The BPE dictionary contains just the 256 single-byte tokens.
    /// Special tokens are handled externally. Total vocab_size is reported
    /// as 64K to match the Borno vocab layout.
    pub fn from_byte_fallback() -> Self {
        // Only the 256 byte tokens go into the BPE dictionary.
        let byte_tokens: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let bpe = BytePairEncoding::from_dictionary(byte_tokens, None);
        let special_tokens = build_special_tokens();

        BornoEncoder {
            bpe,
            special_tokens,
            total_vocab_size: vocab::VOCAB_SIZE,
        }
    }

    /// Create an encoder from a trained token list.
    ///
    /// `tokens` should contain the BPE-encodable tokens ordered by ID
    /// (typically 256 byte tokens followed by learned merges).
    /// Special tokens are handled externally and should NOT be in this list.
    pub fn from_tokens(tokens: Vec<Vec<u8>>) -> Self {
        let bpe = BytePairEncoding::from_dictionary(tokens, None);
        let special_tokens = build_special_tokens();

        BornoEncoder {
            bpe,
            special_tokens,
            total_vocab_size: vocab::VOCAB_SIZE,
        }
    }

    /// Load a serialized encoder from a bincode file.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        let enc: BornoEncoder = bincode::deserialize(&data)?;
        Ok(enc)
    }

    /// Save the encoder to a bincode file.
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let data = bincode::serialize(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Encode text to token IDs.
    ///
    /// 1. Split on special token boundaries (e.g. `<bos>`, `<eos>`)
    /// 2. For non-special segments: NFC normalize, pretokenize, then BPE encode
    /// 3. For special tokens: emit their fixed ID
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        let segments = split_on_special_tokens(text, &self.special_tokens);
        let mut ids = Vec::new();

        for seg in segments {
            match seg {
                Segment::Special(id) => {
                    ids.push(id);
                }
                Segment::Text(t) => {
                    let pre_tokens = pretokenize::pretokenize(&t);
                    for pt in &pre_tokens {
                        let encoded = self.bpe.encode_via_backtracking(pt.as_bytes());
                        ids.extend(encoded);
                    }
                }
            }
        }

        ids
    }

    /// Decode token IDs back to a string.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        if token_ids.is_empty() {
            return String::new();
        }

        let mut result = Vec::new();
        for &id in token_ids {
            // Check if it's a special token.
            if let Some(st) = self.special_tokens.iter().find(|s| s.id == id) {
                result.extend_from_slice(st.text.as_bytes());
            } else if (id as usize) < self.bpe.num_tokens() {
                result.extend_from_slice(self.bpe.token_bytes(id));
            }
            // IDs outside range (reserved, placeholder) are silently skipped.
        }

        String::from_utf8_lossy(&result).into_owned()
    }

    /// Beginning-of-sequence token ID.
    pub fn bos_id(&self) -> u32 {
        vocab::BOS_ID
    }

    /// End-of-sequence token ID.
    pub fn eos_id(&self) -> u32 {
        vocab::EOS_ID
    }

    /// Padding token ID.
    pub fn pad_id(&self) -> u32 {
        vocab::PAD_ID
    }

    /// Unknown token ID.
    pub fn unk_id(&self) -> u32 {
        vocab::UNK_ID
    }

    /// Total vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.total_vocab_size
    }
}

// --- Internal helpers ---

fn build_special_tokens() -> Vec<SpecialToken> {
    vocab::SPECIAL_TOKENS
        .iter()
        .enumerate()
        .map(|(i, &text)| SpecialToken {
            text: text.to_string(),
            id: vocab::BOS_ID + i as u32,
        })
        .collect()
}

#[derive(Debug)]
enum Segment {
    Special(u32),
    Text(String),
}

/// Split input text on special token boundaries.
/// Special tokens are matched literally and greedily (longest first).
fn split_on_special_tokens(text: &str, specials: &[SpecialToken]) -> Vec<Segment> {
    if text.is_empty() {
        return vec![];
    }

    // Sort specials by descending length for longest-match-first.
    let mut sorted: Vec<&SpecialToken> = specials.iter().collect();
    sorted.sort_by(|a, b| b.text.len().cmp(&a.text.len()));

    let mut segments = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        // Try to match a special token at the current position.
        let mut matched = false;
        for st in &sorted {
            if remaining.starts_with(&st.text) {
                segments.push(Segment::Special(st.id));
                remaining = &remaining[st.text.len()..];
                matched = true;
                break;
            }
        }
        if !matched {
            // Find how far until the next special token (or end).
            let end;
            let mut char_iter = remaining.char_indices().skip(1);
            loop {
                match char_iter.next() {
                    Some((idx, _)) => {
                        let at = &remaining[idx..];
                        if sorted.iter().any(|st| at.starts_with(&st.text)) {
                            end = idx;
                            break;
                        }
                    }
                    None => {
                        end = remaining.len();
                        break;
                    }
                }
            }
            segments.push(Segment::Text(remaining[..end].to_string()));
            remaining = &remaining[end..];
        }
    }

    segments
}
