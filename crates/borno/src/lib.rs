pub mod vocab;
pub mod bangla;
pub mod pretokenize;
pub mod encoder;

#[cfg(feature = "train")]
pub mod trainer;

use std::path::Path;

use encoder::BornoEncoder;

/// Top-level Borno tokenizer API.
///
/// Wraps `BornoEncoder` with a convenient interface for encoding/decoding text.
pub struct Borno {
    encoder: BornoEncoder,
}

impl Borno {
    /// Create a byte-fallback-only tokenizer (no BPE merges).
    pub fn from_byte_fallback() -> Self {
        Borno {
            encoder: BornoEncoder::from_byte_fallback(),
        }
    }

    /// Load a tokenizer from a bincode-serialized file.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Borno {
            encoder: BornoEncoder::load(path)?,
        })
    }

    /// Create a tokenizer from a trained token list.
    pub fn from_tokens(tokens: Vec<Vec<u8>>) -> Self {
        Borno {
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

    /// Save the tokenizer to a bincode file.
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        self.encoder.save(path)
    }

    /// Beginning-of-sequence token ID.
    pub fn bos_id(&self) -> u32 {
        self.encoder.bos_id()
    }

    /// End-of-sequence token ID.
    pub fn eos_id(&self) -> u32 {
        self.encoder.eos_id()
    }

    /// Padding token ID.
    pub fn pad_id(&self) -> u32 {
        self.encoder.pad_id()
    }

    /// Unknown token ID.
    pub fn unk_id(&self) -> u32 {
        self.encoder.unk_id()
    }

    /// Total vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.encoder.vocab_size()
    }
}
