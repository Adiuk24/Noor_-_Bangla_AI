//! Noor tokenizer — wraps Borno for encoding/decoding.

use std::path::Path;
use std::io;

/// Noor tokenizer wrapping the Borno BPE tokenizer.
pub struct NoorTokenizer {
    borno: borno::Borno,
}

impl NoorTokenizer {
    /// Create a byte-level fallback tokenizer (no trained BPE merges).
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

    /// Load from file path (supports .bin for Borno encoder, falls back to byte-level).
    pub fn from_vocab_file(path: &Path) -> io::Result<Self> {
        if path.extension().is_some_and(|e| e == "bin") {
            return borno::Borno::load(path)
                .map(|b| Self { borno: b })
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()));
        }
        Ok(Self::byte_level(borno::vocab::VOCAB_SIZE))
    }

    /// Save vocabulary.
    pub fn save_vocab(&self, path: &Path) -> io::Result<()> {
        self.borno.save(path)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))
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
