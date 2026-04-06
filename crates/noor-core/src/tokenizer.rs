//! Simple BPE tokenizer for Phase 0.
//! Uses a basic byte-level fallback. For production, will wrap HuggingFace tokenizers crate.

use std::collections::HashMap;
use std::path::Path;
use std::io::{self, BufRead, Write};

/// Noor tokenizer. Phase 0: simple char-level with special tokens.
/// Phase 2+: proper BPE from HuggingFace tokenizers crate.
pub struct NoorTokenizer {
    /// Token string → ID (used for BPE merge lookups in Phase 2+)
    #[allow(dead_code)]
    token_to_id: HashMap<String, u32>,
    /// ID → token string
    id_to_token: Vec<String>,
    /// Special tokens
    pub bos_id: u32,
    pub eos_id: u32,
    pub pad_id: u32,
    pub unk_id: u32,
}

impl NoorTokenizer {
    /// Create a simple byte-level tokenizer with vocab_size entries.
    /// First 256 entries are raw bytes, rest are special/unused.
    pub fn byte_level(vocab_size: usize) -> Self {
        assert!(vocab_size >= 260, "Vocab must be >= 260 for byte-level + specials");
        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::with_capacity(vocab_size);

        // Byte tokens 0-255
        for i in 0..256u32 {
            let token = format!("<byte_{i:02x}>");
            token_to_id.insert(token.clone(), i);
            id_to_token.push(token);
        }

        // Special tokens
        let specials = ["<bos>", "<eos>", "<pad>", "<unk>"];
        for (j, &s) in specials.iter().enumerate() {
            let id = 256 + j as u32;
            token_to_id.insert(s.to_string(), id);
            id_to_token.push(s.to_string());
        }

        // Fill remaining with unused tokens
        for i in 260..vocab_size {
            let token = format!("<unused_{i}>");
            token_to_id.insert(token.clone(), i as u32);
            id_to_token.push(token);
        }

        Self {
            token_to_id,
            id_to_token,
            bos_id: 256,
            eos_id: 257,
            pad_id: 258,
            unk_id: 259,
        }
    }

    /// Load a vocabulary from a simple text file (one token per line).
    pub fn from_vocab_file(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = io::BufReader::new(file);
        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::new();

        for (i, line) in reader.lines().enumerate() {
            let token = line?;
            token_to_id.insert(token.clone(), i as u32);
            id_to_token.push(token);
        }

        let get_id = |name: &str| -> u32 {
            *token_to_id.get(name).unwrap_or(&0)
        };

        Ok(Self {
            bos_id: get_id("<bos>"),
            eos_id: get_id("<eos>"),
            pad_id: get_id("<pad>"),
            unk_id: get_id("<unk>"),
            token_to_id,
            id_to_token,
        })
    }

    /// Encode text to token IDs (byte-level encoding).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter()
            .filter(|&&id| id < 256) // skip special tokens
            .map(|&id| id as u8)
            .collect();
        String::from_utf8_lossy(&bytes).to_string()
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Save vocabulary to file.
    pub fn save_vocab(&self, path: &Path) -> io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        for token in &self.id_to_token {
            writeln!(file, "{token}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_level_encode_decode() {
        let tok = NoorTokenizer::byte_level(1000);
        let text = "hello world";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text, "Round-trip encode/decode should preserve text");
    }

    #[test]
    fn test_vocab_size() {
        let tok = NoorTokenizer::byte_level(32000);
        assert_eq!(tok.vocab_size(), 32000);
    }

    #[test]
    fn test_special_tokens() {
        let tok = NoorTokenizer::byte_level(1000);
        assert_eq!(tok.bos_id, 256);
        assert_eq!(tok.eos_id, 257);
        assert_eq!(tok.pad_id, 258);
        assert_eq!(tok.unk_id, 259);
    }

    #[test]
    fn test_encode_empty() {
        let tok = NoorTokenizer::byte_level(1000);
        assert_eq!(tok.encode(""), Vec::<u32>::new());
    }

    #[test]
    fn test_encode_unicode() {
        let tok = NoorTokenizer::byte_level(1000);
        let text = "নূর"; // Bangla for "Noor"
        let ids = tok.encode(text);
        assert!(!ids.is_empty());
        // Each UTF-8 byte becomes a token; Bangla chars are 3 bytes each
        assert_eq!(ids.len(), text.len()); // byte length
    }

    #[test]
    fn test_save_load_vocab() {
        let tok = NoorTokenizer::byte_level(500);
        let path = std::env::temp_dir().join("noor_test_vocab.txt");
        tok.save_vocab(&path).unwrap();
        let loaded = NoorTokenizer::from_vocab_file(&path).unwrap();
        assert_eq!(loaded.vocab_size(), 500);
        std::fs::remove_file(path).ok();
    }
}
