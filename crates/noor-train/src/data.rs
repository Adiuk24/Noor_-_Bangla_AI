//! Pre-tokenized binary shard data pipeline.
//!
//! Shard format: consecutive token sequences, each prefixed with length.
//!   [seq_len: u32][token_ids: u32 * seq_len]
//!   [seq_len: u32][token_ids: u32 * seq_len]
//!   ...
//!
//! DataLoader reads shards via mmap for zero-copy loading.

use memmap2::Mmap;
use std::path::{Path, PathBuf};
use std::io::{self, Write};

/// A memory-mapped data shard.
pub struct DataShard {
    mmap: Mmap,
    /// Total number of tokens in this shard
    pub total_tokens: usize,
}

impl DataShard {
    /// Open a shard file.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Count total tokens
        let mut offset = 0;
        let mut total_tokens = 0;
        let data = &mmap[..];
        while offset + 4 <= data.len() {
            let seq_len = u32::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
            ]) as usize;
            offset += 4;
            if offset + seq_len * 4 > data.len() {
                break;
            }
            total_tokens += seq_len;
            offset += seq_len * 4;
        }

        Ok(Self { mmap, total_tokens })
    }

    /// Read all sequences from this shard as flat token array.
    pub fn read_all_tokens(&self) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(self.total_tokens);
        let data = &self.mmap[..];
        let mut offset = 0;

        while offset + 4 <= data.len() {
            let seq_len = u32::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
            ]) as usize;
            offset += 4;
            if offset + seq_len * 4 > data.len() {
                break;
            }
            for i in 0..seq_len {
                let start = offset + i * 4;
                let token = u32::from_le_bytes([
                    data[start], data[start + 1], data[start + 2], data[start + 3],
                ]);
                tokens.push(token);
            }
            offset += seq_len * 4;
        }

        tokens
    }
}

/// Write tokenized data to a shard file.
pub fn write_shard(path: &Path, sequences: &[Vec<u32>]) -> io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    for seq in sequences {
        let len = seq.len() as u32;
        file.write_all(&len.to_le_bytes())?;
        for &token in seq {
            file.write_all(&token.to_le_bytes())?;
        }
    }
    Ok(())
}

/// A training batch: input tokens and target tokens.
pub struct Batch {
    /// Input token IDs: (batch_size, context_length)
    pub input_ids: Vec<Vec<u32>>,
    /// Target token IDs: (batch_size, context_length) — shifted by 1
    pub target_ids: Vec<Vec<u32>>,
}

/// DataLoader that reads from multiple shards and produces batches.
pub struct DataLoader {
    /// All tokens from all shards, concatenated
    tokens: Vec<u32>,
    /// Current position in the token stream
    position: usize,
    /// Context length for each sequence
    context_length: usize,
    /// Number of sequences per batch
    batch_size: usize,
}

impl DataLoader {
    /// Create a loader from a directory of shard files.
    pub fn from_shard_dir(
        dir: &Path,
        context_length: usize,
        batch_size: usize,
    ) -> io::Result<Self> {
        let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "bin"))
            .map(|e| e.path())
            .collect();
        shard_paths.sort();

        let mut tokens = Vec::new();
        for path in &shard_paths {
            let shard = DataShard::open(path)?;
            tokens.extend_from_slice(&shard.read_all_tokens());
        }

        if tokens.is_empty() {
            return Err(io::Error::new(io::ErrorKind::NotFound, "No tokens in shard directory"));
        }

        Ok(Self {
            tokens,
            position: 0,
            context_length,
            batch_size,
        })
    }

    /// Create a loader from raw token data (for testing).
    pub fn from_tokens(tokens: Vec<u32>, context_length: usize, batch_size: usize) -> Self {
        Self {
            tokens,
            position: 0,
            context_length,
            batch_size,
        }
    }

    /// Total number of tokens available.
    pub fn total_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Get the next batch. Returns None when data is exhausted (one epoch).
    pub fn next_batch(&mut self) -> Option<Batch> {
        let seq_len = self.context_length + 1; // +1 for target shift
        let needed = self.batch_size * seq_len;

        if self.position + needed > self.tokens.len() {
            // Wrap around to start (new epoch)
            self.position = 0;
            if needed > self.tokens.len() {
                return None; // not enough data for even one batch
            }
        }

        let mut input_ids = Vec::with_capacity(self.batch_size);
        let mut target_ids = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            let chunk = &self.tokens[self.position..self.position + seq_len];
            input_ids.push(chunk[..self.context_length].to_vec());
            target_ids.push(chunk[1..].to_vec());
            self.position += seq_len;
        }

        Some(Batch { input_ids, target_ids })
    }

    /// Reset to beginning of data.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Current position as fraction of total data.
    pub fn progress(&self) -> f32 {
        self.position as f32 / self.tokens.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_read_shard() {
        let path = std::env::temp_dir().join("noor_test_shard.bin");
        let sequences = vec![
            vec![1u32, 2, 3, 4, 5],
            vec![10, 20, 30],
            vec![100, 200, 300, 400],
        ];
        write_shard(&path, &sequences).unwrap();

        let shard = DataShard::open(&path).unwrap();
        assert_eq!(shard.total_tokens, 12); // 5 + 3 + 4

        let all = shard.read_all_tokens();
        assert_eq!(all, vec![1, 2, 3, 4, 5, 10, 20, 30, 100, 200, 300, 400]);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_data_loader_batches() {
        // 100 tokens, context=4, batch=2 → each batch needs 2 * (4+1) = 10 tokens
        let tokens: Vec<u32> = (0..100).collect();
        let mut loader = DataLoader::from_tokens(tokens, 4, 2);

        assert_eq!(loader.total_tokens(), 100);

        let batch = loader.next_batch().unwrap();
        assert_eq!(batch.input_ids.len(), 2);
        assert_eq!(batch.input_ids[0].len(), 4);
        assert_eq!(batch.target_ids[0].len(), 4);

        // Input is tokens[0..4], target is tokens[1..5]
        assert_eq!(batch.input_ids[0], vec![0, 1, 2, 3]);
        assert_eq!(batch.target_ids[0], vec![1, 2, 3, 4]);

        // Second sequence in batch
        assert_eq!(batch.input_ids[1], vec![5, 6, 7, 8]);
        assert_eq!(batch.target_ids[1], vec![6, 7, 8, 9]);
    }

    #[test]
    fn test_data_loader_wraps() {
        let tokens: Vec<u32> = (0..20).collect();
        let mut loader = DataLoader::from_tokens(tokens, 4, 2);

        // Each batch consumes 10 tokens, we have 20 → 2 batches before wrap
        let _b1 = loader.next_batch().unwrap();
        let _b2 = loader.next_batch().unwrap();
        // Third batch wraps
        let b3 = loader.next_batch().unwrap();
        assert_eq!(b3.input_ids[0], vec![0, 1, 2, 3]); // wrapped to start
    }

    #[test]
    fn test_data_loader_progress() {
        let tokens: Vec<u32> = (0..100).collect();
        let mut loader = DataLoader::from_tokens(tokens, 4, 2);
        assert_eq!(loader.progress(), 0.0);
        loader.next_batch();
        assert!(loader.progress() > 0.0);
    }

    #[test]
    fn test_shard_dir_loading() {
        let dir = std::env::temp_dir().join("noor_test_shards");
        std::fs::create_dir_all(&dir).unwrap();

        // Write two shards
        write_shard(
            &dir.join("shard_000.bin"),
            &[vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        ).unwrap();
        write_shard(
            &dir.join("shard_001.bin"),
            &[vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20]],
        ).unwrap();

        let mut loader = DataLoader::from_shard_dir(&dir, 4, 1).unwrap();
        assert_eq!(loader.total_tokens(), 20);

        let batch = loader.next_batch().unwrap();
        assert_eq!(batch.input_ids[0].len(), 4);

        // Cleanup
        std::fs::remove_dir_all(dir).ok();
    }
}
