//! Data pipeline — reads pre-tokenized binary shards into Burn tensors.
//!
//! Shard format: [seq_len: u32][token_ids: u32 * seq_len] repeated.
//! Compatible with borno-shard output.

use burn::prelude::*;
use memmap2::Mmap;
use std::path::{Path, PathBuf};

/// All tokens loaded from shards, ready for batching.
pub struct ShardDataset {
    tokens: Vec<u32>,
    context_length: usize,
    position: usize,
}

impl ShardDataset {
    pub fn from_shard_dir(dir: &Path, context_length: usize) -> std::io::Result<Self> {
        let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "bin"))
            .map(|e| e.path())
            .collect();
        shard_paths.sort();

        let mut tokens = Vec::new();
        for path in &shard_paths {
            let file = std::fs::File::open(path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            let data = &mmap[..];
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
        }

        eprintln!("Loaded {} tokens from {} shards", tokens.len(), shard_paths.len());

        Ok(Self {
            tokens,
            context_length,
            position: 0,
        })
    }

    pub fn total_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Get next batch as (input_ids, target_ids) Burn tensors.
    /// input_ids: [batch, context_length], target_ids: [batch, context_length]
    pub fn next_batch<B: Backend>(
        &mut self,
        batch_size: usize,
        device: &B::Device,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
        let seq_len = self.context_length + 1; // +1 for target shift
        let needed = batch_size * seq_len;

        // Wrap around if not enough data
        if self.position + needed > self.tokens.len() {
            self.position = 0;
        }

        let mut input_data = Vec::with_capacity(batch_size * self.context_length);
        let mut target_data = Vec::with_capacity(batch_size * self.context_length);

        for b in 0..batch_size {
            let start = self.position + b * seq_len;
            for t in 0..self.context_length {
                input_data.push(self.tokens[start + t] as i64);
                target_data.push(self.tokens[start + t + 1] as i64);
            }
        }

        self.position += needed;

        let input_ids = Tensor::<B, 1, Int>::from_ints(input_data.as_slice(), device)
            .reshape([batch_size, self.context_length]);
        let target_ids = Tensor::<B, 1, Int>::from_ints(target_data.as_slice(), device)
            .reshape([batch_size, self.context_length]);

        (input_ids, target_ids)
    }

    pub fn reset(&mut self) {
        self.position = 0;
    }

    pub fn progress(&self) -> f32 {
        self.position as f32 / self.tokens.len() as f32
    }
}
