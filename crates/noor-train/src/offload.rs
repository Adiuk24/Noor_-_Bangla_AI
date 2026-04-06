//! Expert SSD/RAM offload manager.
//! Only active experts live in memory. Inactive experts are on SSD.
//! LRU cache with 2x prefetch for next batch's predicted experts.

use noor_core::tensor::Tensor;
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

/// Manages offloading expert weights between memory and disk.
pub struct ExpertOffloader {
    /// Experts currently in memory: (layer_idx, expert_idx) → weights
    cache: HashMap<(usize, usize), ExpertWeights>,
    /// LRU order: most recently used at the back
    lru_order: Vec<(usize, usize)>,
    /// Maximum experts to keep in memory
    max_cached: usize,
    /// Directory for expert weight files
    disk_dir: PathBuf,
    /// Stats
    pub hits: usize,
    pub misses: usize,
}

/// Weights for a single expert (SwiGLU: gate + up + down).
#[derive(Clone)]
pub struct ExpertWeights {
    pub w_gate: Tensor,
    pub w_up: Tensor,
    pub w_down: Tensor,
}

impl ExpertWeights {
    pub fn memory_bytes(&self) -> usize {
        (self.w_gate.numel() + self.w_up.numel() + self.w_down.numel()) * 4 // f32
    }
}

impl ExpertOffloader {
    /// Create offloader.
    /// max_cached: max experts in memory (e.g., 5 for 4 active + 1 prefetch).
    /// disk_dir: directory to store expert weight files.
    pub fn new(max_cached: usize, disk_dir: &Path) -> io::Result<Self> {
        std::fs::create_dir_all(disk_dir)?;
        Ok(Self {
            cache: HashMap::new(),
            lru_order: Vec::new(),
            max_cached,
            disk_dir: disk_dir.to_path_buf(),
            hits: 0,
            misses: 0,
        })
    }

    /// Store an expert's weights (initially called to populate all experts to disk).
    pub fn store(&mut self, layer: usize, expert: usize, weights: &ExpertWeights) -> io::Result<()> {
        let path = self.expert_path(layer, expert);
        let mut data = Vec::new();
        // Simple binary format: gate_data | up_data | down_data with shape headers
        write_tensor_to_buf(&mut data, &weights.w_gate);
        write_tensor_to_buf(&mut data, &weights.w_up);
        write_tensor_to_buf(&mut data, &weights.w_down);
        std::fs::write(&path, &data)?;

        // Also cache if we have room
        if self.cache.len() < self.max_cached {
            self.cache_insert(layer, expert, weights.clone());
        }

        Ok(())
    }

    /// Get expert weights. Returns from cache if available, loads from disk otherwise.
    pub fn get(&mut self, layer: usize, expert: usize) -> io::Result<ExpertWeights> {
        let key = (layer, expert);

        if let Some(weights) = self.cache.get(&key) {
            self.hits += 1;
            // Move to back of LRU
            self.lru_order.retain(|k| *k != key);
            self.lru_order.push(key);
            return Ok(weights.clone());
        }

        // Cache miss — load from disk
        self.misses += 1;
        let weights = self.load_from_disk(layer, expert)?;

        // Evict LRU if cache full
        while self.cache.len() >= self.max_cached {
            if let Some(evict_key) = self.lru_order.first().cloned() {
                self.cache.remove(&evict_key);
                self.lru_order.remove(0);
            } else {
                break;
            }
        }

        self.cache_insert(layer, expert, weights.clone());
        Ok(weights)
    }

    /// Prefetch experts that will be needed soon (called after routing decides).
    /// This is synchronous for Phase 0. Phase 1 makes it async with tokio.
    pub fn prefetch(&mut self, keys: &[(usize, usize)]) -> io::Result<()> {
        for &(layer, expert) in keys {
            if !self.cache.contains_key(&(layer, expert)) {
                // Load into cache (will evict LRU if needed)
                let _ = self.get(layer, expert)?;
            }
        }
        Ok(())
    }

    /// Number of experts currently in memory.
    pub fn cached_count(&self) -> usize {
        self.cache.len()
    }

    /// Cache hit rate.
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 { return 0.0; }
        self.hits as f32 / total as f32
    }

    fn expert_path(&self, layer: usize, expert: usize) -> PathBuf {
        self.disk_dir.join(format!("expert_L{layer}_E{expert}.bin"))
    }

    fn cache_insert(&mut self, layer: usize, expert: usize, weights: ExpertWeights) {
        let key = (layer, expert);
        self.lru_order.retain(|k| *k != key);
        self.lru_order.push(key);
        self.cache.insert(key, weights);
    }

    fn load_from_disk(&self, layer: usize, expert: usize) -> io::Result<ExpertWeights> {
        let path = self.expert_path(layer, expert);
        let data = std::fs::read(&path)?;
        let mut offset = 0;
        let w_gate = read_tensor_from_buf(&data, &mut offset);
        let w_up = read_tensor_from_buf(&data, &mut offset);
        let w_down = read_tensor_from_buf(&data, &mut offset);
        Ok(ExpertWeights { w_gate, w_up, w_down })
    }
}

// Simple binary tensor I/O (not GGUF — this is internal offload format).

fn write_tensor_to_buf(buf: &mut Vec<u8>, t: &Tensor) {
    // ndim (u32) + dims (u64 each) + data (f32 each)
    buf.extend_from_slice(&(t.ndim() as u32).to_le_bytes());
    for &d in &t.shape {
        buf.extend_from_slice(&(d as u64).to_le_bytes());
    }
    for &val in &t.data {
        buf.extend_from_slice(&val.to_le_bytes());
    }
}

fn read_tensor_from_buf(data: &[u8], offset: &mut usize) -> Tensor {
    let ndim = u32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap()) as usize;
    *offset += 4;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(u64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap()) as usize);
        *offset += 8;
    }
    let numel: usize = shape.iter().product();
    let mut vals = Vec::with_capacity(numel);
    for _ in 0..numel {
        vals.push(f32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap()));
        *offset += 4;
    }
    Tensor::from_slice(&vals, &shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_expert(d: usize, ffn: usize) -> ExpertWeights {
        ExpertWeights {
            w_gate: Tensor::randn(&[d, ffn], 0.1),
            w_up: Tensor::randn(&[d, ffn], 0.1),
            w_down: Tensor::randn(&[ffn, d], 0.1),
        }
    }

    #[test]
    fn test_store_and_retrieve() {
        let dir = std::env::temp_dir().join("noor_offload_test1");
        let mut offloader = ExpertOffloader::new(10, &dir).unwrap();

        let expert = make_expert(32, 16);
        offloader.store(0, 0, &expert).unwrap();

        let loaded = offloader.get(0, 0).unwrap();
        assert_eq!(loaded.w_gate.shape, expert.w_gate.shape);
        assert_eq!(loaded.w_gate.data, expert.w_gate.data);

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_lru_eviction() {
        let dir = std::env::temp_dir().join("noor_offload_test2");
        let mut offloader = ExpertOffloader::new(3, &dir).unwrap(); // only 3 in memory

        // Store 5 experts
        for i in 0..5 {
            offloader.store(0, i, &make_expert(16, 8)).unwrap();
        }

        // Access experts 0, 1, 2 (fills cache)
        offloader.get(0, 0).unwrap();
        offloader.get(0, 1).unwrap();
        offloader.get(0, 2).unwrap();
        assert_eq!(offloader.cached_count(), 3);

        // Access expert 3 → should evict expert 0 (LRU)
        offloader.get(0, 3).unwrap();
        assert_eq!(offloader.cached_count(), 3);

        // Expert 0 should now be a cache miss (reloaded from disk)
        let prev_misses = offloader.misses;
        offloader.get(0, 0).unwrap();
        assert_eq!(offloader.misses, prev_misses + 1, "Expert 0 should be a miss after eviction");

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_prefetch() {
        let dir = std::env::temp_dir().join("noor_offload_test3");
        let mut offloader = ExpertOffloader::new(5, &dir).unwrap();

        for i in 0..8 {
            offloader.store(0, i, &make_expert(16, 8)).unwrap();
        }

        // Prefetch experts 4, 5, 6
        offloader.prefetch(&[(0, 4), (0, 5), (0, 6)]).unwrap();

        // These should now be cache hits
        let prev_hits = offloader.hits;
        offloader.get(0, 4).unwrap();
        offloader.get(0, 5).unwrap();
        assert_eq!(offloader.hits, prev_hits + 2, "Prefetched experts should be hits");

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_hit_rate() {
        let dir = std::env::temp_dir().join("noor_offload_test4");
        let mut offloader = ExpertOffloader::new(2, &dir).unwrap();

        offloader.store(0, 0, &make_expert(8, 4)).unwrap();
        offloader.store(0, 1, &make_expert(8, 4)).unwrap();

        offloader.get(0, 0).unwrap(); // hit (was cached on store)
        offloader.get(0, 0).unwrap(); // hit
        offloader.get(0, 1).unwrap(); // hit

        let rate = offloader.hit_rate();
        assert!(rate > 0.9, "Hit rate should be high: {rate}");

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_data_integrity() {
        let dir = std::env::temp_dir().join("noor_offload_test5");
        let mut offloader = ExpertOffloader::new(1, &dir).unwrap(); // only 1 cached

        let expert0 = make_expert(32, 16);
        let expert1 = make_expert(32, 16);
        offloader.store(0, 0, &expert0).unwrap();
        offloader.store(0, 1, &expert1).unwrap();

        // Force disk round-trip by accessing both (cache size = 1)
        let _loaded0 = offloader.get(0, 0).unwrap();
        let loaded1 = offloader.get(0, 1).unwrap();
        let reloaded0 = offloader.get(0, 0).unwrap(); // from disk again

        // Verify data matches original
        for i in 0..expert0.w_gate.numel() {
            assert_eq!(reloaded0.w_gate.data[i], expert0.w_gate.data[i],
                "Data corruption after disk roundtrip at index {i}");
        }
        for i in 0..expert1.w_up.numel() {
            assert_eq!(loaded1.w_up.data[i], expert1.w_up.data[i]);
        }

        std::fs::remove_dir_all(dir).ok();
    }
}
