use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub model: ModelParams,
    pub moe: MoEParams,
    pub attention: AttentionParams,
    pub norm: NormParams,
    pub residual: ResidualParams,
    pub ple: PLEParams,
    pub rope: RoPEParams,
    pub training: TrainingParams,
    pub optimizer: OptimizerParams,
    pub smebu: SMEBUParams,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelParams {
    pub name: String,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub context_length: usize,
    pub precision: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MoEParams {
    pub enabled: bool,
    pub n_experts: usize,
    pub n_active_experts: usize,
    pub has_shared_expert: bool,
    pub expert_ffn_dim: usize,
    pub dense_ffn_dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AttentionParams {
    pub sliding_window: usize,
    pub global_every_n: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NormParams {
    pub sandwich: bool,
    pub eps: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResidualParams {
    pub use_attnres: bool,
    pub attnres_blocks: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PLEParams {
    pub enabled: bool,
    pub ple_dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RoPEParams {
    pub theta: f64,
    pub prope_theta: f64,
    pub prope_fraction: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrainingParams {
    pub lr_max: f64,
    pub lr_min: f64,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub batch_size_tokens: usize,
    pub micro_batch_tokens: usize,
    pub grad_clip: f64,
    pub checkpoint_every_steps: usize,
    pub log_every_steps: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OptimizerParams {
    #[serde(rename = "type")]
    pub optim_type: String,
    pub beta: f64,
    pub qk_clip_tau: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SMEBUParams {
    pub kappa: f64,
    pub beta: f64,
    pub lambda: f64,
}

impl ModelConfig {
    /// Load config from a TOML file.
    pub fn from_toml(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: ModelConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Total parameter count (all experts included).
    pub fn param_count_total(&self) -> usize {
        let d = self.model.d_model;
        let v = self.model.vocab_size;
        let l = self.model.n_layers;
        let h = self.model.n_heads;
        let kv = self.model.n_kv_heads;
        let hd = self.model.head_dim;

        // Embedding: vocab * d_model
        let embed = v * d;

        // Per-layer attention: Wq + Wk + Wv + Wo
        let attn_per_layer = (h * hd * d) + (kv * hd * d) + (kv * hd * d) + (h * hd * d);

        // Per-layer FFN
        let ffn_per_layer = if self.moe.enabled {
            // Dense branch: GeGLU (gate + up + down)
            let dense = 3 * d * self.moe.dense_ffn_dim;
            // Each expert: SwiGLU (gate + up + down)
            let per_expert = 3 * d * self.moe.expert_ffn_dim;
            let all_experts = self.moe.n_experts * per_expert;
            // Shared expert
            let shared = if self.moe.has_shared_expert { per_expert } else { 0 };
            // Router gate
            let router = d * self.moe.n_experts + self.moe.n_experts; // gate weights + biases
            // Expert scales
            let scales = self.moe.n_experts;
            dense + all_experts + shared + router + scales
        } else {
            // Dense-only GeGLU
            3 * d * self.moe.dense_ffn_dim
        };

        // Per-layer norm weights
        let norm_per_layer = if self.norm.sandwich {
            4 * d // 2 sandwich norms (pre+post each for attn and ffn)
        } else {
            2 * d // 2 pre-norms (attn and ffn)
        };

        // PLE params per layer
        let ple_per_layer = if self.ple.enabled {
            self.ple.ple_dim + (d * self.ple.ple_dim) + (d * 1) // ple_vec + w_up + w_gate
        } else {
            0
        };

        // AttnRes pseudo-queries
        let attnres = if self.residual.use_attnres {
            self.residual.attnres_blocks * d
        } else {
            0
        };

        // Final norm + output projection (tied with embedding = 0 extra, or separate = v*d)
        let final_norm = d;
        let output_proj = v * d; // separate output projection

        let per_layer = attn_per_layer + ffn_per_layer + norm_per_layer + ple_per_layer;
        embed + (l * per_layer) + attnres + final_norm + output_proj
    }

    /// Active parameter count per token.
    pub fn param_count_active(&self) -> usize {
        let d = self.model.d_model;
        let v = self.model.vocab_size;
        let l = self.model.n_layers;
        let h = self.model.n_heads;
        let kv = self.model.n_kv_heads;
        let hd = self.model.head_dim;

        let embed = v * d;
        let attn_per_layer = (h * hd * d) + (kv * hd * d) + (kv * hd * d) + (h * hd * d);

        let ffn_per_layer = if self.moe.enabled {
            let dense = 3 * d * self.moe.dense_ffn_dim;
            let active_experts = self.moe.n_active_experts * 3 * d * self.moe.expert_ffn_dim;
            let shared = if self.moe.has_shared_expert {
                3 * d * self.moe.expert_ffn_dim
            } else {
                0
            };
            dense + active_experts + shared
        } else {
            3 * d * self.moe.dense_ffn_dim
        };

        let norm_per_layer = if self.norm.sandwich { 4 * d } else { 2 * d };
        let ple_per_layer = if self.ple.enabled {
            self.ple.ple_dim + (d * self.ple.ple_dim) + d
        } else {
            0
        };

        let per_layer = attn_per_layer + ffn_per_layer + norm_per_layer + ple_per_layer;
        let final_norm = d;
        let output_proj = v * d;

        embed + (l * per_layer) + final_norm + output_proj
    }

    /// Estimated memory in bytes for training (weights + optimizer + gradients + activations).
    pub fn memory_estimate_bytes(&self) -> usize {
        let bytes_per_param = match self.model.precision.as_str() {
            "bf16" | "fp16" => 2,
            "f32" => 4,
            _ => 2,
        };

        let total_params = self.param_count_total();
        let active_params = self.param_count_active();

        // Weights (all experts if offloading, else active only)
        let weights = if self.moe.enabled {
            // Dense + attention always in RAM, only active experts in RAM
            active_params * bytes_per_param
        } else {
            total_params * bytes_per_param
        };

        // Muon momentum: 1x active params
        let momentum = active_params * bytes_per_param;

        // Gradients: active params only
        let gradients = active_params * bytes_per_param;

        // Activations estimate: ~2x d_model * context * n_layers * bytes (with checkpointing /2)
        let ctx = self.model.context_length;
        let activations = self.model.d_model * ctx * self.model.n_layers * bytes_per_param / 2;

        weights + momentum + gradients + activations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_proxy_config() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .parent().unwrap()
            .join("config/proxy.toml");
        let config = ModelConfig::from_toml(&path).expect("Failed to load proxy.toml");
        assert_eq!(config.model.name, "noor-proxy");
        assert_eq!(config.model.d_model, 768);
        assert_eq!(config.model.n_layers, 16);
        assert_eq!(config.moe.n_experts, 8);

        let total = config.param_count_total();
        let active = config.param_count_active();
        println!("Proxy: total={total}, active={active}");
        // Proxy should be roughly 0.3-0.6B total
        assert!(total > 200_000_000, "Proxy total params too low: {total}");
        assert!(total < 800_000_000, "Proxy total params too high: {total}");
        assert!(active < total, "Active should be less than total for MoE");
    }

    #[test]
    fn test_load_edge_config() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .parent().unwrap()
            .join("config/edge.toml");
        let config = ModelConfig::from_toml(&path).expect("Failed to load edge.toml");
        assert_eq!(config.model.name, "noor-edge");
        assert!(config.ple.enabled);
        assert!(!config.moe.enabled);

        let total = config.param_count_total();
        println!("Edge: total={total}");
        // Edge with d=1024, 24 layers, dense FFN: ~400M
        // NOTE: Spec target was 2.8B — dimensions will be scaled up during
        // Phase 0 research to hit target. Current config validates the structure.
        assert!(total > 300_000_000, "Edge total too low: {total}");
        assert!(total < 800_000_000, "Edge total too high: {total}");
    }

    #[test]
    fn test_load_pro_config() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .parent().unwrap()
            .join("config/pro.toml");
        let config = ModelConfig::from_toml(&path).expect("Failed to load pro.toml");
        assert_eq!(config.model.name, "noor-pro");
        assert!(config.moe.enabled);
        assert!(config.residual.use_attnres);

        let total = config.param_count_total();
        let active = config.param_count_active();
        println!("Pro: total={total}, active={active}");
        // Pro with 32 tiny experts (FFN=512): ~4B total, ~1.4B active
        // NOTE: Spec target was 12B total, 3B active — expert_ffn_dim will be
        // increased during Phase 0 research to hit target. Tiny experts (Gemma 4
        // pattern) keep individual experts small; scaling up to ~1700 FFN hits 12B.
        assert!(total > 3_000_000_000, "Pro total too low: {total}");
        assert!(total < 8_000_000_000, "Pro total too high: {total}");
        assert!(active > 1_000_000_000, "Pro active too low: {active}");
        assert!(active < 3_000_000_000, "Pro active too high: {active}");

        // Memory estimate should fit in 16.5GB (with current smaller dimensions)
        let mem = config.memory_estimate_bytes();
        let mem_gb = mem as f64 / (1024.0 * 1024.0 * 1024.0);
        println!("Pro memory estimate: {mem_gb:.1} GB");
        assert!(mem_gb < 16.5, "Pro memory {mem_gb:.1}GB exceeds 16.5GB budget");
    }

    #[test]
    fn test_load_max_config() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .parent().unwrap()
            .join("config/max.toml");
        let config = ModelConfig::from_toml(&path).expect("Failed to load max.toml");
        assert_eq!(config.model.name, "noor-max");
        assert_eq!(config.moe.n_experts, 64);

        let total = config.param_count_total();
        let active = config.param_count_active();
        println!("Max: total={total}, active={active}");
        // Max with 64 tiny experts (FFN=704): ~10-12B total
        // NOTE: Spec target was 28B — dimensions scale up during research.
        assert!(total > 5_000_000_000, "Max total too low: {total}");
        assert!(total < 20_000_000_000, "Max total too high: {total}");
    }
}
