//! Model configuration — loads from proxy.toml / edge.toml / pro.toml.
//! Reuses the same TOML format as the legacy noor-core config.

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct NoorConfig {
    pub model: ModelSection,
    pub moe: MoeSection,
    pub attention: AttentionSection,
    pub norm: NormSection,
    pub residual: ResidualSection,
    pub ple: PleSection,
    pub rope: RopeSection,
    pub training: TrainingSection,
    pub optimizer: OptimizerSection,
    pub smebu: SmebuSection,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelSection {
    pub name: String,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub context_length: usize,
    pub precision: String,
    #[serde(default = "default_bottleneck")]
    pub output_proj_bottleneck: usize,
}

fn default_bottleneck() -> usize { 0 } // 0 = no bottleneck

#[derive(Debug, Clone, Deserialize)]
pub struct MoeSection {
    pub enabled: bool,
    pub n_experts: usize,
    pub n_active_experts: usize,
    pub has_shared_expert: bool,
    pub expert_ffn_dim: usize,
    pub dense_ffn_dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AttentionSection {
    pub sliding_window: usize,
    pub global_every_n: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NormSection {
    pub sandwich: bool,
    pub eps: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResidualSection {
    pub use_attnres: bool,
    pub attnres_blocks: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PleSection {
    pub enabled: bool,
    pub ple_dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeSection {
    pub theta: f64,
    pub prope_theta: f64,
    pub prope_fraction: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrainingSection {
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
pub struct OptimizerSection {
    #[serde(rename = "type")]
    pub optim_type: String,
    pub beta: f64,
    pub qk_clip_tau: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SmebuSection {
    pub kappa: f64,
    pub beta: f64,
    pub lambda: f64,
}

impl NoorConfig {
    pub fn from_toml(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: NoorConfig = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Batch size in sequences (context_length tokens each).
    pub fn batch_size(&self) -> usize {
        self.training.batch_size_tokens / self.model.context_length
    }

    /// Output projection bottleneck dim, or d_model if disabled.
    pub fn output_bottleneck(&self) -> usize {
        if self.model.output_proj_bottleneck > 0 {
            self.model.output_proj_bottleneck
        } else {
            self.model.d_model
        }
    }
}
