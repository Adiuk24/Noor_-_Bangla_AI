//! Evaluation harness: perplexity and generation quality.

use noor_core::model::NoorModel;

/// Compute perplexity on a sequence of tokens.
/// PPL = exp(avg_cross_entropy)
pub fn eval_perplexity(model: &mut NoorModel, tokens: &[u32]) -> f32 {
    let context_len = model.config.model.context_length;
    let vocab = model.config.model.vocab_size;

    if tokens.len() < 2 {
        return f32::INFINITY;
    }

    let mut total_loss = 0.0f64;
    let mut total_tokens = 0usize;

    // Slide window through the data
    let stride = context_len.min(tokens.len() - 1);
    let mut pos = 0;

    while pos + 1 < tokens.len() {
        let end = (pos + context_len).min(tokens.len() - 1);
        let input = &tokens[pos..end];
        let targets = &tokens[pos + 1..end + 1];

        let output = model.forward(input, None);

        // Compute per-token cross-entropy
        for (t, &target) in targets.iter().enumerate() {
            let logit_offset = t * vocab;
            let logit_slice = &output.logits.data[logit_offset..logit_offset + vocab];

            // Log-softmax for this token
            let max_val = logit_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp: f64 = logit_slice.iter()
                .map(|&x| ((x - max_val) as f64).exp())
                .sum::<f64>()
                .ln() + max_val as f64;

            let log_prob = logit_slice[target as usize] as f64 - log_sum_exp;
            total_loss -= log_prob;
            total_tokens += 1;
        }

        pos += stride;
        if pos + 1 >= tokens.len() {
            break;
        }
    }

    if total_tokens == 0 {
        return f32::INFINITY;
    }

    let avg_loss = total_loss / total_tokens as f64;
    avg_loss.exp() as f32
}

/// Generate text and return token IDs.
pub fn generate(
    model: &mut NoorModel,
    prompt: &[u32],
    max_tokens: usize,
    temperature: f32,
) -> Vec<u32> {
    if temperature <= 0.0 {
        return model.generate_greedy(prompt, max_tokens);
    }

    let mut all_ids = prompt.to_vec();
    let vocab = model.config.model.vocab_size;

    // Prefill
    let out = model.forward(prompt, None);
    let mut caches = out.kv_caches;

    // Sample from last position
    let last_start = (prompt.len() - 1) * vocab;
    let logits = &out.logits.data[last_start..last_start + vocab];
    let next_id = sample_with_temperature(logits, temperature);
    all_ids.push(next_id);

    // Decode
    for _ in 1..max_tokens {
        let token = [*all_ids.last().unwrap()];
        let out = model.forward(&token, Some(caches));
        caches = out.kv_caches;
        let logits = &out.logits.data[..vocab];
        let next_id = sample_with_temperature(logits, temperature);
        all_ids.push(next_id);
    }

    all_ids
}

fn sample_with_temperature(logits: &[f32], temperature: f32) -> u32 {
    // Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

    // Softmax
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&x| x / sum).collect();

    // Sample (simple — use rand for production)
    let r: f32 = rand_simple();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Simple pseudo-random for sampling (no external dependency in this module).
fn rand_simple() -> f32 {
    use std::time::SystemTime;
    let t = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    // Simple hash
    let x = t.wrapping_mul(2654435761);
    (x as f32) / (u32::MAX as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use noor_core::config::ModelConfig;
    use std::path::Path;

    fn tiny_model() -> (ModelConfig, NoorModel) {
        let config_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .parent().unwrap()
            .join("config/proxy.toml");
        let mut config = ModelConfig::from_toml(&config_path).unwrap();
        config.model.d_model = 32;
        config.model.n_layers = 2;
        config.model.n_heads = 2;
        config.model.n_kv_heads = 1;
        config.model.head_dim = 16;
        config.model.vocab_size = 100;
        config.model.context_length = 16;
        config.moe.n_experts = 4;
        config.moe.n_active_experts = 2;
        config.moe.expert_ffn_dim = 16;
        config.moe.dense_ffn_dim = 32;
        config.attention.sliding_window = 8;
        let model = NoorModel::from_config(&config);
        (config, model)
    }

    #[test]
    fn test_perplexity_random_model() {
        let (config, mut model) = tiny_model();
        let tokens: Vec<u32> = (0..50).map(|i| (i % 100) as u32).collect();
        let ppl = eval_perplexity(&mut model, &tokens);
        // Random model should have PPL ≈ vocab_size
        assert!(ppl.is_finite(), "PPL should be finite");
        assert!(ppl > 10.0, "Random model PPL should be high, got {ppl}");
        eprintln!("Random model PPL: {ppl} (vocab={}, expected ~{})", config.model.vocab_size, config.model.vocab_size);
    }

    #[test]
    fn test_generate_returns_valid_tokens() {
        let (config, mut model) = tiny_model();
        let prompt = vec![1u32, 5, 10];
        let generated = generate(&mut model, &prompt, 5, 0.0);
        assert_eq!(generated.len(), 8); // 3 prompt + 5 generated
        for &t in &generated[3..] {
            assert!((t as usize) < config.model.vocab_size);
        }
    }
}
