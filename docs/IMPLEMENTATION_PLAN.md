# Noor Implementation Plan — Zero Drift

Every step has: **input** (what must exist before starting), **action** (exactly what to do), **output** (what must exist when done), **test** (how to verify it works). No ambiguity. No shortcuts.

---

## PHASE 0: Project Foundation & Research Proxy

### Step 0.1 — Project Scaffolding

**Input:** Empty `noor/` directory with docs and git.

**Action:**
1. Create Rust workspace at project root with `Cargo.toml` (workspace members: `crates/noor-core`, `crates/noor-train`, `crates/noor-cli`)
2. Create `crates/noor-core/` — tensor types, GGUF reader/writer, tokenizer bindings
3. Create `crates/noor-train/` — training loop, data pipeline, expert offload, eval harness
4. Create `crates/noor-cli/` — CLI binary (`noor train|run|eval|bench|convert`)
5. Create `kernels/` with placeholder dirs: `kernels/zig/`, `kernels/metal/`, `kernels/c/`
6. Create `julia/` for research proxy scripts
7. Create `config/` with TOML config files: `proxy.toml`, `edge.toml`, `pro.toml`, `max.toml`
8. Add dependencies to Cargo.toml: `memmap2`, `serde`, `toml`, `tokenizers`, `indicatif`, `clap`
9. Verify `cargo build` succeeds with empty crate stubs

**Output:** Working Rust workspace that compiles. Config files for all model variants.

**Test:** `cargo build --workspace` exits 0. `cargo run -p noor-cli -- --help` prints usage.

---

### Step 0.2 — Model Config System

**Input:** Step 0.1 complete.

**Action:**
1. In `noor-core/src/config.rs`, define `ModelConfig` struct:
   ```
   d_model, n_layers, n_heads, n_kv_heads, head_dim,
   n_experts, n_active_experts, has_shared_expert,
   expert_ffn_dim, dense_ffn_dim, vocab_size,
   sliding_window, global_every_n, use_attnres, attnres_blocks,
   use_sandwich_norm, use_ple, ple_dim,
   rope_theta, prope_theta, prope_fraction,
   context_length, precision (bf16|fp16|fp32)
   ```
2. Implement `ModelConfig::from_toml(path)` deserializer
3. Write all 4 config TOML files matching the architecture spec exactly:
   - `proxy.toml`: d=768, 16 layers, 8 experts, 2 active, context=2048, vocab=32000
   - `edge.toml`: d=1024, 24 layers, PLE, no MoE, context=4096, vocab=64000
   - `pro.toml`: d=2048, 32 layers, 32 experts, 4+1 active, context=4096, vocab=64000
   - `max.toml`: d=2816, 36 layers, 64 experts, 4+1 active, context=4096, vocab=64000
4. Add `ModelConfig::param_count()` method that computes total and active params
5. Add `ModelConfig::memory_estimate()` method that returns bytes needed for weights, optimizer, gradients, activations

**Output:** `ModelConfig` that can load any variant from TOML. Memory estimates match architecture spec.

**Test:** Unit test loads each TOML, asserts param counts match spec (proxy ~0.5B, edge ~2.8B, pro ~12B, max ~28B). Memory estimate for pro < 16.5GB.

---

### Step 0.3 — Tensor Type & Basic Ops

**Input:** Step 0.2 complete.

**Action:**
1. In `noor-core/src/tensor.rs`, define `Tensor` struct:
   - Shape (Vec<usize>), data (Vec<f32> for now — BF16 later), strides
   - Contiguous flag, device marker (CPU for now)
2. Implement basic ops (all CPU, f32, no SIMD yet):
   - `zeros(shape)`, `ones(shape)`, `randn(shape, rng)`, `from_slice(data, shape)`
   - `matmul(a, b)` — naive triple loop
   - `add(a, b)`, `mul(a, b)`, `scale(a, scalar)`
   - `transpose(a, dim0, dim1)`
   - `softmax(a, dim)`, `sigmoid(a)`
   - `rms_norm(a, weight, eps)`
   - `reshape(a, new_shape)`, `slice(a, ranges)`
   - `cross_entropy_loss(logits, targets)`
3. Every op returns a new Tensor (no in-place mutation yet)

**Output:** Working tensor library with all basic ops needed for a forward pass.

**Test:** Unit tests for each op. matmul(2x3, 3x4) == expected. softmax sums to 1. rms_norm output has unit RMS. cross_entropy matches hand-computed value.

---

### Step 0.4 — Embedding + RoPE Layer

**Input:** Step 0.3 complete.

**Action:**
1. In `noor-core/src/layers/embedding.rs`:
   - `Embedding { weight: Tensor }` — vocab_size x d_model
   - `forward(token_ids: &[u32]) -> Tensor` — lookup + return (seq_len, d_model)
2. In `noor-core/src/layers/rope.rs`:
   - `RoPE { inv_freq: Tensor, theta: f64 }` — precomputed inverse frequencies
   - `apply(x: &Tensor, seq_offset: usize) -> Tensor` — rotate pairs of dims
   - Support both standard RoPE (theta=10K, all dims) and p-RoPE (theta=1M, fraction of dims)
3. In `noor-core/src/layers/mod.rs`, re-export all layers

**Output:** Embedding lookup and RoPE rotation working.

**Test:** Embedding of token 0 returns first row of weight matrix. RoPE with offset=0 leaves magnitude unchanged. RoPE with different offsets produces different rotations. p-RoPE only rotates specified fraction of dims.

---

### Step 0.5 — GQA Attention Layer

**Input:** Step 0.4 complete.

**Action:**
1. In `noor-core/src/layers/attention.rs`:
   - `GQAAttention { wq, wk, wv, wo: Tensor, n_heads, n_kv_heads, head_dim }`
   - `forward(x, rope, mask, kv_cache) -> (output, updated_cache)`
   - Implement GQA: repeat KV heads to match Q head count
   - Apply RoPE to Q and K after projection
   - Sliding window: mask positions beyond window distance
   - Global variant: no window mask, use p-RoPE
   - Causal mask: prevent attending to future tokens
2. KV cache: `KVCache { k: Tensor, v: Tensor, seq_len: usize }`
   - Append new K,V to cache, return full attention over cached sequence

**Output:** Working GQA attention with sliding window and global variants.

**Test:** Causal mask: output at position i depends only on positions 0..i. Sliding window: position 100 with window=10 only attends to 90-100. GQA with 4Q/2KV: each KV head serves 2 Q heads.

---

### Step 0.6 — FFN Layers (GeGLU + SwiGLU)

**Input:** Step 0.3 complete.

**Action:**
1. In `noor-core/src/layers/ffn.rs`:
   - `GeGLUFFN { w_gate, w_up, w_down: Tensor }`
   - `forward(x) -> Tensor`: `down(gelu(gate(x)) * up(x))`
   - `SwiGLUFFN { w_gate, w_up, w_down: Tensor }`
   - `forward(x) -> Tensor`: `down(silu(gate(x)) * up(x))`
2. Implement `gelu(x)` and `silu(x)` activation functions in `noor-core/src/ops.rs`

**Output:** Both FFN types working.

**Test:** FFN output shape matches input shape (d_model → d_ffn → d_model). GELU(0) ≈ 0. SiLU(0) = 0. SiLU(large) ≈ large.

---

### Step 0.7 — RMSNorm (Standard + Sandwich)

**Input:** Step 0.3 complete.

**Action:**
1. In `noor-core/src/layers/norm.rs`:
   - `RMSNorm { weight: Tensor, eps: f64 }`
   - `forward(x) -> Tensor`: normalize to unit RMS, scale by weight
   - `SandwichNorm { pre_norm: RMSNorm, post_norm: RMSNorm, depth_scale: f32 }`
   - `pre(x) -> Tensor`: `pre_norm(x) * depth_scale`
   - `post(x) -> Tensor`: `post_norm(x) * depth_scale`
   - `depth_scale = 1.0 / sqrt(layer_idx + 1)`

**Output:** Both norm types working with depth scaling.

**Test:** RMSNorm output has unit RMS (within eps). Sandwich depth_scale at layer 0 = 1.0, layer 3 = 0.5, layer 15 ≈ 0.25.

---

### Step 0.8 — MoE Router (Sigmoid + Top-K)

**Input:** Step 0.6, 0.7 complete.

**Action:**
1. In `noor-core/src/layers/moe.rs`:
   - `MoERouter { gate: Tensor, expert_biases: Tensor, expert_scales: Tensor }`
   - `route(x) -> (expert_indices, expert_weights)`:
     ```
     scores = sigmoid(x @ gate.T + expert_biases)
     top_k_indices = topk(scores, k=n_active)
     weights = scores[top_k_indices] * expert_scales[top_k_indices]
     ```
   - `MoELayer { router, experts: Vec<SwiGLUFFN>, shared_expert: SwiGLUFFN }`
   - `forward(x) -> Tensor`:
     ```
     indices, weights = router.route(x)
     expert_out = sum(weights[i] * experts[indices[i]].forward(x) for i in 0..k)
     shared_out = shared_expert.forward(x)
     return expert_out + shared_out
     ```
2. Track expert utilization counts for logging

**Output:** MoE routing and expert dispatch working.

**Test:** Router returns exactly k indices per token. All indices in [0, n_experts). Weights are positive (sigmoid output). With uniform random input, expert utilization should be roughly balanced (no collapse).

---

### Step 0.9 — Parallel Dense+MoE Combiner

**Input:** Step 0.6, 0.8 complete.

**Action:**
1. In `noor-core/src/layers/parallel_ffn.rs`:
   - `ParallelFFN { dense: GeGLUFFN, moe: MoELayer }`
   - `forward(x) -> Tensor`:
     ```
     dense_out = dense.forward(x)
     moe_out = moe.forward(x)
     return (dense_out + moe_out) / sqrt(2.0)
     ```

**Output:** Parallel dense+MoE working.

**Test:** Output shape = input shape. Output variance ≈ input variance (√2 scaling works).

---

### Step 0.10 — PLE Layer (Edge Variant)

**Input:** Step 0.3 complete.

**Action:**
1. In `noor-core/src/layers/ple.rs`:
   - `PLE { embeddings: Tensor, w_gate: Tensor, w_up: Tensor }`
   - `embeddings` shape: (n_layers, ple_dim) — each layer gets unique vector
   - `forward(x, layer_idx) -> Tensor`:
     ```
     ple_vec = embeddings[layer_idx]
     gate = sigmoid(x @ w_gate)
     modulation = gate * (ple_vec @ w_up)
     return x + modulation
     ```

**Output:** PLE gated modulation working.

**Test:** PLE with zero-init gate weights returns x unchanged. Different layer indices produce different modulations.

---

### Step 0.11 — Full Transformer Block

**Input:** Steps 0.4-0.10 complete.

**Action:**
1. In `noor-core/src/layers/block.rs`:
   - `TransformerBlock` combining all components:
     ```
     // For Pro/Max (MoE variant):
     struct MoEBlock {
       sandwich_norm_attn: SandwichNorm,
       attention: GQAAttention,
       sandwich_norm_ffn: SandwichNorm,
       parallel_ffn: ParallelFFN,
       is_global: bool,  // every 6th layer
     }
     
     // For Edge (PLE variant):
     struct PLEBlock {
       norm_attn: RMSNorm,
       attention: GQAAttention,
       norm_ffn: RMSNorm,
       ffn: GeGLUFFN,
       ple: PLE,
     }
     ```
   - `forward(x, layer_idx, rope, kv_cache) -> (output, cache)`

**Output:** Complete transformer block for both MoE and PLE variants.

**Test:** Block output shape = input shape. Residual connection: if all weights are zero, output ≈ input.

---

### Step 0.12 — Full Model Assembly

**Input:** Step 0.11 complete.

**Action:**
1. In `noor-core/src/model.rs`:
   - `NoorModel`:
     ```
     struct NoorModel {
       config: ModelConfig,
       embedding: Embedding,
       blocks: Vec<Block>,  // enum of MoEBlock | PLEBlock
       final_norm: RMSNorm,
       output: Tensor,  // tied or separate from embedding
     }
     ```
   - `NoorModel::from_config(config) -> Self` — random init all weights
   - `forward(token_ids, kv_caches) -> (logits, updated_caches)`
   - `param_count() -> usize` — count all params
2. Weight initialization:
   - Embeddings: N(0, 0.02)
   - Attention/FFN: N(0, 0.02 / sqrt(2 * n_layers))
   - Router gate: N(0, 0.01)
   - Expert biases: zeros
   - Expert scales: ones
   - AttnRes pseudo-queries: zeros (if enabled)

**Output:** Complete model that runs forward pass on any config.

**Test:** Load `proxy.toml`, create model, forward pass with random tokens → logits shape is (seq_len, vocab_size). Param count matches config estimate within 5%.

---

### Step 0.13 — GGUF Checkpoint Save/Load

**Input:** Step 0.12 complete.

**Action:**
1. In `noor-core/src/gguf.rs`:
   - Implement GGUF v3 format writer:
     - Header: magic, version, tensor count, metadata KV count
     - Metadata: model config as KV pairs, tokenizer info, training state
     - Tensor data: name, dims, type (F32/BF16/F16), offset, data
   - `save_gguf(model, path, metadata)` — writes all weights + config
   - `load_gguf(path) -> (ModelConfig, HashMap<String, Tensor>)` — reads back
2. Training state metadata: step count, LR, SMEBU biases, expert utilization, data shard position

**Output:** Round-trip save/load of model weights in GGUF format.

**Test:** Save model → load model → forward pass produces identical logits (bit-exact f32).

---

### Step 0.14 — Tokenizer Integration

**Input:** Step 0.1 complete.

**Action:**
1. In `noor-core/src/tokenizer.rs`:
   - Wrap HuggingFace `tokenizers` Rust crate
   - `NoorTokenizer::from_file(path) -> Self`
   - `encode(text) -> Vec<u32>`
   - `decode(ids) -> String`
   - `vocab_size() -> usize`
2. For proxy phase: use existing Llama/Qwen tokenizer (download from HF)
3. For production: train custom 64K BPE tokenizer on English+Bangla+code (later phase)

**Output:** Working tokenizer that can encode/decode text.

**Test:** encode("hello world") returns token IDs. decode(encode("hello world")) == "hello world". vocab_size matches config.

---

### Step 0.15 — Data Pipeline

**Input:** Step 0.14 complete.

**Action:**
1. In `noor-train/src/data.rs`:
   - `DataShard` — memory-mapped binary file of pre-tokenized sequences
     - Format: `[seq_len: u32][token_ids: u32 * seq_len]` repeated
   - `ShardReader` — reads from multiple shards with shuffling
     - `new(shard_dir, context_length, batch_size)`
     - `next_batch() -> (input_ids, target_ids)` — input is tokens[:-1], target is tokens[1:]
   - `Preprocessor` — takes raw text, tokenizes, writes to shard format
     - `tokenize_and_shard(input_dir, output_dir, tokenizer, shard_size_mb)`
2. Data format: pre-tokenized binary shards on disk, memory-mapped for zero-copy

**Output:** Data pipeline that reads pre-tokenized shards into batches.

**Test:** Write 1000 tokens to shard → read back → tokens match. Batches have correct shape (batch_size, context_length).

---

### Step 0.16 — Backward Pass (Manual Gradients)

**Input:** Step 0.12 complete.

**Action:**
1. In `noor-core/src/backward/` create gradient functions for each layer type:
   - `embedding_backward(grad_output, token_ids) -> grad_weight`
   - `rope_backward(grad_output, ...) -> grad_input` (rotation is its own inverse with negated angle)
   - `attention_backward(grad_output, q, k, v, attn_weights) -> (grad_q, grad_k, grad_v, grad_wq, grad_wk, grad_wv, grad_wo)`
   - `rms_norm_backward(grad_output, x, weight) -> (grad_input, grad_weight)`
   - `geglu_backward(grad_output, gate_pre, up_pre) -> (grad_gate, grad_up, grad_down)`
   - `swiglu_backward(...)` — same pattern
   - `sigmoid_backward(grad_output, output) -> grad_input` — `grad * output * (1 - output)`
   - `cross_entropy_backward(logits, targets) -> grad_logits` — `softmax(logits) - one_hot(targets)`
   - `moe_backward(grad_output, expert_indices, expert_weights, ...) -> grad per active expert`
2. In `noor-core/src/backward/model.rs`:
   - `backward(model, forward_cache, loss) -> Gradients`
   - Full backward pass through all layers in reverse
   - Only compute gradients for active experts (not all 32/64)

**Output:** Complete backward pass that produces gradients for all parameters.

**Test:** Numerical gradient check: perturb each weight by epsilon, compute loss delta, compare to analytical gradient. Must match within 1e-4 relative error for a small (d=64, 2 layer) model.

---

### Step 0.17 — Muon Optimizer

**Input:** Step 0.16 complete.

**Action:**
1. In `noor-train/src/optim/muon.rs`:
   - `Muon { momentum: HashMap<String, Tensor>, beta: f32, lr: f32 }`
   - `step(params, grads)`:
     ```
     for (name, param) in params:
       m = beta * momentum[name] + grads[name]
       momentum[name] = m
       ortho_m = newton_schulz_orthogonalize(m, iterations=5)
       param -= lr * ortho_m
     ```
   - `newton_schulz_orthogonalize(M, iters)`:
     ```
     X = M / ||M||_F
     for _ in 0..iters:
       A = X @ X^T
       X = 1.5 * X - 0.5 * A @ X
     return X
     ```
2. In `noor-train/src/optim/smebu.rs`:
   - `SMEBU { biases: Tensor, momentum: Tensor, kappa: f32, beta: f32, lambda: f32 }`
   - `update(expert_utilization_counts, n_tokens)`:
     ```
     f_target = 1.0 / n_experts
     for i in 0..n_experts:
       f_i = utilization[i] / n_tokens
       momentum[i] = beta * momentum[i] + lambda * (f_target - f_i)
       biases[i] = kappa * tanh(momentum[i] / kappa)
     ```

**Output:** Both optimizers working.

**Test:** Muon: after 100 steps on a simple quadratic loss, loss decreases monotonically. SMEBU: with imbalanced utilization, biases shift to favor underused experts.

---

### Step 0.18 — QK-Clip

**Input:** Step 0.17 complete.

**Action:**
1. In `noor-train/src/optim/qk_clip.rs`:
   - `QKClip { tau: f32 }`  (default tau=100)
   - `clip(model, attention_logit_stats)`:
     ```
     for layer in model.layers:
       for head in 0..n_heads:
         s_max = attention_logit_stats[layer][head].max
         if s_max > tau:
           scale = sqrt(tau / s_max)
           layer.attention.wq[head] *= scale
           layer.attention.wk[head] *= scale
     ```
   - Called AFTER optimizer step, BEFORE next forward pass

**Output:** QK-Clip that prevents attention logit explosion.

**Test:** Artificially set attention weights to produce logits > 200. After QK-Clip, max logit ≤ 100.

---

### Step 0.19 — Training Loop

**Input:** Steps 0.15, 0.16, 0.17, 0.18 complete.

**Action:**
1. In `noor-train/src/loop.rs`:
   ```
   fn train(config, model, data, optimizer, smebu, qk_clip):
     for step in 0..total_steps:
       // Forward
       batch = data.next_batch()
       logits, cache = model.forward(batch.input_ids)
       loss = cross_entropy(logits, batch.target_ids)
       
       // Backward
       grads = backward(model, cache, loss)
       
       // Gradient clipping
       grad_norm = global_norm(grads)
       if grad_norm > 1.0:
         scale_grads(grads, 1.0 / grad_norm)
       
       // Optimizer step
       optimizer.step(model.params, grads)
       smebu.update(cache.expert_utilization)
       qk_clip.clip(model, cache.attention_stats)
       
       // LR schedule (WSD)
       lr = wsd_schedule(step, warmup, total, lr_max, lr_min)
       optimizer.set_lr(lr)
       
       // Logging
       if step % log_every == 0:
         log(step, loss, lr, grad_norm, expert_util, max_attn_logit)
       
       // Checkpoint
       if step % ckpt_every == 0:
         save_gguf(model, optimizer, step, ...)
   ```
2. WSD schedule: warmup for `warmup_steps`, constant at `lr_max`, cosine decay from step `decay_start` to `lr_min`
3. Gradient accumulation: accumulate grads over `accum_steps` micro-batches before optimizer step

**Output:** Complete training loop with logging and checkpointing.

**Test:** Train proxy model (d=128, 4 layers, 4 experts) on 10K random tokens for 100 steps. Loss must decrease. Expert utilization must not collapse to < 2 experts.

---

### Step 0.20 — Evaluation Harness

**Input:** Step 0.12, 0.14 complete.

**Action:**
1. In `noor-train/src/eval.rs`:
   - `eval_perplexity(model, tokenizer, text) -> f64` — exp(avg cross-entropy)
   - `eval_hellaswag(model, tokenizer, data_path) -> f64` — accuracy on HellaSwag
   - `eval_winogrande(model, tokenizer, data_path) -> f64` — accuracy on WinoGrande
   - `eval_mmlu(model, tokenizer, data_path) -> f64` — 5-shot accuracy on MMLU
2. Download eval datasets to `data/eval/`

**Output:** Eval harness that reports PPL + benchmark scores.

**Test:** Run eval on random-init model → PPL ≈ vocab_size (uniform distribution). Accuracy ≈ chance level (25% for 4-choice).

---

### Step 0.21 — CLI Binary

**Input:** Steps 0.19, 0.20 complete.

**Action:**
1. In `noor-cli/src/main.rs`:
   ```
   noor train --config <path> --data <dir> [--resume <gguf>]
   noor run   --model <gguf> --prompt <text> [--max-tokens N]
   noor eval  --model <gguf> --bench hellaswag,winogrande,mmlu
   noor bench --model <gguf> --device <cpu|metal>
   noor convert --input <dir> --output <gguf>
   ```
2. Use `clap` for argument parsing
3. `noor run` implements simple greedy/temperature sampling

**Output:** Single binary that does everything.

**Test:** `noor train --config config/proxy.toml --data data/test/ --steps 10` runs without error. `noor run --model checkpoint.gguf --prompt "Hello"` generates tokens.

---

### Step 0.22 — Validate Proxy Model

**Input:** All Phase 0 steps complete.

**Action:**
1. Download small dataset (~100MB text): WikiText-103 or similar
2. Tokenize into shards: `noor convert` or preprocessing script
3. Train proxy model (0.5B, config/proxy.toml) for 5,000 steps
4. Evaluate: check loss curve, expert utilization, attention logits
5. Run eval harness: PPL on held-out set

**Exit Criteria:**
- [ ] Loss converges (decreasing trend, no divergence)
- [ ] No expert collapse (all experts get > 5% utilization)
- [ ] No attention logit explosion (max < 100 after QK-Clip)
- [ ] Sandwich norm stable (no NaN/Inf in any layer)
- [ ] PPL improving over training
- [ ] Checkpoint save/load round-trips correctly (resume produces same loss)

**Test:** All exit criteria pass. If any fail, debug and fix before proceeding.

---

## PHASE 1: Zig Kernel Acceleration

### Step 1.1 — BF16 Tensor Type

**Input:** Phase 0 complete.

**Action:**
1. Add BF16 support to Tensor: `enum DType { F32, BF16, F16 }`
2. BF16 storage: pack as u16, convert to f32 for compute
3. In `kernels/zig/bf16.zig`: conversion functions `f32_to_bf16`, `bf16_to_f32`
4. FFI bridge: Rust calls Zig functions via `extern "C"`
5. Update all ops to dispatch based on dtype

**Output:** Tensors can store BF16, compute in F32.

**Test:** f32 → bf16 → f32 round-trip loses < 1% relative error. Training with BF16 storage produces similar loss curve to F32.

---

### Step 1.2 — NEON MatMul Kernel

**Input:** Step 1.1 complete.

**Action:**
1. In `kernels/zig/matmul_neon.zig`:
   - Tiled matmul for BF16 inputs, F32 accumulation
   - Tile sizes tuned for M4 cache: 64x64 or 128x128
   - Use ARM NEON `vdot` intrinsics
   - Multi-threaded: split M dimension across threads
2. Benchmark against naive Rust matmul
3. FFI: `extern fn noor_matmul(a_ptr, b_ptr, c_ptr, M, N, K, ...)`

**Output:** Fast NEON matmul that beats naive by > 10x.

**Test:** Results match naive matmul within BF16 precision. Benchmark: 2048x2048 matmul < 50ms on M4.

---

### Step 1.3 — Metal GPU Kernels

**Input:** Step 1.1 complete.

**Action:**
1. In `kernels/metal/matmul.metal`: GPU matmul shader
2. In `kernels/metal/attention.metal`: Fused attention (flash-attention style)
3. Metal dispatch from Rust via `metal-rs` crate or raw Objective-C FFI
4. Device abstraction: `enum Device { CPU, Metal }` with dispatch

**Output:** GPU-accelerated matmul and attention on Apple Metal.

**Test:** Metal matmul matches CPU matmul results. GPU matmul > 5x faster than CPU for large matrices.

---

### Step 1.4 — Remaining Zig Kernels

**Input:** Step 1.2 complete.

**Action:**
1. `kernels/zig/rmsnorm.zig` — fused RMSNorm forward (+ backward when needed)
2. `kernels/zig/rope.zig` — vectorized RoPE rotation
3. `kernels/zig/geglu.zig` — fused gate * GELU
4. `kernels/zig/silu.zig` — SiLU for SwiGLU experts
5. `kernels/zig/topk.zig` — sigmoid scores + top-k selection
6. `kernels/zig/softmax.zig` — fused softmax + scale
7. `kernels/zig/cross_entropy.zig` — fused softmax + cross-entropy loss

**Output:** All hot-path ops have Zig NEON implementations.

**Test:** Each kernel matches Rust reference implementation within BF16 precision.

---

### Step 1.5 — Expert SSD Offload (Rust)

**Input:** Phase 0 complete.

**Action:**
1. In `noor-train/src/offload.rs`:
   - `ExpertOffloader`:
     - `active_experts: HashMap<(layer, expert_id), Tensor>` — in GPU/CPU memory
     - `disk_path: PathBuf` — directory with expert weight files
     - `lru_cache: LruCache<(layer, expert_id)>` — tracks recently used
   - `prefetch(layer, expert_ids)` — async read from SSD into memory
   - `evict()` — write least-recently-used expert back to SSD
   - `get(layer, expert_id) -> &Tensor` — returns from cache, blocking if prefetch pending
2. Prefetch strategy: when routing decides experts for batch, immediately prefetch next batch's predicted experts (2x prefetch)
3. Use `tokio` or `async-std` for async file I/O

**Output:** Expert weights transparently paged between SSD and RAM.

**Test:** Create 32 experts, allow only 5 in memory. Access all 32 in sequence. Verify correct weights returned. Measure: M4 SSD swap < 2ms per expert.

---

## PHASE 2: Train Noor-Edge (2.8B)

### Step 2.1 — Prepare Training Data

**Input:** Phase 1 complete.

**Action:**
1. Download/generate training data:
   - English: FineWeb-Edu subset (high quality, ~50GB)
   - Bangla: CC-100 Bangla + Oscar Bangla (~10GB)
   - Code: The Stack v2 subset (Python, JS, Rust, ~10GB)
2. Run synthetic rephrasing with Gemma E2B on M4:
   - Rephrase knowledge documents in varied styles
   - Generate step-by-step math explanations
   - Cross-language Bangla↔English pairs
3. Tokenize all data into binary shards
4. Target: ~2B tokens total

**Output:** Pre-tokenized binary shards ready for training.

**Test:** Shard files readable. Token distribution looks reasonable (no degenerate sequences). Total token count ≈ 2B.

---

### Step 2.2 — Train Noor-Edge on Kaggle T4

**Input:** Step 2.1 complete. NoorTorch compiles on Linux/CUDA (may need CUDA kernel variants).

**Action:**
1. Upload NoorTorch binary + data shards to Kaggle
2. Train with `config/edge.toml` on single T4 (FP16)
3. Checkpoint every 30 min → auto-upload to HuggingFace
4. Resume across 12-hour Kaggle sessions
5. Monitor: loss curve, PPL, generation samples

**Exit Criteria:**
- [ ] PPL competitive with Gemma E2B at same param count
- [ ] Coherent English text generation
- [ ] Coherent Bangla text generation
- [ ] PLE modulation active (not dead gates)
- [ ] Loss converged (no improvement for 1000+ steps)

---

### Step 2.3 — Evaluate + LASER Denoise Noor-Edge

**Input:** Step 2.2 complete.

**Action:**
1. Run full eval harness: PPL, HellaSwag, WinoGrande, MMLU
2. Apply LASER SVD to MLP layers in latter half
3. Sweep: which layers, which matrices, how much rank reduction
4. Re-evaluate after LASER — expect +5-20 points on some benchmarks
5. AdiTurbo TQ3 quantize → test on phone

**Output:** Optimized Noor-Edge model, quantized for deployment.

---

## PHASE 3: Grow → Train Noor-Pro (12B MoE)

### Step 3.1 — Progressive Growing from Edge to Pro

**Input:** Phase 2 complete.

**Action:**
1. Load Noor-Edge checkpoint
2. Expand d_model: 1024 → 2048 (duplicate + add noise)
3. Expand attention: 8Q/2KV → 16Q/4KV (duplicate heads + noise)
4. Copy Edge FFN → Pro dense branch
5. Copy Edge FFN → Pro shared expert
6. Random-init 32 routed experts (scale=0.01)
7. Add AttnRes pseudo-queries (zero-init)
8. Add sandwich norm (init from Edge pre-norm weights)
9. Expand layers: 24 → 32 (copy last 8 layers from Edge, add noise)
10. Save as GGUF checkpoint

**Output:** Warm-started Noor-Pro checkpoint.

**Test:** Forward pass produces valid logits. Loss is higher than Edge's final loss but not random (should be < 2x Edge's loss).

---

### Step 3.2 — Train Noor-Pro on Kaggle 2x T4

**Input:** Step 3.1 complete.

**Action:**
1. Pipeline parallel: layers 1-16 on GPU 0, layers 17-32 on GPU 1
2. Expert offload: inactive experts in CPU RAM (30GB available)
3. Train with `config/pro.toml` on Kaggle 2x T4
4. Target: 3B tokens, ~24 weeks at 30 hrs/week (or burst on RunPod)
5. Checkpoint + upload every 30 min

**Exit Criteria:**
- [ ] PPL beats Qwen-3B at same active params
- [ ] Tool-calling accuracy > 90%
- [ ] Bangla generation coherent
- [ ] Expert utilization > 80%
- [ ] HellaSwag + WinoGrande competitive
- [ ] AttnRes improving over standard residual (compare with ablation)

---

### Step 3.3 — Post-Training Pipeline for Pro

**Input:** Step 3.2 complete.

**Action:**
1. LASER SVD denoising (sweep layers + rank)
2. Layer pruning experiments (how many layers can be removed?)
3. Self-training loop:
   a. Rephrase own training data
   b. Generate 1000+ tool specs for ADE
   c. Actor-critic self-improvement (RLVR + self-critique rubrics)
4. AdiTurbo TQ3 quantize
5. Full eval harness

**Output:** Production-ready Noor-Pro model.

---

## PHASE 4: ADE Integration & Deployment

### Step 4.1 — ADE Wrapping

**Input:** Phase 3 complete.

**Action:**
1. Connect Noor-Pro to ADE P1-P6 pipeline in Eyla
2. P2 Instruments: register Noor's tool-calling format
3. P5 Competence Boundary: calibrate from self-training data
4. P6 Confidence: implement calibrated uncertainty (temperature scaling)
5. Test end-to-end: user query → ADE → Noor → ADE verification → response

**Output:** Noor serving through Eyla AIOS with full ADE verification.

---

### Step 4.2 — Multi-Device Deployment

**Input:** Step 4.1 complete.

**Action:**
1. Noor-Edge TQ3 → phone (llama.cpp), RPi (NEON), browser (WebGPU)
2. Noor-Pro TQ3 → laptop (Eyla AIOS)
3. Noor-Pro BF16 → desktop (Eyla AIOS)
4. Test on each target device
5. Measure: latency, memory, quality

**Output:** Noor running on all target devices.

**Final Exit Criteria:**
- [ ] Single `noor` binary < 50MB
- [ ] Phone inference: > 5 tok/s
- [ ] Laptop inference: > 20 tok/s
- [ ] All ADE P1-P6 programs functional
- [ ] Eyla serves Noor to users

---

## Rules for AI Agents

1. **Never skip a step.** Every step has a test. Pass the test before moving on.
2. **Never change the architecture.** The spec in `docs/2026-04-06-noor-architecture-design.md` is authoritative. If something seems wrong, flag it — don't silently change it.
3. **Never add Python.** Not even "temporarily." Not even for "just this one script."
4. **Never use Adam/AdamW.** Muon + SMEBU only. No exceptions.
5. **Never use PyTorch.** NoorTorch is the framework. Build what's missing, don't import what exists.
6. **Test before proceeding.** Every step has explicit test criteria. If tests fail, fix the issue in the current step. Do not proceed to the next step with failing tests.
7. **Commit after every step.** Each step gets its own git commit with a descriptive message.
8. **Log everything.** Loss, LR, grad norm, expert utilization, max attention logit, memory usage — every training step.
9. **Checkpoint frequently.** Every 30 minutes during training. GGUF format. Include full training state.
10. **Match the spec exactly.** Config values, layer types, hyperparameters — all must match the architecture document. The numbers are not suggestions.
