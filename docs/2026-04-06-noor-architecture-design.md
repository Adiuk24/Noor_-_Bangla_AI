# Noor (নূর) — Architecture & Training Specification

**Version:** 1.0
**Date:** 2026-04-06
**Author:** Arif Adito | Adioris Tech Ltd
**Status:** Design Complete — Awaiting Implementation Plan

---

## 1. Philosophy

Nothing traditional. Every component of the standard LLM stack is replaced:

| Traditional | Noor |
|-------------|------|
| Python | Mojo + Rust + Zig + Julia |
| PyTorch | NoorTorch (custom ~11K lines) |
| CUDA | Metal (Apple) + CUDA (Kaggle burst) |
| Adam optimizer | MuonClip + SMEBU |
| Fixed residuals | Block AttnRes (learned) |
| Softmax routing | Sigmoid + SMEBU + per-expert scales |
| Dense FFN | Parallel Dense + MoE |
| Pre-norm only | Sandwich RMSNorm (depth-scaled) |
| Pickle checkpoints | GGUF (checkpoint = deployment) |
| Cloud training | Consumer M4 + free Kaggle T4 |

**Core principle:** What matters is not how many parameters you have, but how efficiently you use them. MIT proved 50% of layers can be removed. K2 proved 1T params with 32B active beats dense models. Noor targets the sweet spot: maximum capability per active parameter on consumer hardware.

---

## 2. Model Family

One architecture, three sizes. Train once at max scale, prune down for smaller variants.

| Spec | Noor-Edge (E1B) | Noor-Pro (E3B) | Noor-Max (E8B) |
|------|-----------------|----------------|----------------|
| **Target device** | Phone, RPi, browser | Laptop (24GB M4) | Workstation (64GB+) |
| **Total params** | 2.8B | 12B | 28B |
| **Active params/token** | ~1B | ~3B | ~4B |
| **d_model** | 1,024 | 2,048 | 2,816 |
| **Layers** | 24 | 32 | 36 |
| **Q heads / KV heads** | 8 / 2 | 16 / 4 | 16 / 8 |
| **Head dim** | 128 | 128 | 176 |
| **FFN type** | PLE (no MoE) | Parallel dense + MoE | Parallel dense + MoE |
| **MoE experts** | N/A | 32 | 64 |
| **Active experts** | N/A | 4 + 1 shared | 4 + 1 shared |
| **Expert FFN dim** | N/A | 512 | 704 |
| **Dense FFN dim** | 2,816 | 1,536 | 2,112 |
| **Attention** | Sliding only (w=512) | 5:1 sliding/global | 5:1 sliding/global |
| **Context length** | 32K | 128K | 256K |
| **Residual type** | Standard | Block AttnRes (8 blocks) | Block AttnRes (8 blocks) |
| **Normalization** | Pre-RMSNorm | Sandwich RMSNorm | Sandwich RMSNorm |
| **Position encoding** | RoPE (standard) | Dual RoPE (std + p-RoPE 25%) | Dual RoPE (std + p-RoPE 25%) |
| **Routing** | N/A | Sigmoid + SMEBU + per-expert scales | Sigmoid + SMEBU + per-expert scales |
| **Vocab size** | 64,000 | 64,000 | 64,000 |
| **BF16 weight size** | ~5.6 GB | ~24 GB | ~56 GB |
| **TQ3 quantized** | ~1.2 GB | ~5 GB | ~12 GB |

Vocabulary: 64,000 tokens covering English + Bangla + code + special tokens. Smaller than Gemma's 262K — reduces embedding memory without sacrificing coverage for our target languages.

---

## 3. Layer Architecture

### 3.1 Noor-Pro / Noor-Max: Full Architecture (per layer)

```
// Sliding attention layer (5 of every 6 layers)
input
  → SandwichRMSNorm(depth_scaled, scale=1/√layer_idx)
  → GQA Attention(Q=16, KV=4, window=1024, RoPE θ=10K)
  → SandwichRMSNorm(depth_scaled)
  → PARALLEL {
      Dense GeGLU FFN(d_model → d_dense → d_model)
      MoE Router(sigmoid + SMEBU) → top-4 of E + 1 shared
        each expert: SwiGLU(d_model → d_expert → d_model)
        × per_expert_scale[i]  (learned scalar)
  } → sum / √2 → PostNorm
  → BlockAttnRes(pseudo_query, sources=[block_outputs])

// Global attention layer (1 of every 6 layers)
Same structure, but attention uses:
  GQA(Q=16, KV=4, FULL context, p-RoPE θ=1M, 25% dims rotated)
```

### 3.2 Noor-Edge: PLE Architecture (per layer)

No MoE. Instead, Per-Layer Embeddings provide per-layer specialization:

```
input
  → Pre-RMSNorm
  → Sliding GQA Attention(Q=8, KV=2, window=512, RoPE θ=10K)
  → Pre-RMSNorm
  → GeGLU FFN(1024 → 2816 → 1024)
  → PLE Gate: sigmoid(W_gate @ h) * (W_up @ ple_vector[layer_idx])
  → Standard residual: y = F(x) + x

PLE params: d_base=1024, d_ple=128
```

PLE gives per-layer specialization without routing overhead. Each layer gets a unique 128-dim learned vector that modulates behavior through a gated bottleneck. Inspired by Gemma 4 E2B.

---

## 4. Architectural Innovations

### 4.1 Sigmoid Routing + SMEBU (from Trinity)

**Problem:** Softmax routing creates competition between experts — selecting one suppresses others. Auxiliary load-balancing losses are fragile.

**Solution:** Sigmoid routing with SMEBU bias balancing.

Each expert gets an independent sigmoid score (not competing via softmax). Top-k selection picks the highest-scoring experts. SMEBU maintains per-expert bias terms with momentum:

```
b_i(t+1) = β · b_i(t) + λ · (f_target − f_i(t))
b_i(t+1) = κ · tanh(b_i(t+1) / κ)    // soft clamp

router_score_i = sigmoid(W_route @ h + b_i)
```

**Hyperparameters:** κ=5 (clamp range), β=0.9 (momentum), λ=0.01 (update rate)

The tanh clamping prevents bias runaway. Biases naturally reflect expert utilization — they ARE the monitoring signal. No separate instrumentation needed.

**Per-expert learned scales** (from Gemma 4): Each expert has a learned scalar multiplier applied to its output. This lets the model learn which experts should contribute more strongly without changing routing decisions.

### 4.2 Parallel Dense + MoE (from Gemma 4)

Every MoE layer runs both a dense FFN and the MoE branch in parallel:

```
ffn_out = (dense_ffn(x) + moe_out(x)) / √2
```

The dense branch ensures every token gets a baseline FFN transformation regardless of routing. The MoE branch adds specialized knowledge. Division by √2 keeps the output variance stable.

This is strictly better than MoE-only: if routing fails or an expert collapses, the dense branch provides a safety net.

### 4.3 Block AttnRes (from Kimi/Moonshot)

**Problem:** Fixed residuals `y = F(x) + x` dilute information across depth. Every layer's output gets the same weight regardless of relevance.

**Solution:** Block AttnRes groups layers into blocks and learns which earlier blocks each later block should attend to.

```
blocks = [layers 0-3, 4-7, 8-11, 12-15, 16-19, 20-23, 24-27, 28-31]  // 8 blocks

For block B_k:
  pseudo_query_k  (learned d_model vector, zero-initialized)
  keys = [output of B_0, B_1, ..., B_{k-1}]
  attn_weights = softmax(pseudo_query_k @ keys^T / √d)
  residual = Σ(attn_weights_i × block_output_i)
```

**Properties:**
- O(B × d) memory where B=8, not O(L × d) where L=32
- Zero-init means it starts as standard residual and learns to deviate
- Published results: +7.5 GPQA-Diamond, 25% compute savings at 48B scale (Kimi)

**Design decision:** AttnRes is optional in Phase 1 validation. Start with standard residuals. Add AttnRes in Phase 2 only if proxy model shows depth limitations.

### 4.4 Sandwich RMSNorm (from Trinity)

Pre-norm AND post-norm around each sublayer, with depth-scaled gain:

```
gain(layer_idx) = 1 / √layer_idx

// Before sublayer
x_normed = RMSNorm(x) * gain(layer_idx)
// Sublayer compute
y = sublayer(x_normed)
// After sublayer
y_normed = RMSNorm(y) * gain(layer_idx)
```

Depth scaling prevents magnitude explosion in deep MoE networks. Without it, gradient norms grow with depth and destabilize training. Trinity validated this at scale.

### 4.5 Dual RoPE with Proportional Extension

**Sliding layers:** Standard RoPE with θ=10K on all head dimensions. Window=1024.

**Global layers (every 6th):** Proportional RoPE (p-RoPE) with θ=1M applied to 25% of head dimensions. Remaining 75% use standard RoPE θ=10K.

p-RoPE enables context extension beyond training length without fine-tuning. Training at 4096 context, extending to 128K+ at inference via RoPE scaling. The 5:1 sliding-to-global ratio (from Gemma 4) keeps compute manageable — most attention is cheap local sliding, with periodic global layers for long-range dependencies.

### 4.6 GQA (Grouped Query Attention)

Pro/Max: 16 query heads, 4 KV heads (4:1 ratio) — saves 75% KV cache vs MHA.
Edge: 8 query heads, 2 KV heads (4:1 ratio).

Standard GQA with no modifications. Proven effective, no reason to change.

---

## 5. Optimizer: MuonClip

### 5.1 Muon Base (from Kimi/Moonshot)

Muon uses Newton-Schulz orthogonalization of momentum matrices. ~2x more token-efficient than AdamW with only 1 momentum state (same memory as SGD).

```
momentum = β · momentum + gradient
orthogonalized = newton_schulz_orthogonalize(momentum)  // 5 iterations
update = lr · orthogonalized
```

**Memory:** 1x model params (vs Adam's 3x). This is what makes training on 24GB feasible.

### 5.2 QK-Clip (from K2)

**Problem:** At scale, Muon causes exploding attention logits (>1000 observed in K2's 9B/53B pilot).

**Solution:** QK-Clip monitors max attention logit S_max per head per batch. When S_max exceeds threshold τ:

```
For each attention head h:
  S_max_h = max(Q_h @ K_h^T)
  if S_max_h > τ:
    scale = τ / S_max_h
    W_Q_h *= √scale
    W_K_h *= √scale
```

**Properties:**
- τ = 100 (K2's validated threshold)
- Post-update weight adjustment — doesn't alter current step's forward/backward
- For MLA: only scale unshared head-specific components, leave shared rotary key untouched
- K2 result: zero loss spikes across 15.5T token training run

### 5.3 Router Optimizer: SMEBU

SMEBU runs independently of Muon. It only updates the per-expert bias terms in the routing layer. See Section 4.1.

---

## 6. Training Stack: NoorTorch

### 6.1 Language Roles

| Layer | Language | What It Does | Key Files |
|-------|----------|-------------|-----------|
| **Core Engine** | Mojo | Tensor lib, all layers, forward+backward, optimizers | `tensor.mojo`, `metal.mojo`, `backward.mojo`, `model.mojo`, `muon.mojo`, `smebu.mojo` |
| **Kernels** | Zig + C | Hot-path compute via FFI. ARM NEON + Apple AMX + Metal GPU. | `matmul_neon.zig`, `matmul_metal.metal`, `attention_metal.metal`, `rmsnorm.zig`, `rope.zig`, `geglu.zig`, `silu.zig`, `topk.zig`, `cross_entropy.zig`, `softmax.zig` |
| **Orchestration** | Rust | Training loop, data pipeline, expert offload, CLI, eval, logging | `loop.rs`, `offload.rs`, `shards.rs`, `tokenizer.rs`, `checkpoint.rs`, `eval.rs`, `logger.rs`, `tui.rs`, `main.rs` |
| **Research** | Julia | Architecture prototyping on 0.5B proxy. Throwaway. | `proxy_model.jl`, `smebu_sweep.jl`, `attnres_test.jl`, `routing_test.jl`, `parallel_moe_test.jl` |

### 6.2 Ten Layer Types (forward + backward)

1. Embedding + RoPE (standard + p-RoPE)
2. GQA Sliding Attention (window=1024)
3. GQA Global Attention (full context, p-RoPE 25%)
4. Sandwich RMSNorm (depth-scaled)
5. Dense GeGLU FFN
6. Sigmoid MoE Router + SMEBU + per-expert scales
7. SwiGLU Expert FFN
8. Parallel Dense+MoE combiner (sum/√2)
9. Block AttnRes (pseudo-query depth attention)
10. Output projection + cross-entropy loss

Each has hand-written forward and backward passes in Mojo. No autograd graph overhead.

### 6.3 Why This Stack

The Noor Way isn't about training speed. 95% of training step time is GPU matmul — framework dispatch overhead is noise. The stack matters for:

1. **Zero-dependency deployment** — single `noor` binary, no Python install
2. **Inference speed** — Mojo/Zig IS faster for serving (startup in ms, not seconds)
3. **Memory at inference** — no Python runtime saves 400MB (20% of a 2GB phone)
4. **Expert offloading** — Rust async I/O for SSD swap enables 12B MoE on 8GB device
5. **Owning the stack** — no dependency on PyTorch/Google/NVIDIA roadmaps
6. **ADE integration** — GGUF checkpoint = deployment model, no conversion step

### 6.4 CLI Interface

```
noor train   --config config/pro.toml --data data/shards/ --resume checkpoint.gguf
noor run     --model noor-pro.gguf --prompt "..."
noor eval    --model noor-pro.gguf --bench hellaswag,winogrande,mmlu
noor bench   --model noor-pro.gguf --device m4
noor convert --input safetensors/ --output noor.gguf
```

---

## 7. Training Configuration

### 7.1 Noor-Pro Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Precision | BF16 (M4) / FP16 (T4) | Apple native BF16. T4 is Turing — no BF16 tensor cores. |
| Optimizer | MuonClip (β=0.9, τ=100) | 1 momentum state. QK-Clip prevents logit explosion. |
| Router optimizer | SMEBU (κ=5, β=0.9, λ=0.01) | Tanh-clamped momentum bias. Trinity-proven. |
| Learning rate | 3e-4 peak | WSD schedule: 2K warmup → constant → cosine decay to 3e-5 |
| Batch size (tokens) | 512K | Micro-batch=2K, gradient accumulation=256 steps |
| Training context | 4,096 | Short context for training. Extend to 128K via RoPE scaling. |
| Gradient clipping | 1.0 (global norm) | Standard. Works. |
| Activation checkpointing | Every 2 layers | Trade 2x compute for 50% activation memory savings |
| Expert offload | SSD/RAM-backed, LRU, 2x prefetch | 4+1 active in memory, rest on SSD. Rust async I/O. |
| Checkpoint format | GGUF | Includes optimizer state, step count, LR position, SMEBU biases, expert utilization, data position. |
| Checkpoint frequency | Every 30 minutes | Survives 12-hour Kaggle session limits. Auto-upload to HuggingFace. |

### 7.2 Memory Budget: M4 24GB

Metal GPU budget: 18GB (75% of 24GB). OS: ~1.5GB. **Available: 16.5GB.**

| Component | Memory | Note |
|-----------|--------|------|
| Dense layers + attention (BF16) | 3.5 GB | Always in RAM |
| Active experts (4+1, BF16) | 1.5 GB | Swapped per batch via Rust async |
| Muon momentum (BF16, active params) | 5.0 GB | Only dense + active expert momentum in RAM |
| Gradients (BF16, active params) | 2.5 GB | Computed, applied, discarded per micro-batch |
| Activations (BF16, checkpointed) | 3.0 GB | Recompute on backward |
| NoorTorch framework | 0.1 GB | Mojo+Rust binary, no interpreter |
| **TOTAL** | **15.6 GB** | **Fits in 16.5GB budget** |

Remaining 27 experts on SSD: ~4.5GB. M4 SSD reads at 7.4 GB/s = swap in <1ms per expert.

### 7.3 Memory Budget: Kaggle 2x T4

| Component | GPU 0 | GPU 1 | CPU RAM |
|-----------|-------|-------|---------|
| Dense layers + attention (FP16) | 3.5 GB | 3.5 GB | — |
| Active experts (4+1, FP16) | 1.5 GB | 1.5 GB | — |
| Inactive 27 experts (FP16) | — | — | 4.5 GB |
| Muon momentum (active) | 5.0 GB | 5.0 GB | — |
| Gradients (active) | 2.5 GB | 2.5 GB | — |
| Activations (checkpointed) | 3.0 GB | 3.0 GB | — |
| **TOTAL per GPU** | **15.5 GB** | **15.5 GB** | **4.5 GB** |

Strategy: layers 1-16 on GPU 0, layers 17-32 on GPU 1 (pipeline parallel). Experts offload to CPU RAM (30GB available).

---

## 8. Training Strategy

### 8.1 Data Strategy: Quality Over Quantity

The Spurious Rewards paper (arXiv:2506.10947) proved that RLVR amplifies pretraining behaviors — it doesn't teach new reasoning. Distillation has the same ceiling. **Pretraining quality is the ceiling for everything that follows.**

Approach: High-quality curated data + synthetic rephrasing (K2 pattern).

### 8.2 Phased Training Plan

| Phase | What | Hardware | Tokens | Duration |
|-------|------|----------|--------|----------|
| **0. Research** | Julia proxy (0.5B). Validate SMEBU + routing + parallel dense+MoE. Find hyperparameters. | M4 24GB | — | 2 weeks |
| **1. Build NoorTorch** | Mojo tensor lib + layers + backward. Zig kernels. Rust orchestration. Test on 0.5B proxy. | M4 24GB | — | 6-8 weeks |
| **2. Train Noor-Edge** | Train 2.8B PLE model. First real model. Generate synthetic data with Gemma E2B on M4. | Kaggle 1x T4 | 2B | 6 weeks |
| **3. Grow → Train Noor-Pro** | Initialize from Edge weights. Add MoE layers (copy dense → shared expert, random init routed). Full architecture. | Kaggle 2x T4 | 3B | 24 weeks |
| **4. Fine-tune** | Tool-calling, ADE integration, Bangla, reasoning (Noor-Thinking). | Kaggle T4 | 500M | 2 weeks |
| **5. Post-train optimize** | LASER SVD denoising + layer pruning + AdiTurbo TQ3 quantization. | M4 24GB | — | 2 weeks |
| **6. ADE wrap + deploy** | P1-P6 integration. Eyla serves Noor. | M4 24GB | — | 2 weeks |

**Total tokens:** ~5.5B. **Total cloud cost:** $0 (Kaggle free tier).

Optional: $200 RunPod burst (1x A100) for Phase 3 cuts 24 weeks → ~4 weeks.

### 8.3 Progressive Growing (Phase 3)

Initialize Noor-Pro from Noor-Edge:
1. Copy Edge's dense FFN weights into Pro's dense FFN branch
2. Copy Edge's dense FFN weights into Pro's shared expert
3. Random-init the 32 routed experts (small init scale)
4. Copy attention weights, expand heads (8→16 Q, 2→4 KV) via duplication + noise
5. Add AttnRes pseudo-query vectors (zero-init)
6. Resume training — experts learn to specialize from warm start

### 8.4 Synthetic Rephrasing Pipeline (K2 Pattern)

Run Gemma E2B on M4 to generate training data:
- **Knowledge domains:** Style/perspective-diverse prompting rewrites facts in different tones, formats, viewpoints
- **Mathematics:** Formal text → step-by-step explanations with cross-language translation
- **Code:** Function documentation → implementation variants, test cases
- **Bangla:** Parallel translations of English content + native Bangla web content

Goal: Say the same thing in many pedagogically useful ways without inducing overfitting.

### 8.5 K2-Inspired Self-Training Loop (Post Phase 4)

Three-stage self-bootstrapping:

**Stage 1 — Rephrase own training data:**
Earlier Noor checkpoint generates rephrasings of its training data. More signal per token.

**Stage 2 — Synthetic tool environments:**
Generate 1,000+ tool specifications for ADE integration. Create multi-turn trajectories. Judge model filters failures. Noor practices being an agent until it becomes one.

**Stage 3 — Actor-critic self-improvement:**
- **Verifiable rewards (RLVR):** Binary right/wrong for math, coding, instruction following
- **Self-critique rubric rewards:** Noor as generator + critic, guided by:
  - Core rubrics: clarity, relevance, helpfulness
  - Prescriptive rubrics: anti-reward-hacking rules
  - ADE rubrics: P5 Competence Boundary compliance, P6 Confidence calibration
- Temperature decay + token budget limits + PTX loss (prevent forgetting)

This maps directly to ADE P4 (Experience Engine) and P5 (Competence Boundary).

---

## 9. Post-Training Optimization

### 9.1 LASER SVD Denoising (MIT, ICLR 2024)

Apply rank reduction to MLP weight matrices in the latter half of the model. No retraining needed.

```
For layers in [L/2 ... L-1]:
  U, S, V = SVD(W_mlp)
  W_mlp_denoised = U[:, :rank] @ diag(S[:rank]) @ V[:rank, :]
```

Higher-order SVD components encode conflicting/noisy responses from training. Removing them makes weakly learned facts more accessible. Published results: +20-30 percentage points on specific benchmarks. Rank can be reduced up to 99% in targeted layers.

Three hyperparameters to sweep: which layer, which matrix type, how much rank.

### 9.2 Layer Pruning for Device Variants (MIT, ICLR 2025)

Train Noor-Pro at full depth (32 layers). Create smaller variants by pruning:

```
1. Compute angular distance between hidden states at each layer
2. Find optimal block of N layers where representations change least
3. Excise the block (or use simple heuristic: delete from back)
4. NEVER prune the final layer (maximal dissimilarity, critical for output)
5. Heal with QLoRA fine-tuning (single GPU, small dataset)
```

**Key insight:** Knowledge tasks are robust to pruning. Math reasoning degrades first. Pruned variants should be evaluated on both.

**SkipGPT extension:** For inference, add learned routers that dynamically skip layers per-token. 40% parameter reduction with full performance recovery. Attention modules are more redundant than MLP modules.

### 9.3 Sparse Pre-training Integration (MIT CSAIL, ICLR 2025)

During training itself (not post-hoc):
- **Start dense** for first 25% of training compute
- **Begin iterative magnitude pruning** at 25%
- **Conclude pruning** at 75%
- **Continue training** the sparse model for remaining 25%

Modified Chinchilla scaling law: sparse and dense follow same curves when measured by average parameter count over training.

### 9.4 AdiTurbo TQ3 Quantization

Apply AdiTurbo's ternary quantization for edge deployment. Existing NEON kernels from AdiTurbo project handle quantize/dequantize. Results in ~4.5x compression (BF16 → TQ3).

---

## 10. Deployment

### 10.1 Device Matrix

| Device | Model | Format | Size | Runtime |
|--------|-------|--------|------|---------|
| Phone (Android/iOS) | Edge TQ3 | GGUF | 1.2 GB | llama.cpp / MediaPipe |
| Raspberry Pi 5 | Edge TQ3 | GGUF | 1.2 GB | llama.cpp (NEON) |
| Browser | Edge Q4 | ONNX/GGUF | 1.5 GB | transformers.js / WebGPU |
| Qualcomm NPU | Edge INT4 | QNN | 0.8 GB | QNN SDK / Snapdragon |
| Laptop (M4/x86) | Pro TQ3 | GGUF | 5 GB | Eyla AIOS + ADE |
| Desktop / Mac Studio | Pro BF16 | GGUF | 24 GB | Eyla AIOS + ADE |
| Workstation (64GB+) | Max TQ3 | GGUF | 12 GB | Eyla AIOS + ADE |
| Cloud / GPU server | Max BF16 | GGUF / safetensors | 56 GB | vLLM / TensorRT |

### 10.2 Ecosystem Integration

```
Noor (this spec)
  ↓ GGUF checkpoint (no conversion needed)
AdiTurbo
  ↓ TQ3/Q4/INT4 quantization
ADE (P1-P6)
  ↓ Verification + confidence wrapping
  ↓ P1 State: system awareness
  ↓ P2 Instruments: tool calling
  ↓ P3 Memory: semantic + episodic + procedural (EverOS-inspired)
  ↓ P4 Experience: learning from interactions (K2 self-critique pattern)
  ↓ P5 Competence Boundary: knows what it can't do
  ↓ P6 Confidence: calibrated uncertainty
Eyla AIOS
  ↓ Serves Noor to users via CLI/TUI
130M+ users via Grameenphone/Robi Axiata
```

---

## 11. World Model Roadmap (Future)

Beyond language modeling, Noor will evolve toward world modeling capabilities. Phased approach, risk-ordered:

| Phase | What | Architecture | Risk |
|-------|------|-------------|------|
| WM-1 | Action-free latent prediction (JEPA-style) | Predict representations, not pixels | Low |
| WM-2 | Action-conditioned dynamics | Add action tokens + reward heads on logged trajectories | Medium |
| WM-3 | Generative simulator | Diffusion or autoregressive long-horizon rollouts | High |
| WM-4 | Planning + policy learning | MPC/MCTS/imagination actor-critic with eval gates | High |

**Key decisions:**
- Bias toward offline world modeling (avoids Python simulator dependency)
- Transformer dynamics (TD-MPC2 style) most compatible with NoorTorch infra
- Predictive coding / Liquid Neural Networks for P1 State Engine (research module)
- Graph-of-Thought for ADE decision gate / tool-use planning

---

## 12. Research Sources

| Innovation | Source | Paper/Reference |
|-----------|--------|----------------|
| SMEBU, sigmoid routing, sandwich norm | Arcee Trinity | 2025 |
| Block AttnRes | Kimi/Moonshot | arXiv:2603.15031 |
| MuonClip, QK-Clip, self-training loop | Kimi K2 | arXiv:2507.20534 |
| Parallel dense+MoE, PLE, tiny experts, 5:1 attention | Google Gemma 4 | 2025 |
| Layer pruning (50% removable) | MIT/Meta FAIR | arXiv:2403.17887 (ICLR 2025) |
| LASER SVD rank reduction | MIT | arXiv:2312.13558 (ICLR 2024) |
| Sparse pre-training schedule | MIT CSAIL | arXiv:2501.12486 (ICLR 2025) |
| Spurious rewards (pretraining = ceiling) | — | arXiv:2506.10947 |
| EverMemOS self-organizing memory | EverMind AI | 2025 |
| Manifold Hyper-Connections | DeepSeek | Jan 2026 |
| MLA (Multi-head Latent Attention) | DeepSeek | V3, 2025 |
| Gated DeltaNet | Qwen3-Next | 2025 |
| Dynamic skip (SkipGPT) | — | June 2025 |
| JEPA, predictive representations | Meta (LeCun) | 2024-2025 |
| Dreamer RSSM, TD-MPC2 | Various | 2023-2025 |
| Diffusion World Models | Various | 2024-2025 |
| Liquid Neural Networks | MIT CSAIL | 2023 |

---

## 13. Exit Criteria

### Phase 0 (Research Proxy)
- [ ] SMEBU biases balanced (no expert collapse)
- [ ] Sandwich norm stable (no gradient explosion)
- [ ] Loss converges on 0.5B proxy
- [ ] Hyperparameter ranges locked

### Phase 2 (Noor-Edge)
- [ ] PPL competitive with Gemma E2B at same param count
- [ ] Coherent text generation in English and Bangla
- [ ] Runs on phone (TQ3, <1.5GB)

### Phase 3 (Noor-Pro)
- [ ] PPL beats Qwen-3B at same active params
- [ ] Tool-calling accuracy >90% on ADE benchmark
- [ ] Bangla generation coherent
- [ ] Expert utilization >80% (no collapse)
- [ ] HellaSwag + WinoGrande competitive

### Phase 5 (Post-Training)
- [ ] LASER improves accuracy on at least 3 benchmarks
- [ ] Pruned variants maintain >90% of full-model performance
- [ ] TQ3 quantization within 5% of BF16 quality

### Phase 6 (Deployment)
- [ ] ADE P5 boundary detection works
- [ ] P6 confidence calibrated (ECE < 0.1)
- [ ] Eyla serves Noor on laptop, phone, RPi
- [ ] Single `noor` binary < 50MB

---

## 14. What's Different From Everyone Else

- **No CUDA** — Metal-native from day 1. No NVIDIA dependency ever.
- **No Adam** — MuonClip + SMEBU. 67% less optimizer memory.
- **No fixed residuals** — Block AttnRes. Selective depth-wise attention.
- **No softmax routing** — Sigmoid + SMEBU. Independent expert scores.
- **No PyTorch** — NoorTorch (Mojo+Zig+Rust). ~11K lines, not 3M.
- **No cloud training** — Consumer M4 + free Kaggle T4. $0 cloud cost.
- **No pickle checkpoints** — GGUF. Checkpoint = deployment.
- **Expert offloading** — SSD-backed MoE. 3x more experts than RAM allows.
- **ADE-wrapped** — P5 Competence Boundary + P6 Confidence. No other model has this.
- **Self-training** — K2 pattern: rephrase → practice → self-critique. Model improves itself.
- **One train, many sizes** — Train once, prune to Edge/Pro/Max. MIT-proven.
- **Bangla-native** — First high-quality bilingual English+Bangla model at this scale.
