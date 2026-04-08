# Noor Project — Knowledge Graph

> **Purpose:** Single source of truth for any AI agent entering this project. Read this FIRST before any action. Prevents context drift and wrong suggestions.
>
> **Last updated:** 2026-04-08

---

## 1. WHO

**Arif Adito (Adi)** — Solo founder, Adioris Tech Ltd, Bangladesh.
- Builds everything himself. No team.
- Hardware: Mac M4 24GB, Desktop i7-14700K + RTX 3060 12GB, RunPod A100 (pay-per-hour), Kaggle free tier.
- HuggingFace: `Adiuk`
- GitHub: `Adiuk24/Noor_-_Bangla_AI`
- Languages: Rust, C, Zig, Mojo. Some Julia. TypeScript for Eyla frontend.
- **Communication style:** Direct, impatient. Wants solutions, not research dumps. Gets frustrated when AI suggests traditional ML approaches.

---

## 2. WHAT

### The Ecosystem

```
Noor (custom LLM)
  → AdiTurbo (TQ3 quantization engine, fork of llama.cpp)
    → ADE (Artificial Divine Engine — 6-program verification pipeline)
      → Eyla AIOS (Tauri + React + LangGraph agent OS)
        → 130M+ users via Grameenphone/Robi in Bangladesh
```

### Noor = The Model

Custom sparse MoE language model. Three variants:

| Spec | Noor-Edge | Noor-Pro | Noor-Max |
|------|-----------|----------|----------|
| Target | Phone, RPi, browser | Laptop (24GB) | Workstation (64GB+) |
| Total params | 2.8B | 12B | 28B |
| Active params | ~1B | ~3B | ~4B |
| d_model | 1,024 | 2,048 | 2,816 |
| Layers | 24 | 32 | 36 |
| Q/KV heads | 8/2 | 16/4 | 16/8 |
| MoE | **No** — PLE instead | 32 experts, top-4 + shared | 64 experts, top-4 + shared |
| Expert FFN | N/A | 512 | 512 |
| Attention | Sliding only (512) | 5:1 sliding/global | 5:1 sliding/global |
| Context | 32K | 128K | 256K |
| Vocab | 64,000 (Borno BPE) | same | same |
| TQ3 size | ~1.2 GB | ~5 GB | ~12 GB |

### Eyla = The OS

Offline-first agentic AI OS for Bangladesh. Tauri 2.10 + React 19 + Node.js LangGraph sidecar.
- Code at: `/Users/adi/Eyla-Bangladesh-First-Agentic-OS/` (v2.0.0)
- Features: Bangla voice (STT+TTS), sub-agents, SQLite memory (FTS5), APO+SFT self-improvement
- Model routing via Ollama: Qwen3 (simple/complex), Tiger-Gemma-9B (Bangla)

### AdiTurbo = The Quantization Engine

Fork of llama.cpp with custom TQ3_0 quantization (3.06 bpw).
- Code at: `/Users/adi/PAIA_V1/llama-cpp-turboquant/`
- Custom Metal GPU kernels + ARM NEON kernels
- imatrix importance-weighted scale optimization
- MoE-aware tensor routing (gate tensors get Q8_0 precision)

### ADE = The Verification Pipeline

6 programs: P1 State, P2 Instruments (tools), P3 Memory, P4 Experience, P5 Competence Boundary, P6 Confidence.

---

## 3. THE RULES (NON-NEGOTIABLE)

```
1. NEVER add Python, PyTorch, or CUDA toolkit dependencies.
2. NEVER use Adam/AdamW optimizer — Muon + SMEBU only.
3. NEVER change the architecture without explicit approval.
4. NEVER suggest "just use HuggingFace/transformers" — Noor is a custom Rust stack.
5. NEVER suggest LoRA/QLoRA/PEFT — not implemented in Burn, not on the roadmap.
6. NEVER suggest MergeKit/model merging tools — they require PyTorch.
7. All code must compile on macOS (Accelerate) AND Windows/Linux (OpenBLAS).
8. Checkpoint format: Burn CompactRecorder (.mpk) for training, GGUF for inference/deployment.
9. The deployment chain is: Train → GGUF → AdiTurbo quantize → ADE wrap → Eyla serve.
```

---

## 4. THE STACK

### Training Stack

| Component | Implementation | Location |
|-----------|---------------|----------|
| Framework | **Burn** (Rust ML framework with autodiff) | `crates/noor-burn/` |
| Reference impl | Pure Rust (no autodiff, for inference) | `crates/noor-core/` |
| Tokenizer | **Borno** (বর্ণ) — 64K BPE, Bangla-native | `crates/borno/` |
| Data format | Binary shards: `[seq_len: u32][token_ids: u32...]` | `crates/borno/src/bin/borno_shard.rs` |
| Data loader | Memory-mapped, zero-copy | `crates/noor-burn/src/data.rs` |
| Optimizer | AdamW placeholder (Muon not yet in Burn) | `crates/noor-burn/src/training.rs` |
| Checkpoints | Burn CompactRecorder (.mpk), ~821MB per Edge checkpoint | `checkpoints/` |
| Configs | TOML files per variant | `config/*.toml` |

### Borno Tokenizer

- 64,000 vocab BPE with byte fallback
- Bangla grapheme-cluster-aware pretokenization (`bangla.rs`)
- NFC normalization for Bangla
- Special tokens: bos(256), eos(257), pad(258), unk(259), user(260), assistant(261), system(262), tool_call(263), tool_result(264), think(265), /think(266), memory(267), /memory(268), code(269), /code(270)
- Trained on ~1GB corpus (400MB English + 350MB Bangla + 250MB code)
- Encoding speed: 2-5M tokens/sec (backtracking BPE via rs-bpe)

### Data Pipeline

```
Raw (Parquet/JSONL)
  → parquet_to_jsonl.py (format conversion)
  → borno-shard (dedup → tokenize → pack into .bin shards)
  → DataLoader (mmap → batch → training loop)
```

- Dedup: FNV-1a hash of (length + first 200 bytes + last 100 bytes), persistent across runs
- Shard format: `[seq_len: u32][token_id: u32]...` per sequence, 500K tokens per shard
- Supports formats: text, instruction, reasoning, hermes, bangla

### BLAS Dispatch

- macOS: Apple Accelerate (AMX-optimized)
- Windows/Linux: OpenBLAS
- Fallback: CBLAS > Zig NEON > pure-Rust tiled matmul
- `cblas` feature auto-detected by `build.rs` — never set manually

---

## 5. TRAINING STATUS (2026-04-08)

### Completed

| Phase | Data | Steps | Loss | Where | Checkpoint |
|-------|------|-------|------|-------|------------|
| Phase 1 — Base | 575M tokens, 1,154 shards | 20,000 | 7.4→3.0 | RunPod A100 | `Adiuk/noor-edge-checkpoints/edge_a100_20k/noor_final.mpk` |

### In Progress

| Phase | Data | Steps | Loss | Where |
|-------|------|-------|------|-------|
| Phase 2 — Bangla CC | ~2B tokens, 17,231 shards | 25,000 (target) | 3.0→2.6 (ongoing) | RunPod A100 |

### Pending

| Phase | Data | Shards | Purpose |
|-------|------|--------|---------|
| Phase 3 — Reasoning | DeepSeek R1 | 121 shards (229MB) | Reasoning capability |
| Phase 4 — Instruction | OpenHermes + Bangla + Opus reasoning | Multiple sets | Instruction following |

### Training Configs

**Phase 1** (`config/edge_a100.toml`):
- lr_max=1e-4, lr_min=1e-5, warmup=200, total=20K, batch=2048 tokens, checkpoint_every=500

**Phase 2** (`config/edge_a100_phase2.toml`):
- lr_max=3e-5, lr_min=1e-6, warmup=100, total=25K (lower LR for continued training)

### Data Sources

**On ADISDRIVE (external drive):**
- 290K verified JSONL lines (~556MB) — wikipedia, alpaca, hellaswag, bengali_empathetic, openhermes, sharegpt4, triviaqa, platypus, mmlu, dolly, gsm8k, arc
- 16GB EYLA_MASTER_TRAINING.jsonl
- Golden datasets organized: BENGALI/, ENGLISH/, CONVERSATIONAL/, MULTILINGUAL/

**On HuggingFace (`Adiuk/noor-edge-checkpoints`):**
- Base training shards (1,154)
- Distillation shards: deepseek_r1 (121), openhermes (205), opus_reasoning (5), bangla (1)
- Bangla CC chunks (59 chunks → 17,231 shards when extracted)
- Checkpoints: kaggle_20k, edge_a100_20k

---

## 6. ARCHITECTURE SOURCES

Noor combines techniques from:

| Source | What Noor Takes |
|--------|----------------|
| **Trinity** (Arcee AI) | SMEBU bias balancing, sigmoid routing, sandwich norm, Muon optimizer |
| **AttnRes** (Kimi/Moonshot) | Block-level learned residuals, pseudo-query depth attention |
| **Gemma 4** (Google, April 2026) | Parallel dense+MoE, tiny experts, 5:1 sliding/global, PLE, per-expert scales |
| **Kimi K2** | MuonClip (Muon + QK-Clip), self-training loop |
| **MIT** | Layer pruning (50% removable), LASER SVD |

### Key Differences from Gemma 4

| Feature | Gemma 4 | Noor |
|---------|---------|------|
| Routing | Softmax + top-k | **Sigmoid + SMEBU** (from Trinity) |
| Attention scaling | 1.0 (QK-norm replaces 1/sqrt(d)) | 1/sqrt(d) (standard) |
| KV sharing | 20/35 layers share KV (Edge) | Not yet |
| Norm depth-scaling | No | **Yes** (from Trinity) |
| Vocab | 262,144 | 64,000 (Borno, Bangla-native) |

---

## 7. HARDWARE CONSTRAINTS

| Machine | GPU | VRAM | Role |
|---------|-----|------|------|
| Mac M4 | Metal | 18GB usable (24GB total) | Dev, research, quantization, inference |
| Desktop | RTX 3060 | 12GB | Training (small runs), inference |
| Kaggle | 2x T4 | 32GB total | Free training (30 hrs/week, 12hr sessions) |
| RunPod | A100 80GB | 80GB | Serious training ($1.49/hr) |

**Key constraints:**
- No FP8 on Apple Silicon — use BF16 (M4) / FP16 (T4)
- 75% Metal RAM cap — 18GB usable on 24GB Mac
- Kaggle: 12hr session limit, need robust checkpointing
- RunPod: pay-per-hour, budget-sensitive (~$10-50 per run)

---

## 8. NEXT STEPS (PRIORITY ORDER)

### Immediate (this week)
1. ✅ Phase 1 base training complete
2. 🔄 Phase 2 Bangla CC training (running)
3. Phase 3 reasoning training (after Phase 2)
4. Phase 4 instruction tuning (after Phase 3)
5. Upload all checkpoints + logs to HF

### Short-term (Borno v2)
1. Collect 30GB corpus (Mac, free)
2. Switch BPE → Unigram LM
3. Increase vocab 64K → 80K (25K Bangla + 30K English + 12K code + 13K other)
4. Add language tags + xVal number tokens
5. Re-shard all data, retrain from scratch

### Medium-term (Architecture)
1. Test attention scaling = 1.0 with QK-norm
2. Add KV sharing across layers for Edge (saves ~40% KV memory)
3. Consider increasing experts from 32 → 64-128 for Pro
4. Implement Muon optimizer in Burn (replace AdamW placeholder)
5. Implement SMEBU in training loop

### Research Directions (validated, not committed)
- **PCsInit** — PCA-based embedding initialization from training data
- **Wave Network** — complex-valued token encoding (2.4M matching 100M BERT)
- **Hopfield direct storage** — write knowledge into attention weights without gradient descent
- **xVal** — continuous number encoding
- **WaveletGPT** — multi-scale wavelet structure in embeddings (2x faster pretraining)
- **NSLLM** — spike-based inference for edge deployment (19.8x energy efficiency)

---

## 9. FILE MAP

```
noor/
├── CLAUDE.md                          # Agent instructions (rules, build, train)
├── KNOWLEDGE_GRAPH.md                 # THIS FILE — full project context
├── Cargo.toml                         # Rust workspace
├── config/
│   ├── edge.toml                      # Edge model config
│   ├── edge_a100.toml                 # Phase 1 training config
│   ├── edge_a100_phase2.toml          # Phase 2 training config
│   ├── pro.toml                       # Pro model config
│   ├── max.toml                       # Max model config
│   └── proxy*.toml                    # Test model configs
├── crates/
│   ├── borno/                         # Tokenizer
│   │   ├── src/encoder.rs             # BPE encoding (rs-bpe backtracking)
│   │   ├── src/bangla.rs              # Bangla grapheme cluster rules
│   │   ├── src/pretokenize.rs         # Script-aware segmentation
│   │   ├── src/vocab.rs               # Vocabulary constants + special tokens
│   │   ├── src/trainer.rs             # BPE training
│   │   └── src/bin/borno_shard.rs     # Streaming JSONL → binary shards
│   ├── noor-core/                     # Reference implementation (inference)
│   │   ├── src/model.rs               # Full model assembly
│   │   ├── src/config.rs              # ModelConfig
│   │   ├── src/gguf.rs                # GGUF v3 reader/writer
│   │   ├── src/backward.rs            # Manual backward pass
│   │   └── src/layers/               # All layer implementations
│   │       ├── attention.rs           # GQA with sliding window
│   │       ├── moe.rs                # Sigmoid router + expert dispatch
│   │       ├── ple.rs                # Per-Layer Embeddings
│   │       ├── ffn.rs                # GeGLU + SwiGLU
│   │       ├── norm.rs               # RMSNorm + SandwichNorm
│   │       ├── rope.rs               # RoPE + p-RoPE
│   │       ├── block.rs              # MoE block + PLE block
│   │       ├── parallel_ffn.rs       # Dense + MoE parallel combiner
│   │       └── embedding.rs          # Token embeddings
│   └── noor-burn/                     # Burn backend (training)
│       ├── src/model.rs               # Burn model
│       ├── src/training.rs            # Training loop
│       ├── src/data.rs                # DataLoader (mmap shards)
│       └── src/bin/
│           ├── noor_train.rs          # Training binary
│           └── noor_infer.rs          # Inference binary
├── scripts/
│   ├── parquet_to_jsonl.py            # Data format conversion
│   ├── stream_parquet_to_shard.sh     # Direct parquet → shard pipeline
│   ├── generate_bangla_synthetic.py   # Synthetic Bangla data generation
│   ├── extract_and_shard.sh           # Full data pipeline orchestration
│   └── build_proxy_dataset.sh         # Proxy dataset builder
├── checkpoints/
│   └── tokenizer/                     # Borno trained artifacts
│       ├── borno_encoder.bin          # Compiled encoder (bincode)
│       └── tokenizer.json             # HF format
├── data/
│   ├── noor_training/shards/          # Phase 1 base training shards
│   └── distillation/shards/           # Phase 2-4 shards
│       ├── bangla_cc/                 # 17,231 Bangla CC shards (symlink to volume)
│       ├── deepseek_r1/               # 121 reasoning shards
│       ├── openhermes/                # 205 instruction shards
│       ├── opus_reasoning/            # 5 Opus reasoning shards
│       └── bangla/                    # 1 Bangla instruction shard
└── docs/
    ├── IMPLEMENTATION_PLAN.md         # Zero-drift implementation plan (Phases 0-10)
    ├── 2026-04-06-noor-architecture-design.md  # Architecture spec
    └── superpowers/specs/             # Component design docs
```

---

## 10. COMMON MISTAKES TO AVOID

| Mistake | Why It's Wrong | What To Do Instead |
|---------|---------------|-------------------|
| Suggest PyTorch/HuggingFace tools | Noor is pure Rust/Burn | Work within the Burn ecosystem |
| Suggest LoRA/QLoRA/PEFT | Not implemented, not planned | Use phased continued pretraining |
| Suggest MergeKit/model merging | Requires PyTorch, wrong checkpoint format | Task vectors possible in Rust if needed |
| Suggest "just use a 7B model" | Noor IS the model, custom-built | Improve Noor, don't replace it |
| Ignore the training budget | RunPod costs real money ($1.49/hr) | Always estimate cost before suggesting training runs |
| Assume standard transformer | Noor has PLE (Edge), sigmoid routing, SMEBU, sandwich norms | Read the architecture spec first |
| Suggest data filtering tools (FineWeb-Edu etc) | Borno-shard already has dedup + quality filtering | Improve the existing pipeline |
| Treat Edge as "small/weak" | 2.8B/430M active is a serious model training on A100 | Respect the architecture |
| Forget about Bangla | Noor is Bangla-first | Every suggestion must consider Bangla impact |
| Suggest solutions that need gradient descent | Sometimes direct weight manipulation is preferred | Consider analytical/closed-form alternatives |
