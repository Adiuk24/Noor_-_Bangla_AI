# Noor Training Efficiency — Research & Implementation Spec

**Goal**: 80%+ faster training through data optimization + compute optimization + architecture innovations.

---

## 1. Data-Level Optimizations (implemented)

### Deduplication (borno-shard --dedup-file)
- FNV-1a hash of (length + first 200 bytes + last 100 bytes)
- Persistent across runs — later sources skip entries already seen
- Priority ordering: high-quality curated first, bulk last

### Quality Filtering (proxy dataset)
- Only use curated sources for proxy: SOUL phases, Phase1_Phase2 benchmarks, Bengali golden
- Skip 16GB MASTER (superset, mostly duplicates) and 18GB ling_coder (overkill for proxy)
- Target: ~800M-1B BPE tokens for 10K-step proxy training

### Future: Superfiltering (Phase 2+)
- Score all 44GB with Noor-Proxy using IFD (Instruction-Following Difficulty)
- Keep top 20% → equivalent quality to full dataset
- Paper: arXiv:2402.00530 — 5% of data matches full-data performance

### Future: SemDeDup (Phase 2+)  
- Embed all data with proxy model, k-means cluster, remove semantic near-duplicates
- Paper: arXiv:2303.09540 — remove 50% with minimal loss
- Implementation: straightforward in Rust with our own model for embeddings

---

## 2. Compute-Level Optimizations (for Windows Claude)

### GPU-Resident Tensors (40s → 15s/step)
- Keep all model weights as CudaSlice<f32> on RTX 3060 VRAM
- Proxy = ~1.1GB f32, fits in 12GB with room for gradients + optimizer
- Eliminate PCIe transfers per matmul call

### Skip Wasted grad_input (saves 50% of backward)
- Current backward computes grad_input + grad_weight, but only grad_weight is used
- New function: compute_grad_weight_only(grad_output, input) = single matmul

### Factored Output Projection (3x reduction on biggest matmul)
- Replace (768, 64000) with (768, 256) × (256, 64000) 
- 100B FLOPs → 34B FLOPs
- Standard technique: GPT-2, T5 use bottleneck projections

### Skip NS for Embedding/Output
- Newton-Schulz too expensive on 768×64000 matrices
- Use plain SGD+momentum for embedding and output_proj only
- NS for ~100 internal weight matrices (where it matters)

### FP16 Mixed Precision (2x throughput on tensor cores)
- RTX 3060 Ampere: 25.4 TFLOPS FP16 vs 12.7 TFLOPS FP32
- Forward/backward in FP16, loss accumulation in FP32

### Flash Attention (future)
- CUDA PTX kernels via cudarc for RTX 3060
- Metal shaders via objc2-metal for M4
- Paper: Dao-AILab Flash Attention 2

---

## 3. Architecture-Level Optimizations (Phase 2+)

### Mixture of Depths (MoD) — arXiv:2404.02258
- Top-k router per layer decides which tokens compute (rest skip via residual)
- 50% FLOP reduction with same quality
- Combines with MoE: which experts (MoE) AND whether to compute (MoD)
- Static compute graph — hardware-friendly

### G_stack Progressive Training — arXiv:2405.15319
- Start with half-depth model, train, duplicate layers, continue
- 54% fewer tokens to converge at 7B scale
- For Noor: start Edge as 12-layer, duplicate to 24 midway

### Multi-Token Prediction — arXiv:2404.19737
- Predict 4 future tokens per step instead of 1
- +12% HumanEval, +17% MBPP, 3x inference speedup
- 4 extra linear heads + modified cross-entropy loss

### Token Merging (MrT5) — arXiv:2410.20771
- Learned delete gate merges redundant tokens after initial layers
- Up to 75% sequence length reduction
- Language-adaptive: learns different compression for Bangla vs English

---

## 4. Stack Evaluation

### Keep (proven, working)
- Zig NEON kernels for CPU
- cudarc + cuBLAS for CUDA matmul
- Accelerate for macOS BLAS

### Evaluate (high potential)
- CubeCL/Burn: write Rust kernels, JIT to CUDA/Metal/Vulkan
- Metal Flash Attention: MIT-licensed, Rust-callable
- Metal 4 (fall 2026): native ML primitives from Rust

### Skip (poor fit)
- Mojo: Python-first, no Rust interop
- FP8 training: needs Hopper+ hardware
- Zig GPU backends: too experimental
- XLA from Rust: massive dependency

---

## 5. Training Timeline (with optimizations)

### Proxy (Phase 0): ~10-14 hours on RTX 3060
- 10K steps × 3.5-5s/step
- ~800M BPE tokens from quality-filtered dataset
- Validates: SMEBU, routing, parallel dense+MoE, sandwich norm

### Edge (Phase 2): ~1 week on RTX 3060
- ~2B tokens from expanded dataset (add NQ, mixture_thoughts, more Bengali)
- G_stack: start 12-layer → grow to 24
- MoD: 50% FLOP reduction

### Pro (Phase 3): ~4 weeks on RTX 3060 (or burst on RunPod A100)
- ~3B tokens, full 44GB dataset (Superfiltered + SemDeduped → ~8GB gold)
- Multi-token prediction heads
- Full MoE + MoD

---

## Key Research Papers

| Technique | Paper | Key Result |
|-----------|-------|------------|
| Superfiltering | arXiv:2402.00530 | 5% of data = full performance |
| SemDeDup | arXiv:2303.09540 | Remove 50%, no quality loss |
| DoReMi data mixing | NeurIPS 2023 | 2.6x speedup from optimal mixing |
| Mixture of Depths | arXiv:2404.02258 | 50% FLOP reduction |
| G_stack | arXiv:2405.15319 | 54% fewer tokens |
| Multi-token prediction | arXiv:2404.19737 | +12% code, 3x inference |
| Token merging (MrT5) | arXiv:2410.20771 | 75% sequence reduction |
| CompAct activations | arXiv:2410.15352 | 30% memory savings |
| Flash Attention | Dao-AILab | 2-10x faster attention |
| Byte Latent Transformer | arXiv:2412.09871 | 50% FLOP savings (v2) |
| CubeCL matmul | burn.dev | Near-cuBLAS from Rust |
| Metal Flash Attention | philipturner/mfa | M4 GPU training |
