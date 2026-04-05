# Noor (নূর) — Light

A sparse Mixture-of-Experts language model. Nothing traditional.

**No Python. No PyTorch. No CUDA. No Adam. No fixed residuals. No softmax routing.**

## Architecture

Noor combines innovations from Trinity (SMEBU), Kimi K2 (MuonClip, self-training), Gemma 4 (parallel dense+MoE, PLE), and MIT research (layer pruning, LASER SVD) — all built on a custom Mojo+Rust+Zig stack.

| Variant | Active | Total | Target | Size (TQ3) |
|---------|--------|-------|--------|------------|
| **Noor-Edge** | ~1B | 2.8B | Phone, RPi, browser | 1.2 GB |
| **Noor-Pro** | ~3B | 12B | Laptop (24GB M4) | 5 GB |
| **Noor-Max** | ~4B | 28B | Workstation (64GB+) | 12 GB |

### What Makes It Different

- **MuonClip optimizer** — 2x more token-efficient than Adam, 67% less memory, QK-Clip for MoE stability
- **Sigmoid + SMEBU routing** — independent expert scores with momentum bias balancing, zero expert collapse
- **Block AttnRes** — learned pseudo-query residuals replace fixed `y = F(x) + x`
- **Parallel dense + MoE** — every token gets both baseline FFN + expert specialization
- **Sandwich RMSNorm** — depth-scaled pre+post normalization for deep MoE stability
- **Self-training loop** — K2 pattern: rephrase → practice in synthetic environments → actor-critic self-improvement
- **One train, many sizes** — train once, prune with MIT layer pruning + LASER SVD for device variants
- **ADE-wrapped** — P5 Competence Boundary + P6 Confidence. The model knows what it doesn't know.

## Stack

| Language | Role |
|----------|------|
| **Mojo** | Core engine — tensor lib, all layers, forward+backward, optimizers |
| **Zig + C** | Compute kernels — ARM NEON, Apple AMX, Metal GPU shaders |
| **Rust** | Orchestration — training loop, data pipeline, expert SSD offload, CLI |
| **Julia** | Research — architecture prototyping on 0.5B proxy (throwaway) |

## Training

- **Hardware:** M4 24GB (dev) + free Kaggle 2x T4 (training). $0 cloud cost.
- **Data:** High-quality curated + synthetic rephrasing (K2 pattern). ~5.5B tokens total.
- **Strategy:** Progressive growing — train Edge first, grow to Pro by copying weights + adding experts.
- **Checkpoints:** GGUF format. Checkpoint IS the deployable model.

## Ecosystem

```
Noor (this) ── model weights + architecture
    ↓
AdiTurbo ────── TQ3 quantization for edge
    ↓
ADE (P1-P6) ── verification + confidence wrapping
    ↓
Eyla AIOS ──── serves Noor to 130M+ users
```

## Docs

- [Architecture Specification](docs/2026-04-06-noor-architecture-design.md) — complete design document

## Project Structure

```
noor/
├── docs/          # Architecture specs, research notes
├── src/           # NoorTorch framework (Mojo + Rust)
├── kernels/       # NEON/Metal/Zig compute kernels
├── data/          # Training data pipeline
├── scripts/       # Build, benchmark, conversion scripts
├── tests/         # Unit + integration tests
└── config/        # Model configs (dimensions, experts, etc.)
```

## Origin

Bangladesh — Arif Adito | Adioris Tech Ltd

First high-quality bilingual English+Bangla model at this scale.
