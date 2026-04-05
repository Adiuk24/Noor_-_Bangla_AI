# Noor (নূর) — Light

A sparse Mixture-of-Experts language model designed for consumer hardware.

**Architecture:** SMEBU-gated MoE + AttnRes residuals + depth-scaled sandwich norm
**Target:** 24GB Apple M4 (training and inference)
**Purpose:** ADE-native reasoning model for Eyla AIOS
**Origin:** Bangladesh — Adioris Tech Ltd

## Model Variants

| Variant | Active Params | Total Params | Description |
|---------|--------------|-------------|-------------|
| Noor-MoE | 2-3B | 8-12B | Base pre-trained model |
| Noor-Thinking | 2-3B | 8-12B | Reasoning fine-tune |
| Noor-TQ3 | 2-3B | 8-12B | AdiTurbo quantized for deployment |

## Architecture Innovations (vs standard transformer)

1. **AttnRes** — Learnable pseudo-query residuals replace fixed `y = F(x) + x`
2. **SMEBU MoE** — Sigmoid routing + soft-clamped momentum bias prevents expert collapse
3. **Sandwich Norm** — RMSNorm before + after sublayers, depth-scaled
4. **Dense+MoE Hybrid** — First N layers dense, rest MoE for routing stability
5. **SGD+SMEBU Optimizer** — 1x model memory (not 3x like Adam)

## Stack

| Language | Role |
|----------|------|
| Rust | Training framework, autodiff, data pipeline |
| C | NEON/Metal compute kernels (from AdiTurbo) |
| Zig | Custom high-speed math kernels |
| Julia | Architecture research and hyperparameter sweeps |

## Project Structure

```
noor/
├── docs/          # Architecture specs, research notes
├── src/           # Training framework (Rust)
├── kernels/       # NEON/Metal/Zig compute kernels
├── data/          # Training data pipeline
├── scripts/       # Build, benchmark, conversion scripts
├── tests/         # Unit + integration tests
└── config/        # Model configs (dimensions, experts, etc.)
```

## Relationship to PAIA Ecosystem

```
Noor (this) ── the model weights + architecture
    ↓
AdiTurbo ────── quantizes Noor for deployment
    ↓
ADE (P1-P6) ── wraps Noor with verification + confidence
    ↓
Eyla AIOS ──── serves Noor to users via CLI/TUI
```

## Author

Arif Adito | Adioris Tech Ltd
