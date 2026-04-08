# Noor — AI Agent Instructions

## FIRST: Read the Knowledge Graph
**Before doing ANYTHING, read `KNOWLEDGE_GRAPH.md` in this repo root.** It contains the full project context: architecture, stack, rules, training status, file map, and common mistakes to avoid. This prevents context drift.

## Project
Noor is a custom sparse MoE language model built in Rust (no Python, no PyTorch).
- Knowledge graph: `KNOWLEDGE_GRAPH.md`
- Architecture spec: `docs/2026-04-06-noor-architecture-design.md`
- Implementation plan: `docs/IMPLEMENTATION_PLAN.md`

## Build

```bash
# Install OpenBLAS first (Windows):
#   vcpkg install openblas:x64-windows
#   OR download pre-built zip from https://github.com/OpenMathLib/OpenBLAS/releases
#   Then: set OPENBLAS_PATH=C:\path\to\openblas\lib
#
# macOS: no setup needed — Apple Accelerate is always available.
# Linux: sudo apt install libopenblas-dev   (or equivalent)

cargo build --release -p noor-cli
```

## Training

```bash
# Preprocess data
cargo run --release -p noor-cli -- preprocess \
    --input data/raw/text.txt \
    --output data/train/ \
    --vocab-size 64000

# Train proxy model (0.5B — ~25 min on i7-14700K with OpenBLAS)
cargo run --release -p noor-cli -- train \
    --config config/proxy.toml \
    --data data/train/ \
    --steps 1000 \
    --checkpoint-dir checkpoints/

# Train Edge model (2.8B — longer run)
cargo run --release -p noor-cli -- train \
    --config config/edge.toml \
    --data data/train/ \
    --checkpoint-dir checkpoints/
```

## Rules
1. NEVER add Python, PyTorch, or CUDA toolkit dependencies.
2. NEVER use Adam/AdamW optimizer — Muon + SMEBU only.
3. NEVER change the architecture without explicit approval.
4. Checkpoint every 500 steps to GGUF format.
5. Push checkpoints to the repo after each training session.
6. All code must compile on both macOS (Accelerate) and Windows/Linux (OpenBLAS).

## BLAS / Kernel dispatch
- `cblas` feature is set automatically by `build.rs` — do NOT set it manually in Cargo.toml or via `--features`.
- On macOS: links `Accelerate.framework` (AMX-optimised).
- On Windows/Linux: links `openblas` (set `OPENBLAS_PATH` if the linker cannot find it).
- Fallback priority: CBLAS > Zig NEON (`zig_kernels` feature) > pure-Rust tiled matmul.

## Tokenizer
Borno (বর্ণ) is the 64K BPE tokenizer at `crates/borno/`.
Trained vocab at `checkpoints/tokenizer/`.
