#!/usr/bin/env bash
# =============================================================================
# Noor Full Architecture POC — RunPod A100 80GB
# Run this ONCE after SSH into RunPod. It does everything:
#   1. Setup (Rust, clone repo, download data) — ~10 min
#   2. Train Proxy MoE 0.5B (full arch validation) — ~2-3 hr
#   3. Train Edge 2.8B phased (Bangla CC → reasoning → instruction) — ~8-10 hr
#   4. Upload checkpoints to HuggingFace
# =============================================================================
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
WORKSPACE="/workspace"
NOOR="$WORKSPACE/noor"
HF_USER="Adiuk"
HF_DATA_REPO="Adiuk/noor-training-data"
HF_CKPT_REPO="Adiuk/noor-edge-checkpoints"

echo "============================================"
echo " Noor Training — A100 80GB Setup"
echo " $(date)"
echo "============================================"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'checking...')"

# --- Step 1: Install Rust ---
if ! command -v cargo &>/dev/null; then
    echo "[1/6] Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "[1/6] Rust already installed"
    source "$HOME/.cargo/env" 2>/dev/null || true
fi

# --- Step 2: Clone repo ---
if [ ! -d "$NOOR" ]; then
    echo "[2/6] Cloning noor repo..."
    cd "$WORKSPACE"
    git clone https://github.com/Adiuk24/Noor_-_Bangla_AI.git noor
else
    echo "[2/6] Repo exists, pulling latest..."
    cd "$NOOR" && git pull origin main || true
fi
cd "$NOOR"

# --- Step 3: Install HF CLI + download data ---
echo "[3/6] Downloading training data from HuggingFace..."
pip install -q huggingface_hub zstd 2>/dev/null

python3 -c "
from huggingface_hub import snapshot_download
import os

# Download base training shards
print('  Downloading base training shards...')
snapshot_download(
    repo_id='$HF_DATA_REPO',
    repo_type='dataset',
    local_dir='$NOOR/data/hf_data',
    allow_patterns=['shards/*.bin', 'tokenizer/*'],
)

# Download distillation shards
print('  Downloading distillation shards...')
snapshot_download(
    repo_id='$HF_DATA_REPO',
    repo_type='dataset',
    local_dir='$NOOR/data/hf_data',
    allow_patterns=['distillation/**'],
)

# Download Edge checkpoint
print('  Downloading Edge checkpoint...')
snapshot_download(
    repo_id='$HF_CKPT_REPO',
    local_dir='$NOOR/checkpoints/edge_kaggle',
    allow_patterns=['noor_final.mpk'],
)
print('  Data download complete!')
"

# Set up data directories
mkdir -p data/noor_training/shards
if [ -d data/hf_data/shards ]; then
    ln -sf "$NOOR/data/hf_data/shards"/*.bin data/noor_training/shards/ 2>/dev/null || true
fi

# Decompress bangla_cc if archive exists
BANGLA_ARCHIVE="data/hf_data/distillation/bangla_cc_shards.tar.zst"
if [ -f "$BANGLA_ARCHIVE" ]; then
    echo "  Decompressing bangla_cc shards..."
    mkdir -p data/distillation/shards
    cd data/distillation/shards
    zstd -d "$NOOR/$BANGLA_ARCHIVE" -c | tar xf -
    cd "$NOOR"
    echo "  Bangla CC shards ready: $(ls data/distillation/shards/bangla_cc/ | wc -l) files"
fi

# Link distillation shards
for subdir in deepseek_r1 openhermes opus_reasoning bangla; do
    if [ -d "data/hf_data/distillation/$subdir" ]; then
        mkdir -p "data/distillation/shards/$subdir"
        ln -sf "$NOOR/data/hf_data/distillation/$subdir"/*.bin "data/distillation/shards/$subdir/" 2>/dev/null || true
    fi
done

echo "  Data ready:"
for d in data/noor_training/shards data/distillation/shards/*/; do
    [ -d "$d" ] && echo "    $d: $(ls "$d"/*.bin 2>/dev/null | wc -l) shards"
done

# --- Step 4: Build with CUDA ---
echo "[4/6] Building noor-burn with CUDA..."
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
cargo build --release -p noor-burn --features cuda --bin noor-train 2>&1 | tail -5
cargo build --release -p noor-burn --features cuda --bin noor-infer 2>&1 | tail -3
echo "  Build complete!"

# --- Step 5: Train Proxy MoE (full architecture validation) ---
echo ""
echo "============================================"
echo " PHASE 1: Proxy MoE 0.5B — Full Arch POC"
echo " $(date)"
echo "============================================"

mkdir -p checkpoints/proxy

cargo run --release -p noor-burn --features cuda --bin noor-train -- \
    --config config/proxy_a100.toml \
    --data data/noor_training/shards \
    --checkpoint-dir checkpoints/proxy \
    2>&1 | tee training_proxy.log

echo ""
echo "Proxy training complete! $(date)"
echo "Uploading proxy checkpoints..."

python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='checkpoints/proxy',
    repo_id='$HF_CKPT_REPO',
    path_in_repo='proxy/',
)
print('Proxy checkpoints uploaded!')
"

# --- Step 6: Phased Edge Training ---
echo ""
echo "============================================"
echo " PHASE 2: Edge 2.8B — Phased Training"
echo " $(date)"
echo "============================================"

mkdir -p checkpoints/edge_a100

# Phase 2a: Continue base training on noor_training shards
echo "--- Phase 2a: Base training (resume from Kaggle) ---"
cargo run --release -p noor-burn --features cuda --bin noor-train -- \
    --config config/edge_a100.toml \
    --data data/noor_training/shards \
    --checkpoint-dir checkpoints/edge_a100 \
    --resume checkpoints/edge_kaggle/noor_final.mpk \
    2>&1 | tee training_edge_base.log

# Phase 2b: Bangla CC
if [ -d data/distillation/shards/bangla_cc ] && [ "$(ls data/distillation/shards/bangla_cc/*.bin 2>/dev/null | wc -l)" -gt 0 ]; then
    echo ""
    echo "--- Phase 2b: Bangla CC training ---"
    LATEST_CKPT=$(ls -t checkpoints/edge_a100/noor_step_*.mpk 2>/dev/null | head -1)
    [ -z "$LATEST_CKPT" ] && LATEST_CKPT="checkpoints/edge_a100/noor_final.mpk"

    cargo run --release -p noor-burn --features cuda --bin noor-train -- \
        --config config/edge_a100.toml \
        --data data/distillation/shards/bangla_cc \
        --checkpoint-dir checkpoints/edge_a100 \
        --resume "$LATEST_CKPT" \
        2>&1 | tee training_edge_bangla.log
fi

# Phase 2c: Reasoning (deepseek_r1)
if [ -d data/distillation/shards/deepseek_r1 ]; then
    echo ""
    echo "--- Phase 2c: Reasoning training ---"
    LATEST_CKPT=$(ls -t checkpoints/edge_a100/noor_step_*.mpk 2>/dev/null | head -1)
    [ -z "$LATEST_CKPT" ] && LATEST_CKPT="checkpoints/edge_a100/noor_final.mpk"

    cargo run --release -p noor-burn --features cuda --bin noor-train -- \
        --config config/edge_a100.toml \
        --data data/distillation/shards/deepseek_r1 \
        --checkpoint-dir checkpoints/edge_a100 \
        --resume "$LATEST_CKPT" \
        2>&1 | tee training_edge_reasoning.log
fi

# Phase 2d: Instruction (openhermes)
if [ -d data/distillation/shards/openhermes ]; then
    echo ""
    echo "--- Phase 2d: Instruction training ---"
    LATEST_CKPT=$(ls -t checkpoints/edge_a100/noor_step_*.mpk 2>/dev/null | head -1)
    [ -z "$LATEST_CKPT" ] && LATEST_CKPT="checkpoints/edge_a100/noor_final.mpk"

    cargo run --release -p noor-burn --features cuda --bin noor-train -- \
        --config config/edge_a100.toml \
        --data data/distillation/shards/openhermes \
        --checkpoint-dir checkpoints/edge_a100 \
        --resume "$LATEST_CKPT" \
        2>&1 | tee training_edge_instruct.log
fi

# Upload final checkpoints
echo ""
echo "============================================"
echo " Uploading final checkpoints to HuggingFace"
echo " $(date)"
echo "============================================"

python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='checkpoints/edge_a100',
    repo_id='$HF_CKPT_REPO',
    path_in_repo='edge_a100/',
)
print('All checkpoints uploaded!')
"

echo ""
echo "============================================"
echo " ALL DONE! $(date)"
echo "============================================"
echo "Logs:"
echo "  training_proxy.log"
echo "  training_edge_base.log"
echo "  training_edge_bangla.log"
echo "  training_edge_reasoning.log"
echo "  training_edge_instruct.log"
echo ""
echo "Checkpoints on HF: https://huggingface.co/$HF_CKPT_REPO"
