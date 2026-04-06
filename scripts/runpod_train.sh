#!/usr/bin/env bash
# RunPod training launch script for Noor-Edge 2.8B
# Run on RunPod L40S after data transfer is complete.
set -euo pipefail

export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

WORKSPACE="/workspace/noor"
DATA_DIR="$WORKSPACE/data/noor_training/shards"
CKPT_DIR="$WORKSPACE/checkpoints"
CONFIG="$WORKSPACE/config/edge_runpod.toml"
ARCHIVE="/workspace/noor_shards.tar.zst"

echo "=== Noor Edge 2.8B Training Setup ==="
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"

# Extract data if archive exists
if [ -f "$ARCHIVE" ]; then
    echo "  Extracting training data..."
    cd "$WORKSPACE/data/noor_training"
    rm -rf shards/
    zstd -d "$ARCHIVE" -o /tmp/noor_shards.tar
    tar xf /tmp/noor_shards.tar
    rm /tmp/noor_shards.tar
    echo "  Extracted: $(ls shards/shard_*.bin | wc -l) shards"
    rm "$ARCHIVE"
fi

# Verify data
SHARD_COUNT=$(ls "$DATA_DIR"/shard_*.bin 2>/dev/null | wc -l)
echo "  Shards: $SHARD_COUNT"
if [ "$SHARD_COUNT" -lt 100 ]; then
    echo "ERROR: Not enough shards. Transfer data first."
    exit 1
fi

# Pull latest code and rebuild
cd "$WORKSPACE"
git pull origin main 2>/dev/null || true
echo "  Building with CUDA..."
cargo build --release -p noor-burn --features cuda --bin noor-train 2>&1 | tail -5

# Create checkpoint directory
mkdir -p "$CKPT_DIR"

echo ""
echo "=== Starting Training ==="
echo "  Config: $CONFIG"
echo "  Data: $DATA_DIR"
echo "  Checkpoints: $CKPT_DIR"
echo ""

# Launch training (nohup so it survives SSH disconnect)
nohup cargo run --release -p noor-burn --features cuda --bin noor-train -- \
    --config "$CONFIG" \
    --data "$DATA_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    > "$WORKSPACE/training.log" 2>&1 &

TRAIN_PID=$!
echo "Training started! PID: $TRAIN_PID"
echo "  Monitor: tail -f $WORKSPACE/training.log"
echo "  Kill: kill $TRAIN_PID"
echo ""

# Show first few lines
sleep 5
tail -20 "$WORKSPACE/training.log" 2>/dev/null || echo "Waiting for output..."
