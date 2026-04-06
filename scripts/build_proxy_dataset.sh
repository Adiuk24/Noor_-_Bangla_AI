#!/bin/bash
# Build lean, quality-filtered proxy training dataset.
# Only high-value curated sources. Deduplicated. BPE tokenized.
# Target: ~800M-1B tokens (enough for 10K steps × 64K batch + margin)
#
# For proxy validation only. Edge/Pro training will use the full 44GB pipeline.

set -euo pipefail

ENCODER="/Users/adi/noor/checkpoints/tokenizer/borno_encoder.bin"
SHARD_BIN="/Users/adi/noor/target/release/borno-shard"
SHARD_DIR="/Users/adi/noor/data/noor_training/shards"
DEDUP_FILE="/Users/adi/noor/data/noor_training/dedup_hashes.bin"
DRIVE="/Volumes/ADISDRIVE"
CTX=2048
SHARD_SIZE=500000

for f in "$ENCODER" "$SHARD_BIN"; do
    if [ ! -f "$f" ]; then echo "ERROR: $f not found"; exit 1; fi
done
if [ ! -d "$DRIVE" ]; then echo "ERROR: ADISDRIVE not mounted"; exit 1; fi

echo "Clearing old shards..."
rm -f "$SHARD_DIR"/shard_*.bin
rm -f "$DEDUP_FILE"
mkdir -p "$SHARD_DIR"

get_next_shard() {
    local max
    max=$(ls "$SHARD_DIR"/shard_*.bin 2>/dev/null | sort -V | tail -1 | grep -o '[0-9]\{4\}' | tail -1 || echo "")
    if [ -n "$max" ]; then
        echo $((10#$max + 1))
    else
        echo "0"
    fi
}

run_shard_dir() {
    local label="$1"
    local format="$2"
    local dir="$3"
    local start
    start=$(get_next_shard)

    local files
    files=$(find "$dir" -name "*.jsonl" -not -name "._*" 2>/dev/null | sort)
    if [ -z "$files" ]; then
        echo "  SKIP: $label — no JSONL files"
        return
    fi

    local count
    count=$(echo "$files" | wc -l | tr -d ' ')
    echo ""
    echo "=== $label ($count files, start=$start) ==="

    echo "$files" | xargs cat | "$SHARD_BIN" \
        --encoder "$ENCODER" \
        --output-dir "$SHARD_DIR" \
        --start-index "$start" \
        --context-length "$CTX" \
        --shard-size "$SHARD_SIZE" \
        --format "$format" \
        --dedup-file "$DEDUP_FILE" \
        --min-length 50
}

run_shard_file() {
    local label="$1"
    local format="$2"
    local file="$3"
    local start
    start=$(get_next_shard)

    if [ ! -f "$file" ]; then
        echo "  SKIP: $label — not found"
        return
    fi

    local fsize
    fsize=$(ls -lh "$file" | awk '{print $5}')
    echo ""
    echo "=== $label ($fsize, start=$start) ==="

    "$SHARD_BIN" \
        --encoder "$ENCODER" \
        --output-dir "$SHARD_DIR" \
        --start-index "$start" \
        --context-length "$CTX" \
        --shard-size "$SHARD_SIZE" \
        --format "$format" \
        --dedup-file "$DEDUP_FILE" \
        --min-length 50 \
        < "$file"
}

echo "============================================"
echo "  Noor Proxy Dataset (quality-filtered)"
echo "============================================"

# ── Tier 1: Highest quality curated data ───────────────────

# 1. SOUL phases (148MB) — reasoning, chain-of-thought, coding, culture, math
run_shard_dir "SOUL phases (reasoning/coding/culture)" "text" \
    "$DRIVE/EYLA_SOUL_PHASES"

# 2. Phase1/Phase2 benchmarks (1.1GB) — alpaca, gsm8k, mmlu, hellaswag, etc.
run_shard_dir "Phase1_Phase2 benchmarks" "text" \
    "$DRIVE/Eyla_Training_Data_Golden/Phase1_Phase2_archived_raw"

# 3. Bengali golden — 2 newspaper batches + empathetic + safety + QA (~400MB)
BENGALI_DIR="/tmp/bengali_data/GOLDEN_DATASETS/BENGALI"
if [ -d "$BENGALI_DIR" ]; then
    # Newspaper: only 2 of 8 batches (enough Bangla without dominating)
    echo ""
    echo "=== Bengali newspaper (2 batches, start=$(get_next_shard)) ==="
    for batch in 000 001; do
        BN_FILE="$BENGALI_DIR/kaggle_bangla_newspaper_golden/kaggle_bangla_batch_${batch}.jsonl"
        if [ -f "$BN_FILE" ]; then
            cat "$BN_FILE"
        fi
    done | "$SHARD_BIN" \
        --encoder "$ENCODER" \
        --output-dir "$SHARD_DIR" \
        --start-index "$(get_next_shard)" \
        --context-length "$CTX" \
        --shard-size "$SHARD_SIZE" \
        --format text \
        --dedup-file "$DEDUP_FILE" \
        --min-length 50

    # Empathetic + safety + QA (all of them — small, high quality)
    run_shard_dir "Bengali empathetic/safety/QA" "text" \
        "$BENGALI_DIR/bengali_empathetic_conversations_golden"

    BN_SAFETY="$BENGALI_DIR/bd_shs_golden"
    if [ -d "$BN_SAFETY" ]; then
        run_shard_dir "Bengali safety (BD SHS)" "text" "$BN_SAFETY"
    fi
fi

# 4. Fashion (45MB)
run_shard_file "Fashion phase4_5" "text" \
    "$DRIVE/eyla_master_dataset_phase4_5_fashion.jsonl"

# 5. Artifacts gold (23MB, instruction format)
run_shard_dir "Artifacts gold (reasoning/multilingual)" "instruction" \
    "$DRIVE/Eyla Training Data Golden/artifacts_gold"

# ── Tier 2: Fill remaining budget ──────────────────────────

# 6. EYLA_RAW phases 1-3 (566MB) — already curated
run_shard_dir "EYLA_RAW phases 1-3" "text" \
    "$DRIVE/EYLA_RAW_DATASETS_PHASES_1_TO_3"

# ── Summary ────────────────────────────────────────────────
echo ""
echo "========================================="
TOTAL_SHARDS=$(ls "$SHARD_DIR"/shard_*.bin 2>/dev/null | wc -l | tr -d ' ')
TOTAL_SIZE=$(du -sh "$SHARD_DIR" | cut -f1)
echo "PROXY DATASET COMPLETE"
echo "  Shards: $TOTAL_SHARDS"
echo "  Size: $TOTAL_SIZE"
echo "========================================="
